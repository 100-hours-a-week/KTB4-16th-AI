"""사진 설명문 점수표 — 프롬프트 버전마다 설명이 얼마나 쓸 만한지 숫자로 잰다.

점수는 세 가지다. 판정은 모두 Claude API가 하고, 같은 입력의 응답은 파일에 저장해 다시 쓴다.

- 정보 전달 점수(0~1): 사진을 보지 못하는 Claude가 설명문만 읽고 속성(장소·사진 종류·대상·분위기·
  표정·감정·노래 분위기)을 고르게 한 뒤, 정답(gold_labels.json)과 얼마나 같은지 F1로 센다.
  설명이 사진 정보를 많이 담을수록 높다. 설명을 다른 사진 것과 바꿔 읽힌 점수(기준선)도 함께 낸다.
- 사실 정확도: 사진을 볼 수 있는 Claude가 설명문을 사실 주장으로 쪼개 사진과 대조한다.
  확인되는 주장 ÷ (확인되는 주장 + 틀린 주장). 사진이 흐려 알 수 없는 주장은 뺀다.
- 큰 오류 수: 틀린 주장 중 지어낸 글자, 중심 대상 오인, 빛·색 대오류, 표정 반대 등.

두 버전 이상이면 첫 버전을 기준으로 사진별 짝 비교를 한다(평균 차이, 95% 신뢰구간, 좋아진/나빠진
사진 수, 부호 검정). judgments_<버전>.json이 있으면 Claude 판정이 내 판정과 얼마나 맞는지도 낸다.

실행:
  PYTHONPATH=. .venv/bin/python scripts/photo_eval/score_descriptions.py D E          # 계획만 출력
  PYTHONPATH=. .venv/bin/python scripts/photo_eval/score_descriptions.py D E --run    # API 호출
  --dry-run: 가짜 응답으로 흐름만 확인(비용 없음). 결과: data/photo_eval/scores/
필요: .env의 ANTHROPIC_API_KEY. 호출 수만큼 API 비용이 든다. 사진(짤 포함)이 Anthropic으로 전송된다.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

import anthropic

from app.config import get_settings
from scripts.photo_eval.build_review import descriptions, load_items, load_judgments
from scripts.photo_eval.gold import (
    ATTRS,
    DATA,
    DRAFT_PATH,
    GOLD_PATH,
    LIMITS,
    load_gold,
    load_vocab,
)

SCORES_DIR = DATA / "scores"
SHIFT = 7  # 기준선: 사진 i의 설명 대신 7칸 뒤 사진의 설명을 읽힌다
BOOTSTRAP_ROUNDS = 10_000
CLAIM_VERDICTS = ("supported", "unsupported", "unverifiable")

JUDGE_SYSTEM = """너는 사진 설명문을 검수하는 검수자다. 사진과 설명문이 주어진다.
설명문을 '사진에서 확인할 수 있는 사실 주장' 단위로 나누고, 각 주장을 사진과 대조해 판정한다.
- supported: 사진에서 확인된다.
- unsupported: 사진과 다르거나 사진에 없다.
- unverifiable: 사진이 작거나 흐려서 맞는지 알 수 없다. 확신이 없으면 unsupported가 아니라
  unverifiable로 한다.
색, 대상, 개수, 위치, 동작, 표정, 날씨, 시간대, 사진 속 글자는 사실 주장이다.
"차분한 느낌"처럼 느낌을 말하는 표현은 주장에서 뺀다.
unsupported 중 다음에 해당하면 serious를 true로 한다: 사진에 없는 글자를 지어냄, 사진의
중심 대상을 잘못 말함, 사진 전체의 색이나 빛을 크게 잘못 말함, 사람의 표정이나 감정을 반대로 말함.
JSON만 출력한다.
{"claims": [{"claim": "주장 한 줄", "verdict": "supported", "serious": false}]}"""


def reader_system(vocab: dict[str, list[str]]) -> str:
    lines = [f"- {attr} (최대 {LIMITS[attr]}개): {', '.join(vocab[attr])}" for attr in ATTRS]
    keys = ", ".join(f'"{attr}": []' for attr in ATTRS)
    return (
        "너는 사진 설명문만 읽고 사진의 정보를 고르는 읽기 담당이다. 사진은 볼 수 없다.\n"
        "설명문에 적힌 내용에 근거해서만 고르고, 근거가 없으면 비워 둔다.\n"
        "아래 어휘에서만 고르며, 어휘에 없는 말은 쓰지 않는다.\n"
        + "\n".join(lines)
        + "\n규칙: 사진 종류가 '진짜 사진'이면 대상과 분위기를 고르고 표정·감정은 비운다. "
        "'그림·짤'이면 표정·감정을 고르고 대상과 분위기는 비운다. 장소는 목록에 맞는 곳이 "
        "없으면 비운다. 노래 분위기는 설명문의 내용과 분위기에 어울리는 것을 고른다.\n"
        f"JSON만 출력한다. {{{keys}}}"
    )


# ---------- 순수 계산 (API와 무관) ----------


def f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    hit = len(pred & gold)
    if hit == 0:
        return 0.0
    precision, recall = hit / len(pred), hit / len(gold)
    return 2 * precision * recall / (precision + recall)


def info_score(pred: dict[str, list[str]], gold: dict) -> tuple[float, dict[str, float]]:
    per_attr = {a: f1(set(pred.get(a, [])), set(gold.get(a, []))) for a in ATTRS}
    return statistics.mean(per_attr.values()), per_attr


def parse_json(raw: str) -> dict | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def clean_attributes(raw: dict, vocab: dict[str, list[str]]) -> dict[str, list[str]]:
    """읽기 담당의 답에서 어휘 밖 값을 버리고 최대 개수로 자른다."""
    cleaned = {}
    for attr in ATTRS:
        values = raw.get(attr, [])
        values = values if isinstance(values, list) else []
        picked = []
        for value in values:
            if value in vocab[attr] and value not in picked:
                picked.append(value)
        cleaned[attr] = picked[: LIMITS[attr]]
    return cleaned


def count_claims(raw: dict) -> dict | None:
    claims = raw.get("claims")
    if not isinstance(claims, list):
        return None
    kept = [c for c in claims if isinstance(c, dict) and c.get("verdict") in CLAIM_VERDICTS]
    counts = {v: sum(1 for c in kept if c["verdict"] == v) for v in CLAIM_VERDICTS}
    counts["serious"] = sum(
        1 for c in kept if c["verdict"] == "unsupported" and c.get("serious") is True
    )
    counts["claims"] = [
        {
            "claim": str(c.get("claim", "")),
            "verdict": c["verdict"],
            "serious": c.get("serious") is True,
        }
        for c in kept
    ]
    return counts


def shifted(ids: list[str], shift: int = SHIFT) -> dict[str, str]:
    """사진 id → 기준선에서 대신 읽을 사진 id. 자기 자신이 걸리지 않는다."""
    return {pid: ids[(i + shift) % len(ids)] for i, pid in enumerate(ids)}


def sign_test_p(wins: int, losses: int) -> float:
    """좋아진 사진 수와 나빠진 사진 수만으로 하는 양쪽 정확 부호 검정."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: list[float], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(BOOTSTRAP_ROUNDS)
    )
    return means[int(0.025 * BOOTSTRAP_ROUNDS)], means[int(0.975 * BOOTSTRAP_ROUNDS) - 1]


def paired_compare(base: dict[str, float], new: dict[str, float]) -> dict | None:
    ids = sorted(set(base) & set(new))
    if not ids:
        return None
    diffs = [new[i] - base[i] for i in ids]
    wins = sum(1 for d in diffs if d > 1e-9)
    losses = sum(1 for d in diffs if d < -1e-9)
    low, high = bootstrap_ci(diffs)
    base_mean = statistics.mean(base[i] for i in ids)
    mean_diff = statistics.mean(diffs)
    return {
        "n": len(ids),
        "base_mean": base_mean,
        "new_mean": statistics.mean(new[i] for i in ids),
        "diff": mean_diff,
        "ci": (low, high),
        "relative": mean_diff / base_mean if base_mean else None,
        "wins": wins,
        "losses": losses,
        "ties": len(ids) - wins - losses,
        "p": sign_test_p(wins, losses),
    }


def claim_precision(claims: dict | None) -> float | None:
    """확인되는 주장 ÷ (확인되는 주장 + 틀린 주장). 판정할 주장이 없으면 None."""
    if not claims:
        return None
    checked = claims["supported"] + claims["unsupported"]
    return claims["supported"] / checked if checked else None


def values_by_id(results: list[dict], field: str) -> dict[str, float]:
    return {r["id"]: r[field] for r in results if r.get(field) is not None}


def summarize(results: list[dict]) -> dict:
    scored = [r for r in results if r["info"] is not None]
    judged = [r for r in results if r["supported"] is not None]
    checked = sum(r["supported"] + r["unsupported"] for r in judged)
    controls = [r["control_info"] for r in results if r.get("control_info") is not None]
    summary = {
        "n": len(results),
        "info": statistics.mean(r["info"] for r in scored) if scored else None,
        "control": statistics.mean(controls) if controls else None,
        "attr_f1": {a: statistics.mean(r["attr_f1"][a] for r in scored) for a in ATTRS}
        if scored
        else {},
        "precision": sum(r["supported"] for r in judged) / checked if checked else None,
        "wrong_per_description": statistics.mean(r["unsupported"] for r in judged)
        if judged
        else None,
        "serious_total": sum(r["serious"] for r in judged),
        "serious_descriptions": sum(1 for r in judged if r["serious"] > 0),
        "judged": len(judged),
        "chars": statistics.median(r["chars"] for r in results) if results else None,
    }
    control, info = summary["control"], summary["info"]
    summary["info_over_control"] = (
        (info - control) / (1 - control) if info is not None and control not in (None, 1) else None
    )
    return summary


def manual_agreement(results: list[dict], manual: dict[str, dict]) -> dict:
    """Claude 판정과 사람(내) 판정이 같은 사진 수. 둘 다 판정이 있는 사진만 센다."""
    both = [r for r in results if r["supported"] is not None and r["id"] in manual]
    serious = [(r["serious"] > 0, manual[r["id"]]["level"] == "중대") for r in both]
    any_wrong = [(r["unsupported"] > 0, manual[r["id"]]["level"] != "없음") for r in both]

    def stats(pairs: list[tuple[bool, bool]]) -> dict:
        both_yes = sum(1 for a, b in pairs if a and b)
        both_no = sum(1 for a, b in pairs if not a and not b)
        return {
            "agree": both_yes + both_no,
            "n": len(pairs),
            "both_yes": both_yes,
            "claude_only": sum(1 for a, b in pairs if a and not b),
            "manual_only": sum(1 for a, b in pairs if b and not a),
        }

    return {"serious": stats(serious), "any_wrong": stats(any_wrong)}


# ---------- Claude 호출 ----------


class Claude:
    """mode: plan(호출하지 않고 필요한 호출 수만 센다) / run(API 호출) / dry(가짜 응답)."""

    def __init__(self, mode: str, model: str, concurrency: int):
        self.mode, self.model = mode, model
        self.pending = 0
        self.failed = 0
        self._sem = asyncio.Semaphore(concurrency)
        self._client = None
        if mode == "run":
            key = get_settings().anthropic_api_key
            if not key:
                sys.exit("ANTHROPIC_API_KEY가 비어 있어요. .env를 확인하세요.")
            self._client = anthropic.AsyncAnthropic(api_key=key, timeout=120, max_retries=3)
        self._cache_dir = SCORES_DIR / ("cache_dry" if mode == "dry" else "cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, role: str, system: str, text: str, image: Path | None) -> Path:
        image_hash = hashlib.sha256(image.read_bytes()).hexdigest() if image else ""
        raw = json.dumps([self.model, role, system, text, image_hash], ensure_ascii=False)
        return self._cache_dir / f"{role}_{hashlib.sha256(raw.encode()).hexdigest()[:24]}.json"

    async def ask(
        self, role: str, system: str, text: str, image: Path | None = None
    ) -> dict | None:
        path = self._path(role, system, text, image)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.mode == "plan":
            self.pending += 1
            return None
        result = (
            fake_answer(role, text) if self.mode == "dry" else await self._call(system, text, image)
        )
        if result is None:
            self.failed += 1
            return None
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result

    async def _call(self, system: str, text: str, image: Path | None) -> dict | None:
        content: list[dict] = []
        if image:
            media = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
            data = base64.b64encode(image.read_bytes()).decode()
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
            )
        content.append({"type": "text", "text": text})
        async with self._sem:
            try:
                res = await self._client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=system,
                    messages=[{"role": "user", "content": content}],
                )
            except anthropic.APIError as e:
                print(f"API 오류: {type(e).__name__} {e}", file=sys.stderr)
                return None
        return parse_json("".join(b.text for b in res.content if b.type == "text"))


def fake_answer(role: str, text: str) -> dict:
    """--dry-run용 가짜 응답. 흐름 확인에만 쓰고 숫자에는 의미가 없다."""
    if role == "reader":
        vocab = load_vocab()
        answer = {a: [v for v in vocab[a] if v[:2] in text][: LIMITS[a]] for a in ATTRS}
        answer["사진 종류"] = ["그림·짤"] if ("그림" in text or "캐릭터" in text) else ["진짜 사진"]
        return answer
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return {
        "claims": [
            {"claim": s, "verdict": "unsupported" if '"' in s else "supported", "serious": False}
            for s in sentences
        ]
    }


# ---------- 버전 하나 평가 ----------


async def evaluate_version(
    claude: Claude, version: str, items: dict[str, Path], gold: dict, with_control: bool
) -> list[dict]:
    found = descriptions(f"speed_{version}.csv", f"speed_memes_{version}.csv")
    ids = [pid for pid in items if pid in found and pid in gold]
    vocab = load_vocab()
    system = reader_system(vocab)
    text_of = {pid: found[pid]["text"] for pid in ids}
    swap = shifted(ids)

    async def read(pid: str, source: str) -> dict | None:
        raw = await claude.ask("reader", system, f"설명문:\n{text_of[source]}")
        return clean_attributes(raw, vocab) if raw else None

    async def check(pid: str) -> dict | None:
        raw = await claude.ask("judge", JUDGE_SYSTEM, f"설명문:\n{text_of[pid]}", image=items[pid])
        return count_claims(raw) if raw else None

    reads = await asyncio.gather(*(read(pid, pid) for pid in ids))
    checks = await asyncio.gather(*(check(pid) for pid in ids))
    controls = await asyncio.gather(*(read(pid, swap[pid]) for pid in ids)) if with_control else []

    results = []
    for index, pid in enumerate(ids):
        pred, claims = reads[index], checks[index]
        info, per_attr = info_score(pred, gold[pid]) if pred else (None, {})
        control = (
            info_score(controls[index], gold[pid])[0] if with_control and controls[index] else None
        )
        results.append(
            {
                "id": pid,
                "split": gold[pid].get("split", ""),
                "chars": len(text_of[pid]),
                "pred": pred,
                "info": info,
                "attr_f1": per_attr,
                "control_info": control,
                "precision": claim_precision(claims),
                "supported": claims["supported"] if claims else None,
                "unsupported": claims["unsupported"] if claims else None,
                "unverifiable": claims["unverifiable"] if claims else None,
                "serious": claims["serious"] if claims else None,
                "claims": claims["claims"] if claims else [],
            }
        )
    return results


# ---------- 보고서 ----------


def number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def split_means(results: list[dict], field: str) -> dict[str, float | None]:
    out = {}
    for split in ("tune", "final"):
        values = [r[field] for r in results if r["split"] == split and r[field] is not None]
        out[split] = statistics.mean(values) if values else None
    return out


def build_report(all_results: dict[str, list[dict]], manual: dict[str, dict]) -> str:
    versions = list(all_results)
    summaries = {v: summarize(all_results[v]) for v in versions}
    out = ["# 사진 설명문 점수표", ""]
    out += [
        "## 버전별 점수",
        "",
        "| 버전 | 사진 수 | 정보 전달 | 기준선(섞어 읽기) | 기준선 대비 | 사실 정확도 | "
        "사진당 틀린 주장 | 큰 오류 설명 수 | 글자 수 중앙값 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in versions:
        s = summaries[v]
        out.append(
            f"| {v} | {s['n']} | {number(s['info'])} | {number(s['control'])} | "
            f"{number(s['info_over_control'])} | {number(s['precision'])} | "
            f"{number(s['wrong_per_description'], 2)} | {s['serious_descriptions']} | "
            f"{number(s['chars'], 0)} |"
        )
    out += [
        "",
        "## 속성별 정보 전달 점수(F1)",
        "",
        "| 버전 | " + " | ".join(ATTRS) + " |",
        "|---|" + "---|" * len(ATTRS),
    ]
    for v in versions:
        row = " | ".join(number(summaries[v]["attr_f1"].get(a)) for a in ATTRS)
        out.append(f"| {v} | {row} |")
    out += [
        "",
        "## tune / final 나눠 본 점수",
        "",
        "| 버전 | 정보 전달 tune | 정보 전달 final | 사실 정확도 tune | 사실 정확도 final |",
        "|---|---|---|---|---|",
    ]
    for v in versions:
        info = split_means(all_results[v], "info")
        prec = split_means(all_results[v], "precision")
        out.append(
            f"| {v} | {number(info['tune'])} | {number(info['final'])} | "
            f"{number(prec['tune'])} | {number(prec['final'])} |"
        )
    if len(versions) > 1:
        base = versions[0]
        out += [
            "",
            f"## {base}(기준) 대비 사진별 짝 비교",
            "",
            "| 비교 | 지표 | 기준 | 새 | 차이 | 95% 신뢰구간 | 개선율 | 좋아짐/같음/나빠짐 "
            "| 부호 검정 p |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for v in versions[1:]:
            for label, field in (("정보 전달", "info"), ("사실 정확도", "precision")):
                cmp = paired_compare(
                    values_by_id(all_results[base], field), values_by_id(all_results[v], field)
                )
                if not cmp:
                    continue
                rel = "-" if cmp["relative"] is None else f"{cmp['relative']:+.1%}"
                out.append(
                    f"| {base}→{v} | {label} | {cmp['base_mean']:.3f} | {cmp['new_mean']:.3f} | "
                    f"{cmp['diff']:+.3f} | [{cmp['ci'][0]:+.3f}, {cmp['ci'][1]:+.3f}] | {rel} | "
                    f"{cmp['wins']}/{cmp['ties']}/{cmp['losses']} | {cmp['p']:.2f} |"
                )
        out += ["", "신뢰구간이 0을 지나가거나 p가 0.05보다 크면 차이가 우연일 수 있다."]
    if manual:
        out += [
            "",
            "## Claude 판정과 내 판정이 맞는 정도",
            "",
            "| 버전 | 기준 | 같은 판단 | 둘 다 있음 | Claude만 있음 | 내 판정만 있음 |",
            "|---|---|---|---|---|---|",
        ]
        for v, agreement in manual.items():
            for label, key in (("큰 오류", "serious"), ("틀린 주장 있음", "any_wrong")):
                a = agreement[key]
                out.append(
                    f"| {v} | {label} | {a['agree']}/{a['n']} | {a['both_yes']} | "
                    f"{a['claude_only']} | {a['manual_only']} |"
                )
    out += [
        "",
        "## 읽는 법",
        "",
        "- 정보 전달은 설명만 읽고 속성을 얼마나 되살렸는지, 기준선은 다른 사진의 설명을 "
        "읽혔을 때의 점수다. 기준선이 높으면 어휘만으로도 맞혀지는 속성이 많다는 뜻이다.",
        "- 버전별 표의 사실 정확도는 모든 주장을 합쳐 센 값이고, 짝 비교의 사실 정확도는 "
        "사진별 값의 평균이라 조금 다를 수 있다.",
        "- 이 점수는 문장이 자연스러운지는 잡지 못한다. 사람이 읽고 따로 봐야 한다.",
        "- 정답은 사람이 확인하기 전까지 Claude의 초안이다.",
    ]
    return "\n".join(out) + "\n"


# ---------- 실행 ----------


def load_gold_for(mode: str, allow_unconfirmed: bool) -> dict:
    path = GOLD_PATH if GOLD_PATH.exists() else DRAFT_PATH
    if path == DRAFT_PATH and mode != "dry":
        sys.exit(
            "gold_labels.json이 없어요. label.html에서 정답을 확인하고 내보낸 파일을 "
            "data/photo_eval/gold_labels.json으로 저장하세요."
        )
    gold = load_gold(path)
    unconfirmed = [pid for pid, v in gold.items() if not v.get("confirmed")]
    if unconfirmed and mode != "dry" and not allow_unconfirmed:
        sys.exit(
            f"확인하지 않은 정답이 {len(unconfirmed)}개 있어요. label.html에서 확인하거나 "
            "--allow-unconfirmed로 초안 그대로 쓰세요."
        )
    return gold


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("versions", nargs="*", default=["D", "E"], help="비교할 프롬프트 버전")
    parser.add_argument("--run", action="store_true", help="API를 실제로 호출한다.")
    parser.add_argument("--dry-run", action="store_true", help="가짜 응답으로 흐름만 확인한다.")
    parser.add_argument("--model", default=get_settings().llm_model)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="앞에서 N장만(시험용)")
    parser.add_argument("--no-control", action="store_true", help="기준선(섞어 읽기)을 생략")
    parser.add_argument("--allow-unconfirmed", action="store_true")
    args = parser.parse_args()

    mode = "dry" if args.dry_run else "run" if args.run else "plan"
    gold = load_gold_for(mode, args.allow_unconfirmed)
    items = {i["id"]: DATA / i["src"] for i in load_items()}
    if args.limit:
        items = dict(list(items.items())[: args.limit])
    claude = Claude(mode, args.model, args.concurrency)

    all_results = {}
    for version in args.versions:
        all_results[version] = await evaluate_version(
            claude, version, items, gold, with_control=not args.no_control
        )
    if mode == "plan" and claude.pending:
        print(
            f"모델 {args.model}로 새로 호출할 횟수: {claude.pending}번 "
            "(저장된 응답은 다시 쓰므로 제외)"
        )
        print("읽기 담당(글만), 검수(사진 포함), 기준선(글만) 호출이 섞여 있다. 비용이 든다.")
        print("진행하려면 --run을 붙여 다시 실행하세요.")
        return
    if claude.failed:
        print(f"실패한 호출 {claude.failed}번은 점수에서 빠졌어요.", file=sys.stderr)

    manual = {}
    for version in args.versions:
        judged = load_judgments(f"judgments_{version}.json", [{"id": pid} for pid in items])
        if judged:
            manual[version] = manual_agreement(all_results[version], judged)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_dry" if mode == "dry" else ""
    for version, results in all_results.items():
        out = SCORES_DIR / f"{version}{suffix}.json"
        out.write_text(
            json.dumps(
                {"summary": summarize(results), "items": results}, ensure_ascii=False, indent=1
            ),
            encoding="utf-8",
        )
    report = build_report(all_results, manual)
    (SCORES_DIR / f"report{suffix}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
