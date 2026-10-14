"""시도·시군구 코드표 — 기능5 기억 검색. 질문의 "성동구" 같은 지역 이름을 코드 앞부분으로 바꾼다.

자물쇠의 법정동코드(10자리)는 앞 2자리가 시도, 앞 5자리가 시군구다. 그래서 "서울에서"는 코드가
"11"로 시작하는 모든 자물쇠로, "성동구에서"는 "11200"으로 시작하는 자물쇠로 비교하면 되고 동
목록은 필요 없다. 이름→코드 대응은 app/regions.yaml(법정동코드 전체자료에서 추출)에 있다.

같은 이름이 여러 곳에 있다(중구·서구 등). 그런 이름은 후보 코드를 모두 돌려주고, 이 사용자의
자물쇠에 실제로 있는 지역만 인정하는 건 쓰는 쪽(condition_scoring)이 한다.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.components.name_matcher import blank_out, name_pattern, squash

DEFAULT_REGIONS_PATH = Path(__file__).resolve().parent.parent / "regions.yaml"


@dataclass(frozen=True)
class SidoEntry:
    code: str  # 앞 2자리
    name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class SigunguEntry:
    prefix: str  # 앞 5자리. 구가 있는 시(수원시 등)는 구들의 공통 앞 4자리
    sido: str
    name: str  # 시도 이름을 뺀 이름 ("성동구", "수원시 장안구")


@dataclass(frozen=True)
class RegionHit:
    text: str  # 질문에서 잡힌 글자 (안내 문장에 쓴다)
    prefixes: tuple[str, ...]  # 후보 법정동코드 앞부분
    # 공식 이름·약칭이면 True. '시'·'군'을 뗀 말("수원", "공주")은 일반 낱말과 겹칠 수 있어 False
    reliable: bool


class RegionTable:
    def __init__(self, sido: Sequence[SidoEntry], sigungu: Sequence[SigunguEntry]):
        self.sido = tuple(sido)
        self.sigungu = tuple(sigungu)

        # 같은 이름(공백 없앤 것)이 가리키는 코드를 모은다
        groups: dict[str, tuple[str, list[str], bool]] = {}
        for alias, prefix, reliable in _aliases(self.sido, self.sigungu):
            key = squash(alias)
            display, prefixes, was_reliable = groups.get(key, (alias, [], True))
            if prefix not in prefixes:
                prefixes.append(prefix)
            groups[key] = (display, prefixes, was_reliable and reliable)
        ordered = sorted(groups.values(), key=lambda g: -len(squash(g[0])))  # 긴 이름부터
        self._patterns = [(name_pattern(d), tuple(p), r) for d, p, r in ordered]

    @classmethod
    def load(cls, path: Path = DEFAULT_REGIONS_PATH) -> "RegionTable":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        sido: list[SidoEntry] = []
        for raw in data.get("sido") or []:
            code = str(raw.get("code", ""))
            name = str(raw.get("name", "")).strip()
            if len(code) != 2 or not code.isdigit() or not name:
                raise ValueError(f"{path}: 시도 항목이 올바르지 않아요: {raw}")
            if any(s.code == code for s in sido):
                raise ValueError(f"{path}: 시도 코드 {code}가 중복돼요.")
            aliases = tuple(str(a).strip() for a in raw.get("aliases") or [])
            sido.append(SidoEntry(code, name, aliases))
        if not sido:
            raise ValueError(f"{path}에 시도가 하나도 없어요.")

        codes = {s.code for s in sido}
        sigungu: list[SigunguEntry] = []
        seen: set[str] = set()
        for raw in data.get("sigungu") or []:
            prefix = str(raw.get("prefix", ""))
            sido_code = str(raw.get("sido", ""))
            name = str(raw.get("name", "")).strip()
            if len(prefix) not in (4, 5) or not prefix.isdigit() or not name:
                raise ValueError(f"{path}: 시군구 항목이 올바르지 않아요: {raw}")
            if sido_code not in codes or not prefix.startswith(sido_code):
                raise ValueError(f"{path}: {name}({prefix})의 시도 코드가 맞지 않아요.")
            if prefix in seen:
                raise ValueError(f"{path}: 시군구 코드 {prefix}가 중복돼요.")
            seen.add(prefix)
            sigungu.append(SigunguEntry(prefix, sido_code, name))
        if not sigungu:
            raise ValueError(f"{path}에 시군구가 하나도 없어요.")
        return cls(sido, sigungu)

    def find(self, text: str) -> tuple[list[RegionHit], str]:
        """질문에서 지역 이름을 찾는다. (찾은 지역들, 찾은 글자를 공백으로 지운 질문)을 돌려준다."""
        hits: list[RegionHit] = []
        for pattern, prefixes, reliable in self._patterns:
            match = pattern.search(text)
            if match is None:
                continue
            hits.append(RegionHit(" ".join(match.group(0).split()), prefixes, reliable))
            text = blank_out(text, match.start(), match.end())
        return hits, text


def _aliases(
    sido: Sequence[SidoEntry], sigungu: Sequence[SigunguEntry]
) -> list[tuple[str, str, bool]]:
    """(질문에서 쓸 이름, 코드 앞부분, 믿을 만한 이름인지)를 모두 만든다."""
    out: list[tuple[str, str, bool]] = []
    sido_names = {s.code: (s.name, *s.aliases) for s in sido}

    for s in sido:
        for alias in (s.name, *s.aliases):
            out.append((alias, s.code, True))

    for g in sigungu:
        out.append((g.name, g.prefix, True))
        # "서울 중구"처럼 시도와 함께 말하면 같은 이름이어도 한 곳으로 정해진다
        for sido_name in sido_names[g.sido]:
            out.append((f"{sido_name} {g.name}", g.prefix, True))

        if " " in g.name:  # "수원시 장안구" → "장안구", "수원 장안구"
            city, gu = g.name.split(" ", 1)
            out.append((gu, g.prefix, True))
            if len(city) >= 3 and city[-1] in "시군":
                out.append((f"{city[:-1]} {gu}", g.prefix, True))
        elif len(g.name) >= 3 and g.name[-1] in "시군":  # "수원시" → "수원"
            out.append((g.name[:-1], g.prefix, False))
    return out
