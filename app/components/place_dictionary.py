"""명소 사전 — 기능5 기억 검색. 질문의 "서울숲" 같은 명소 이름을 중심 좌표·반경으로 바꾼다.

자물쇠에는 위경도와 법정동 이름만 있어서 "서울숲", "홍대"처럼 동 이름이 아닌 명소는 좌표 거리로
비교해야 한다. 이름→좌표 대응은 app/places.yaml에 미리 적어 두고 여기서 읽는다.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.components.name_matcher import blank_out, name_pattern, squash

DEFAULT_PLACES_PATH = Path(__file__).resolve().parent.parent / "places.yaml"

# 한반도 범위. 이 밖의 좌표는 위경도를 뒤바꿨거나 잘못 옮겨 적은 것이다
LAT_RANGE = (33.0, 39.0)
LNG_RANGE = (124.0, 132.0)


@dataclass(frozen=True)
class PlaceEntry:
    name: str
    aliases: tuple[str, ...]
    type: str
    lat: float
    lng: float
    radius_km: float


class PlaceDictionary:
    def __init__(self, entries: Sequence[PlaceEntry]):
        self.entries = tuple(entries)
        names = [(alias, entry) for entry in entries for alias in (entry.name, *entry.aliases)]
        # 긴 이름부터 찾는다 — "홍대입구"가 "홍대"보다 먼저 잡혀야 뒤의 "입구"가 남지 않는다
        names.sort(key=lambda pair: -len(squash(pair[0])))
        self._patterns = [(name_pattern(alias), entry) for alias, entry in names]

    @classmethod
    def load(cls, path: Path = DEFAULT_PLACES_PATH) -> "PlaceDictionary":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        radius_by_type = _read_type_radius(data.get("type_radius_km"), path)

        entries: list[PlaceEntry] = []
        owner: dict[str, str] = {}  # 공백 없앤 이름/별칭 → 어느 항목의 것인지
        for raw in data.get("places") or []:
            name = str(raw.get("name", "")).strip()
            if not name:
                raise ValueError(f"{path}: 이름이 없는 항목이 있어요: {raw}")
            type_ = raw.get("type")
            if type_ not in radius_by_type:
                raise ValueError(
                    f"{path}: {name}의 유형 '{type_}'을 type_radius_km에서 찾을 수 없어요."
                )
            lat, lng = _read_center(raw.get("center"), name, path)
            radius = float(raw.get("radius_km") or radius_by_type[type_])
            if radius <= 0:
                raise ValueError(f"{path}: {name}의 반경은 0보다 커야 해요.")

            aliases = tuple(str(a).strip() for a in raw.get("aliases") or [])
            for label in (name, *aliases):
                key = squash(label)
                if key in owner:
                    raise ValueError(
                        f"{path}: '{label}'이(가) {owner[key]}와(과) {name}에 중복돼요."
                    )
                owner[key] = name
            entries.append(PlaceEntry(name, aliases, type_, lat, lng, radius))

        if not entries:
            raise ValueError(f"{path}에 명소가 하나도 없어요.")
        return cls(entries)

    def find(self, text: str) -> tuple[list[PlaceEntry], str]:
        """질문에서 명소를 찾는다. (찾은 명소들, 찾은 글자를 공백으로 지운 질문)을 돌려준다."""
        found: list[PlaceEntry] = []
        for pattern, entry in self._patterns:
            match = pattern.search(text)
            if match is None:
                continue
            if entry not in found:
                found.append(entry)
            text = blank_out(text, match.start(), match.end())
        return found, text


def _read_type_radius(raw: object, path: Path) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: type_radius_km이 없어요.")
    radius = {str(k): float(v) for k, v in raw.items()}
    if any(v <= 0 for v in radius.values()):
        raise ValueError(f"{path}: 유형별 반경은 0보다 커야 해요.")
    return radius


def _read_center(raw: object, name: str, path: Path) -> tuple[float, float]:
    if not (isinstance(raw, list) and len(raw) == 2):
        raise ValueError(f"{path}: {name}의 center는 [위도, 경도] 두 값이어야 해요.")
    lat, lng = float(raw[0]), float(raw[1])
    if not (LAT_RANGE[0] <= lat <= LAT_RANGE[1] and LNG_RANGE[0] <= lng <= LNG_RANGE[1]):
        raise ValueError(f"{path}: {name}의 좌표 [{lat}, {lng}]가 한반도 범위를 벗어났어요.")
    return lat, lng
