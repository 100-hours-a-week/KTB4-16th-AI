"""아직 구현 전인 기능 서비스가 공통으로 쓰는 응답."""

from app.exceptions import NotImplementedFeatureError


def not_implemented(feature: str) -> NotImplementedFeatureError:
    return NotImplementedFeatureError(f"{feature}은(는) 아직 구현되지 않았어요.")
