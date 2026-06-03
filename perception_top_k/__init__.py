from .config import Config, ScoringConfig, SelectionConfig
from .tracks import TrackBuilder
from . import bbox

__all__ = [
    "Config",
    "ScoringConfig",
    "SelectionConfig",
    "TrackBuilder",
    "bbox",
]


def run(config: Config) -> int:
    from .main import run as _run
    return _run(config)
