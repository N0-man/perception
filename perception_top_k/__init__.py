from .src.config import Config, ScoringConfig, SelectionConfig
from .src.tracks import TrackBuilder
from .src import bbox

__all__ = [
    "Config",
    "ScoringConfig",
    "SelectionConfig",
    "TrackBuilder",
    "bbox",
]


def run(config: Config) -> int:
    from .src.main import run as _run
    return _run(config)
