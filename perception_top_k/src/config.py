from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class BboxConfig:
    coord_width: int = 1920
    coord_height: int = 1080
    edge_penalty_threshold_px: int = 32


@dataclass(frozen=True)
class ContextConfig:
    expand_ratio: float = 0.15
    near_duplicate_iou_threshold: float = 0.90
    near_duplicate_area_ratio_threshold: float = 0.85
    foreground_bottom_y_delta: int = 8
    foreground_vertical_overlap_threshold: float = 0.20
    foreground_horizontal_overlap_threshold: float = 0.20
    foreground_target_coverage_threshold: float = 0.05


@dataclass(frozen=True)
class ScoringConfig:
    det_conf_good: float = 0.80
    lap_reference: float = 120.0
    tenengrad_reference: float = 12000.0
    edge_margin_px: int = 32
    min_det_conf: float = 0.20
    max_foreground_overlap: float = 0.85
    max_edge_penalty: float = 0.65
    min_visible_area_ratio: float = 0.80


@dataclass(frozen=True)
class SelectionConfig:
    top_k: int = 2
    strict_temporal_gap: int = 30
    relaxed_temporal_gap: int = 15
    min_score: float = 0.55


@dataclass
class Config:
    input_path: Path = field(default_factory=lambda: Path("output/kafka_output.jsonl"))
    output_path: Path = field(default_factory=lambda: Path("output/perception_tracks.jsonl"))
    frames_dir: Optional[Path] = None
    crops_output_dir: Optional[Path] = None
    
    target_labels: FrozenSet[str] = frozenset({"car", "bus", "truck", "motorcycle", "bicycle"})
    
    bbox: BboxConfig = field(default_factory=BboxConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    
    @classmethod
    def from_paths(
        cls,
        input_path: Path,
        output_path: Path,
        frames_dir: Optional[Path] = None,
        crops_output_dir: Optional[Path] = None,
    ) -> "Config":
        return cls(
            input_path=input_path,
            output_path=output_path,
            frames_dir=frames_dir,
            crops_output_dir=crops_output_dir,
        )
