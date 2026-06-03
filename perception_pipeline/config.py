from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union


FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

INVALID_TRACK_ID = 18446744073709551615
PGIE_GIE_ID = 1
SGIE_GIE_ID = 2

# default is RTSP
TOGGLE_MP4 = True
MP4_FILE = 'models/data/cam_0_222.h264'

VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})  # car, motorcycle, bus, truck

RTSP_STREAMS = [
    ("cam_0", "rtsp://127.0.0.1:8554/cam_0"),
    # ("cam_1", "rtsp://127.0.0.1:8554/cam_1"),
    # ("cam_2", "rtsp://127.0.0.1:8554/cam_2"),
]


@dataclass(frozen=True)
class RTSPSource:
    camera_id: str
    uri: str


@dataclass(frozen=True)
class MP4Source:
    camera_id: str
    file_path: Path


@dataclass
class StreamMuxConfig:
    width: int = FRAME_WIDTH
    height: int = FRAME_HEIGHT
    batch_size: int = 3
    live_source: bool = True
    batched_push_timeout: int = 400000  # microseconds


@dataclass
class TrackerConfig:
    config_file: Path
    width: int = FRAME_WIDTH
    height: int = FRAME_HEIGHT
    gpu_id: int = 0


@dataclass
class InferConfig:
    config_file: Path
    gie_unique_id: int
    batch_size: Optional[int] = None  # If None, use config file value


@dataclass
class MetadataExportConfig:
    only_vehicles: bool = True
    sample_every_n_frames: int = 1


@dataclass
class OutputConfig:
    jsonl_path: Path
    flush_every: int = 30


Source = Union[RTSPSource, MP4Source]


@dataclass
class PipelineConfig:
    sources: list[Source] = field(default_factory=list)
    
    pgie_config: Path = Path("models/configs/config_infer_primary_yolo26.txt")
    sgie_config: Path = Path("models/configs/config_infer_secondary_lpdnet_tao.txt")
    tracker_config: Path = Path("models/configs/tracker.txt")
    
    streammux: StreamMuxConfig = field(default_factory=StreamMuxConfig)
    metadata_export: MetadataExportConfig = field(default_factory=MetadataExportConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(
        jsonl_path=Path("output/kafka_output.jsonl")
    ))
    
    def __post_init__(self):
        if self.sources:
            self.streammux.batch_size = len(self.sources)
            # MP4 sources are not live
            if isinstance(self.sources[0], MP4Source):
                self.streammux.live_source = False

    @classmethod
    def default_config(cls, base_path: Path) -> "PipelineConfig":
        if TOGGLE_MP4:
            sources: list[Source] = [
                MP4Source(camera_id="cam_0", file_path=base_path / MP4_FILE)
            ]
        else:
            sources = [
                RTSPSource(camera_id=cam_id, uri=uri)
                for cam_id, uri in RTSP_STREAMS
            ]
        
        return cls(
            sources=sources,
            pgie_config=base_path / "models/configs/config_infer_primary_yolo26.txt",
            sgie_config=base_path / "models/configs/config_infer_secondary_lpdnet_tao.txt",
            tracker_config=base_path / "models/configs/tracker.txt",
            streammux=StreamMuxConfig(batch_size=len(sources)),
            output=OutputConfig(jsonl_path=base_path / "output/kafka_output.jsonl"),
        )


def build_source_id_mapping(sources: list[RTSPSource]) -> dict[int, str]:
    return {i: src.camera_id for i, src in enumerate(sources)}
