import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

if not Gst.is_initialized():
    Gst.init(None)

from .config import (
    PipelineConfig,
    RTSPSource,
    StreamMuxConfig,
    TrackerConfig,
    InferConfig,
    MetadataExportConfig,
    OutputConfig,
    build_source_id_mapping,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    VEHICLE_CLASS_IDS,
    PGIE_GIE_ID,
    SGIE_GIE_ID,
    RTSP_STREAMS,
)
from .pipeline import (
    PerceptionPipeline,
    PipelineError,
    create_pipeline,
)
from .writers import (
    JsonlWriter,
    NullWriter,
)
from .probes import (
    ProbeConfig,
    PerformanceStats,
    create_metadata_export_probe,
    create_debug_probe,
)
from .elements import (
    make_element,
    create_rtsp_source_bin,
    create_streammux,
    create_pgie,
    create_sgie,
    create_tracker,
    create_fakesink,
    check_rtsp_available,
    check_available_sources,
)

__all__ = [
    # Config
    "PipelineConfig",
    "RTSPSource",
    "StreamMuxConfig",
    "TrackerConfig",
    "InferConfig",
    "MetadataExportConfig",
    "OutputConfig",
    "build_source_id_mapping",
    "FRAME_WIDTH",
    "FRAME_HEIGHT",
    "VEHICLE_CLASS_IDS",
    "PGIE_GIE_ID",
    "SGIE_GIE_ID",
    "RTSP_STREAMS",
    # Pipeline
    "PerceptionPipeline",
    "PipelineError",
    "create_pipeline",
    # Writers
    "JsonlWriter",
    "NullWriter",
    # Probes
    "ProbeConfig",
    "PerformanceStats",
    "create_metadata_export_probe",
    "create_debug_probe",
    # Elements
    "make_element",
    "create_rtsp_source_bin",
    "create_streammux",
    "create_pgie",
    "create_sgie",
    "create_tracker",
    "create_fakesink",
    "check_rtsp_available",
    "check_available_sources",
]

__version__ = "0.1.0"
