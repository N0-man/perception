import sys
from pathlib import Path
from typing import Optional

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

if not Gst.is_initialized():
    Gst.init(None)

from .config import PipelineConfig, StreamMuxConfig, build_source_id_mapping, RTSPSource, MP4Source
from .elements import (
    check_available_sources,
    create_rtsp_source_bin,
    create_mp4_source_bin,
    create_streammux,
    create_pgie,
    create_sgie,
    create_tracker,
    create_fakesink,
)
from .probes import create_metadata_export_probe, ProbeConfig, PerformanceStats
from .writers import JsonlWriter, MetadataWriter


class PipelineError(Exception):
    pass


class PerceptionPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pipeline: Optional[Gst.Pipeline] = None
        self.writer: Optional[MetadataWriter] = None
        self.stats: Optional[PerformanceStats] = None
        self._loop: Optional[GLib.MainLoop] = None
        
        # Available sources (set during build)
        self._available_sources: list[tuple[int, any]] = []
        self._source_bins: list[Gst.Bin] = []
        self._streammux: Optional[Gst.Element] = None
        self._pgie: Optional[Gst.Element] = None
        self._tracker: Optional[Gst.Element] = None
        self._sgie: Optional[Gst.Element] = None
        self._fakesink: Optional[Gst.Element] = None
    
    def build(self) -> "PerceptionPipeline":
        if not self.config.sources:
            raise PipelineError("No sources configured.")
        
        # Check source type and prepare available sources
        first_source = self.config.sources[0]
        
        if isinstance(first_source, MP4Source):
            # Validate MP4 files exist
            for source in self.config.sources:
                if not source.file_path.exists():
                    raise PipelineError(f"MP4 file not found: {source.file_path}")
            self._available_sources = [(i, src) for i, src in enumerate(self.config.sources)]
            print(f"Using {len(self._available_sources)} MP4 source(s)")
        else:
            # RTSP sources - check availability
            self._available_sources = check_available_sources(self.config.sources)
            if not self._available_sources:
                raise PipelineError("No RTSP streams available. Please start at least one stream.")
        
        # Update batch size based on available sources
        batch_size = len(self._available_sources)
        self.config.streammux.batch_size = batch_size
        print(f"Setting streammux batch-size to {batch_size}")
        
        # Create pipeline
        self.pipeline = Gst.Pipeline.new("perception-pipeline")
        if not self.pipeline:
            raise PipelineError("Failed to create pipeline")
        
        self._create_elements()
        self._add_elements_to_pipeline()
        self._link_sources_to_mux()
        self._link_processing_chain()
        self._attach_probes()
        
        return self
    
    def _create_elements(self) -> None:
        config = self.config
        
        # Create source bins based on source type
        self._source_bins = []
        for i, (orig_idx, source) in enumerate(self._available_sources):
            if isinstance(source, MP4Source):
                source_bin = create_mp4_source_bin(i, source.file_path, source.camera_id)
            else:
                source_bin = create_rtsp_source_bin(i, source.uri, source.camera_id)
            self._source_bins.append(source_bin)
        
        # Stream muxer (batch size already updated)
        self._streammux = create_streammux(config.streammux)
        
        # Inference elements
        self._pgie = create_pgie(config.pgie_config)
        self._tracker = create_tracker(config.tracker_config)
        self._sgie = create_sgie(config.sgie_config)
        self._fakesink = create_fakesink()
        
        # Writer
        self.writer = JsonlWriter(
            config.output.jsonl_path,
            flush_every=config.output.flush_every
        )
    
    def _add_elements_to_pipeline(self) -> None:
        for source_bin in self._source_bins:
            self.pipeline.add(source_bin)
        
        self.pipeline.add(self._streammux)
        self.pipeline.add(self._pgie)
        self.pipeline.add(self._tracker)
        self.pipeline.add(self._sgie)
        self.pipeline.add(self._fakesink)
    
    def _link_sources_to_mux(self) -> None:
        for i, source_bin in enumerate(self._source_bins):
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                raise PipelineError(f"Unable to get src pad from source bin {i}")
            
            sinkpad = self._streammux.request_pad_simple(f"sink_{i}")
            if not sinkpad:
                raise PipelineError(f"Unable to get streammux sink pad {i}")
            
            ret = srcpad.link(sinkpad)
            if ret != Gst.PadLinkReturn.OK:
                raise PipelineError(f"Failed to link source {i} to streammux: {ret}")
    
    def _link_processing_chain(self) -> None:
        links = [
            (self._streammux, self._pgie, "streammux → pgie"),
            (self._pgie, self._tracker, "pgie → tracker"),
            (self._tracker, self._sgie, "tracker → sgie"),
            (self._sgie, self._fakesink, "sgie → fakesink"),
        ]
        
        for src, dst, name in links:
            if not src.link(dst):
                raise PipelineError(f"Failed to link {name}")
    
    def _attach_probes(self) -> None:
        tracker_srcpad = self._tracker.get_static_pad("src")
        if not tracker_srcpad:
            raise PipelineError("Unable to get tracker src pad")
        
        self.stats = PerformanceStats(log_interval=5.0)
        
        # Build source mapping for available sources only
        source_mapping = {
            i: source.camera_id 
            for i, (_, source) in enumerate(self._available_sources)
        }
        
        probe_config = ProbeConfig(
            writer=self.writer,
            source_id_to_camera_id=source_mapping,
            only_vehicles=self.config.metadata_export.only_vehicles,
            sample_every_n_frames=self.config.metadata_export.sample_every_n_frames,
            stats=self.stats,
        )
        
        probe_func = create_metadata_export_probe(probe_config)
        tracker_srcpad.add_probe(Gst.PadProbeType.BUFFER, probe_func, None)
    
    def run(self) -> None:
        if not self.pipeline:
            raise PipelineError("Pipeline not built. Call build() first.")
        
        self._loop = GLib.MainLoop()
        
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        
        print("Starting pipeline...")
        
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise PipelineError("Unable to set pipeline to PLAYING state")
        
        try:
            self._loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
    
    def stop(self) -> None:
        print("Stopping pipeline...")
        
        if self.stats:
            self.stats.log_final()
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        if self.writer:
            self.writer.close()
        
        if self._loop and self._loop.is_running():
            self._loop.quit()
    
    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message, user_data=None) -> bool:
        msg_type = message.type
        
        if msg_type == Gst.MessageType.EOS:
            print("End of stream")
            if self._loop:
                self._loop.quit()
        
        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"Warning: {err}", file=sys.stderr)
        
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            src_name = message.src.get_name() if message.src else "unknown"
            print(f"Error from {src_name}: {err}", file=sys.stderr)
            if self._loop:
                self._loop.quit()
        
        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, _ = message.parse_state_changed()
                print(f"Pipeline state: {old.value_nick} → {new.value_nick}")
        
        return True


def create_pipeline(config: PipelineConfig) -> PerceptionPipeline:
    return PerceptionPipeline(config).build()
