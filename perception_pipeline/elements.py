import configparser
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Ensure GStreamer is initialized before any element creation
if not Gst.is_initialized():
    Gst.init(None)

from .config import StreamMuxConfig, RTSPSource, MP4Source


def make_element(factory_name: str, element_name: str) -> Gst.Element:
    element = Gst.ElementFactory.make(factory_name, element_name)
    if not element:
        raise RuntimeError(f"Unable to create element: {factory_name} ({element_name})")
    return element


def check_rtsp_available(uri: str, timeout: float = 2.0) -> bool:
    try:
        parsed = urlparse(uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 554
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        return result == 0
    except Exception:
        return False


def check_available_sources(sources: list[RTSPSource]) -> list[tuple[int, RTSPSource]]:
    available = []
    
    print("\nChecking RTSP stream availability...")
    for i, source in enumerate(sources):
        if check_rtsp_available(source.uri):
            print(f"  ✓ [{source.camera_id}] {source.uri} - AVAILABLE")
            available.append((i, source))
        else:
            print(f"  ✗ [{source.camera_id}] {source.uri} - NOT AVAILABLE")
    
    print(f"\n{len(available)}/{len(sources)} streams available\n")
    return available


def create_rtsp_source_bin(index: int, uri: str, camera_id: Optional[str] = None) -> Gst.Bin:
    if camera_id is None:
        camera_id = f"source_{index}"
    
    bin_name = f"source-bin-{index:02d}"
    source_bin = Gst.Bin.new(bin_name)

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
    if not uri_decode_bin:
        raise RuntimeError(f"Unable to create uridecodebin for source {index}")

    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.set_property("buffer-size", 4096)
    uri_decode_bin.set_property("buffer-duration", 0)

    def on_pad_added(decodebin: Gst.Element, pad: Gst.Pad, data: Gst.Bin) -> None:
        caps = pad.get_current_caps()
        if not caps:
            caps = pad.query_caps(None)

        structure = caps.get_structure(0)
        name = structure.get_name()

        if not name.startswith("video"):
            return

        features = caps.get_features(0)
        
        if features.contains("memory:NVMM"):
            bin_ghost_pad = data.get_static_pad("src")
            if bin_ghost_pad:
                result = bin_ghost_pad.set_target(pad)
                if result:
                    print(f"[{camera_id}] ✓ Stream connected")
                else:
                    print(f"[{camera_id}] ✗ Failed to link decoder pad")

    def on_source_setup(decodebin: Gst.Element, source: Gst.Element, user_data) -> None:
        source_name = source.get_factory().get_name() if source.get_factory() else ""
        
        if "rtspsrc" in source_name:
            source.set_property("latency", 200)
            source.set_property("drop-on-latency", True)
            source.set_property("timeout", 5000000)
            source.set_property("tcp-timeout", 5000000)

    uri_decode_bin.connect("pad-added", on_pad_added, source_bin)
    uri_decode_bin.connect("source-setup", on_source_setup, None)

    source_bin.add(uri_decode_bin)

    # Create ghost pad for the bin
    ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    source_bin.add_pad(ghost_pad)

    return source_bin


def create_streammux(config: StreamMuxConfig) -> Gst.Element:
    streammux = make_element("nvstreammux", "stream-muxer")
    
    streammux.set_property("width", config.width)
    streammux.set_property("height", config.height)
    streammux.set_property("batch-size", config.batch_size)
    streammux.set_property("live-source", 1 if config.live_source else 0)
    streammux.set_property("batched-push-timeout", config.batched_push_timeout)
    
    return streammux


def create_pgie(config_file: Path, batch_size: Optional[int] = None) -> Gst.Element:
    
    pgie = make_element("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", str(config_file))
    
    # Note: Changing batch-size requires rebuilding the engine file
    if batch_size is not None:
        pgie.set_property("batch-size", batch_size)
    
    return pgie


def create_sgie(config_file: Path) -> Gst.Element:
    sgie = make_element("nvinfer", "secondary-inference")
    sgie.set_property("config-file-path", str(config_file))
    
    return sgie


def create_tracker(config_file: Path) -> Gst.Element:
    
    tracker = make_element("nvtracker", "tracker")
    
    config = configparser.ConfigParser()
    config.read(str(config_file))
    
    if 'tracker' not in config.sections():
        raise ValueError(f"Tracker config file missing [tracker] section: {config_file}")
    
    # Property mapping from config key to element property
    property_map = {
        'tracker-width': ('tracker-width', config.getint),
        'tracker-height': ('tracker-height', config.getint),
        'gpu-id': ('gpu-id', config.getint),
        'll-lib-file': ('ll-lib-file', config.get),
        'll-config-file': ('ll-config-file', config.get),
        'display-tracking-id': ('display-tracking-id', config.getint),
    }
    
    for key, (prop_name, getter) in property_map.items():
        if key in config['tracker']:
            try:
                value = getter('tracker', key)
                tracker.set_property(prop_name, value)
            except (ValueError, configparser.Error) as e:
                print(f"Warning: Failed to set tracker property {prop_name}: {e}")
            except TypeError as e:
                # Property may not exist in this DeepStream version
                print(f"Warning: Property {prop_name} not supported: {e}")
    
    return tracker


def create_fakesink(name: str = "fakesink", sync: bool = False) -> Gst.Element:
    
    fakesink = make_element("fakesink", name)
    fakesink.set_property("sync", sync)
    fakesink.set_property("async", False)
    
    return fakesink


def create_mp4_source_bin(index: int, file_path: Path, camera_id: Optional[str] = None) -> Gst.Bin:
    """Create a source bin for MP4/H264 file input.
    
    Pipeline: filesrc -> h264parse -> nvv4l2decoder -> (ghost pad)
    """
    if camera_id is None:
        camera_id = f"source_{index}"
    
    bin_name = f"source-bin-{index:02d}"
    source_bin = Gst.Bin.new(bin_name)
    
    # Create elements
    filesrc = make_element("filesrc", f"file-source-{index}")
    filesrc.set_property("location", str(file_path))
    
    h264parser = make_element("h264parse", f"h264-parser-{index}")
    decoder = make_element("nvv4l2decoder", f"nvv4l2-decoder-{index}")
    
    # Add elements to bin
    source_bin.add(filesrc)
    source_bin.add(h264parser)
    source_bin.add(decoder)
    
    # Link elements: filesrc -> h264parse -> nvv4l2decoder
    if not filesrc.link(h264parser):
        raise RuntimeError(f"Failed to link filesrc to h264parser for source {index}")
    if not h264parser.link(decoder):
        raise RuntimeError(f"Failed to link h264parser to decoder for source {index}")
    
    # Create ghost pad from decoder's src pad
    decoder_srcpad = decoder.get_static_pad("src")
    if not decoder_srcpad:
        raise RuntimeError(f"Unable to get decoder src pad for source {index}")
    
    ghost_pad = Gst.GhostPad.new("src", decoder_srcpad)
    source_bin.add_pad(ghost_pad)
    
    print(f"[{camera_id}] ✓ MP4 source configured: {file_path}")
    
    return source_bin
