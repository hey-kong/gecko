import os
import sys
import logging
import configparser
from functools import partial
from inspect import signature
from typing import List
from collections import defaultdict

import gi
import cv2
import pyds
import numpy as np
from gi.repository import GObject, Gst, GstRtspServer

from app.utils.bus_call import bus_call
from app.utils.callback_function import cb_newpad, decodebin_child_added
from app.utils.is_aarch_64 import is_aarch64
from app.utils.fps import FPSMonitor
from app.utils.bbox import rect_params_to_coords
from app.config import CROPS_DIR

sys.path.append('../../')

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')

bitrate = int(os.environ.get("bitrate"))  # 2000000
rows = int(os.environ.get("rows"))
columns = int(os.environ.get("columns"))

MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 0
rtsp_port_num = 8554
updsink_port_num = 5400


class Pipeline:
    def __init__(self, *, num_sources: int, rtsp_urls: str, rtsp_ids: str,
                 pgie_config_path: str, tracker_config_path: str, enable_osd: bool = True,
                 write_osd_analytics: bool = True, save_crops: bool = False,
                 rtsp_codec: str = "H264", input_shape: tuple = (1920, 1080)):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.num_sources = num_sources
        self.urls = [url.strip() for url in rtsp_urls.split(',')]
        self.ids = [id.strip() for id in rtsp_ids.split(',')]
        self.pgie_config_path = pgie_config_path
        self.tracker_config_path = tracker_config_path
        self.enable_osd = enable_osd
        self.write_osd_analytics = write_osd_analytics
        self.save_crops = save_crops
        self.rtsp_codec = rtsp_codec
        self.input_shape = input_shape
        self.input_width = self.input_shape[0]
        self.input_height = self.input_shape[1]

        # Check input arguments
        self.fps_streams = {}
        for i in range(0, len(self.urls)):
            self.fps_streams["stream{0}".format(i)] = FPSMonitor(i)

        # Save crops
        if self.save_crops:
            if not os.path.isdir(CROPS_DIR):
                os.makedirs(CROPS_DIR)
            self.logger.info(f"Saving crops to '{CROPS_DIR}'.")
            self.track_scores = defaultdict(list)

        # Standard GStreamer initialization
        GObject.threads_init()
        Gst.init(None)

        # Create gstreamer elements
        # Create Pipeline element that will form a connection of other elements
        self.logger.info("Creating Pipeline")
        self.pipeline = Gst.Pipeline()
        self.is_live = False
        if not self.pipeline:
            self.logger.error("Failed to create Pipeline")

        self.elements = []
        self.streammux = None
        self.pgie = None
        self.tracker = None
        self.nvvidconv1 = None
        self.caps1 = None
        self.tiler = None
        self.nvvidconv2 = None
        self.nvosd = None
        self.queue1 = None
        self.sink_bin = None

        self._create_elements()
        self._link_elements()
        self._add_probes()

    def __str__(self):
        return " -> ".join([elm.name for elm in self.elements])

    def _add_element(self, element, idx=None):
        if idx:
            self.elements.insert(idx, element)
        else:
            self.elements.append(element)

        self.pipeline.add(element)

    def _create_element(self, factory_name, name, print_name, detail="", add=True):
        """
        Creates an element with Gst Element Factory make.
        Return the element if successfully created, otherwise print to stderr and return None.
        """
        self.logger.info(f"Creating {print_name}")
        elm = Gst.ElementFactory.make(factory_name, name)

        if not elm:
            self.logger.error(f"Unable to create {print_name}")
            if detail:
                self.logger.error(detail)

        if add:
            self._add_element(elm)

        return elm

    def _create_source_bin(self, index, uri):
        self.logger.info("Creating Source bin")

        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline.
        bin_name = "source-bin-%02d" % index
        print(bin_name)
        nbin = Gst.Bin.new(bin_name)
        if not nbin:
            self.logger.error("Unable to create source bin")

        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.
        uri_decode_bin = self._create_element("uridecodebin", "uri-decode-bin", "URI decode bin",
                                              add=False)
        if not uri_decode_bin:
            sys.stderr.write("Unable to create uri decode bin \n")
        # We set the input uri to the source element.
        uri_decode_bin.set_property("uri", uri)
        # Connect to the "pad-added" signal of the decodebin which generates a
        # callback once a new pad for raw data has been created by the decodebin.
        uri_decode_bin.connect("pad-added", cb_newpad, nbin)
        uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

        # We need to create a ghost pad for the source bin which will act as a proxy
        # for the video decoder src pad. The ghost pad will not have a target right
        # now. Once the decode bin creates the video decoder and generates the
        # cb_newpad callback, we will set the ghost pad target to the video decoder
        # src pad.
        Gst.Bin.add(nbin, uri_decode_bin)
        bin_pad = nbin.add_pad(
            Gst.GhostPad.new_no_target(
                "src", Gst.PadDirection.SRC))
        if not bin_pad:
            self.logger.error("Failed to add ghost pad in source bin")
            return None

        self._add_element(nbin)
        return nbin

    def _create_streammux(self):
        streammux = self._create_element("nvstreammux", "Stream-muxer", "Stream mux")
        streammux.set_property('width', self.input_width)
        streammux.set_property('height', self.input_height)
        streammux.set_property('batch-size', 1)
        streammux.set_property('batched-push-timeout', 4000000)

        return streammux

    def _create_tracker(self):
        tracker = self._create_element("nvtracker", "tracker", "Tracker")

        config = configparser.ConfigParser()
        config.read(self.tracker_config_path)
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width':
                tracker_width = config.getint('tracker', key)
                tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height':
                tracker_height = config.getint('tracker', key)
                tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id':
                tracker_gpu_id = config.getint('tracker', key)
                tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file':
                tracker_ll_lib_file = config.get('tracker', key)
                tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file':
                tracker_ll_config_file = config.get('tracker', key)
                tracker.set_property('ll-config-file', tracker_ll_config_file)
            if key == 'enable-batch-process':
                tracker_enable_batch_process = config.getint('tracker', key)
                tracker.set_property('enable_batch_process', tracker_enable_batch_process)
            if key == 'enable-past-frame':
                tracker_enable_past_frame = config.getint('tracker', key)
                tracker.set_property('enable_past_frame', tracker_enable_past_frame)

        return tracker

    def _create_tiler(self):
        tiler = self._create_element("nvmultistreamtiler", "nvtiler", "Tiler")
        tiler.set_property("rows", rows)
        tiler.set_property("columns", columns)
        tiler.set_property("width", TILED_OUTPUT_WIDTH)
        tiler.set_property("height", TILED_OUTPUT_HEIGHT)

        return tiler

    def _create_rtsp_sink_bin(self):
        rtsp_sink_bin = Gst.Bin.new("rtsp-sink-bin")

        nvvidconv3 = self._create_element("nvvideoconvert", "convertor3", "Converter 3", add=False)
        capsfilter2 = self._create_element("capsfilter", "capsfilter2", "Caps filter 2", add=False)
        capsfilter2.set_property("caps",
                                 Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

        if self.rtsp_codec not in ["H264", "H265"]:
            raise ValueError(f"Invalid codec '{self.rtsp_codec}'")

        # Make the encoder
        encoder = self._create_element(f"nvv4l2{self.rtsp_codec.lower()}enc", "encoder",
                                       f"{self.rtsp_codec} encoder", add=False)
        encoder.set_property('bitrate', bitrate)

        if is_aarch64():
            encoder.set_property('preset-level', 1)
            encoder.set_property('insert-sps-pps', 1)
            encoder.set_property('bufapi-version', 1)

        # Make the payload-encode video into RTP packets
        rtppay = self._create_element(f"rtp{self.rtsp_codec.lower()}pay", "rtppay",
                                      f"{self.rtsp_codec} rtppay", add=False)

        # Make the UDP sink
        sink = self._create_element("udpsink", "udpsink", "UDP sink", add=False)
        sink.set_property('host', '224.224.255.255')
        sink.set_property('port', updsink_port_num)
        sink.set_property('async', False)
        sink.set_property('sync', 0)
        sink.set_property("qos", 0)

        rtsp_sink_bin.add(nvvidconv3)
        rtsp_sink_bin.add(capsfilter2)
        rtsp_sink_bin.add(encoder)
        rtsp_sink_bin.add(rtppay)
        rtsp_sink_bin.add(sink)

        rtsp_sink_bin.add_pad(Gst.GhostPad.new("sink", nvvidconv3.get_static_pad("sink")))
        self._link_sequential([nvvidconv3, capsfilter2, encoder, rtppay, sink])
        self._add_element(rtsp_sink_bin)

        return rtsp_sink_bin

    def _create_elements(self):
        self.streammux = self._create_streammux()

        for i in range(self.num_sources):
            uri_name = self.urls[i]
            if uri_name.find("rtsp://") == 0:
                self.is_live = True
            source_bin = self._create_source_bin(i, uri_name)
            padname = "sink_%u" % i
            sinkpad = self.streammux.get_request_pad(padname)
            if not sinkpad:
                self.logger.error("Unable to create sink pad bin")
            srcpad = source_bin.get_static_pad("src")
            if not srcpad:
                self.logger.error("Unable to create src pad bin")
            srcpad.link(sinkpad)

        # Create pgie
        self.pgie = self._create_element("nvinfer", "primary-inference", "PGIE")
        self.pgie.set_property('config-file-path', self.pgie_config_path)
        pgie_batch_size = self.pgie.get_property("batch-size")
        if pgie_batch_size != self.num_sources:
            print(
                "WARNING: Overriding infer-config batch-size",
                pgie_batch_size,
                " with number of sources ",
                self.num_sources,
                " \n",
            )
            self.pgie.set_property("batch-size", self.num_sources)

        # Create tracker
        self.tracker = self._create_tracker()

        # Create nvvidconv1
        self.nvvidconv1 = self._create_element("nvvideoconvert", "convertor1", "Converter 1")

        # Create a caps1 filter
        self.caps1 = self._create_element("capsfilter", "capsfilter1", "Caps filter 1")
        self.caps1.set_property("caps",
                                Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

        # Create tiler
        self.tiler = self._create_tiler()

        # Create nvvidconv2
        self.nvvidconv2 = self._create_element("nvvideoconvert", "convertor2", "Converter 2")

        # Create nvosd
        if self.enable_osd:
            self.nvosd = self._create_element("nvdsosd", "onscreendisplay", "OSD")

        # Create queue1
        self.queue1 = self._create_element("queue", "queue1", "Queue 1")

        # Create sink_bin
        self.sink_bin = self._create_rtsp_sink_bin()

        if not is_aarch64():
            # Use CUDA unified memory so frames can be easily accessed on CPU in Python.
            mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            self.nvvidconv1.set_property("nvbuf-memory-type", mem_type)
            self.tiler.set_property("nvbuf-memory-type", mem_type)
            self.nvvidconv2.set_property("nvbuf-memory-type", mem_type)

    @staticmethod
    def _link_sequential(elements: list):
        for i in range(0, len(elements) - 1):
            elements[i].link(elements[i + 1])

    def _link_elements(self):
        self.logger.info(f"Linking elements in the Pipeline: {self}")

        elm = self.elements[0]
        for i in range(1 + self.num_sources, len(self.elements)):
            elm.link(self.elements[i])
            elm = self.elements[i]

    def _write_osd_analytics(self, batch_meta, l_frame_meta: List, ll_obj_meta: List[List]):
        pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "Roadsign"]
        PGIE_CLASS_ID_VEHICLE = 0
        PGIE_CLASS_ID_BICYCLE = 1
        PGIE_CLASS_ID_PERSON = 2
        PGIE_CLASS_ID_ROADSIGN = 3

        # Initializing object counter with 0.
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0
        }

        for frame_meta, l_obj_meta in zip(l_frame_meta, ll_obj_meta):
            frame_number = frame_meta.frame_num
            num_rects = frame_meta.num_obj_meta

            for obj_meta in l_obj_meta:
                obj_counter[obj_meta.class_id] += 1

                # Update the object text display
                txt_params = obj_meta.text_params

                # Set display_text. Any existing display_text string will be
                # freed by the bindings' module.
                txt_params.display_text = pgie_classes_str[obj_meta.class_id]

                # Font, font-color and font-size
                txt_params.font_params.font_name = "Serif"
                txt_params.font_params.font_size = 10
                # set(red, green, blue, alpha); set to White
                txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

                # Text background color
                txt_params.set_bg_clr = 1
                # set(red, green, blue, alpha); set to Black
                txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

            self.logger.info("Frame Number = {}, Vehicle Count = {}, Person Count = {}".
                             format(frame_number, obj_counter[PGIE_CLASS_ID_VEHICLE],
                                    obj_counter[PGIE_CLASS_ID_PERSON]))

    def _calculate_crop_score(self, track_id, crop):
        score = crop.size
        num_detections = len(self.track_scores[track_id])

        # Penalizing entry frames
        if num_detections <= 15:
            score -= 1000

        return score

    def _save_crops(self, frames, _, l_frame_meta: List, ll_obj_meta: List[List]):
        self.logger.info(f"Saving crops to '{os.path.realpath(CROPS_DIR)}'")
        for frame, frame_meta, l_obj_meta in zip(frames, l_frame_meta, ll_obj_meta):
            frame_copy = np.array(frame, copy=True, order='C')
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)

            for obj_meta in l_obj_meta:
                track_id = obj_meta.object_id
                x1, y1, x2, y2 = rect_params_to_coords(obj_meta.rect_params)
                crop = frame_copy[y1:y2, x1:x2]
                crop_score = self._calculate_crop_score(track_id, crop)

                if not self.track_scores[track_id] or crop_score > max(self.track_scores[track_id]):
                    crop_dir = os.path.join(CROPS_DIR, f"src_{frame_meta.source_id}",
                                            f"obj_{obj_meta.object_id}_cls_{obj_meta.class_id}")
                    os.makedirs(crop_dir, exist_ok=True)
                    for f in os.listdir(crop_dir):
                        os.remove(os.path.join(crop_dir, f))
                    crop_path = os.path.join(crop_dir, f"frame_{frame_meta.frame_num}.jpg")
                    cv2.imwrite(crop_path, crop)
                    self.logger.debug(f"Saved crop to '{crop_path}'")

                # To calculate scene change rate
                self.track_scores[track_id].append(crop_score)

    def _probe_fn_wrapper(self, _, info, probe_fn, get_frames=False):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            self.logger.error("Unable to get GstBuffer")
            return

        frames = []
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        l_frame = batch_meta.frame_meta_list
        l_frame_meta = []
        ll_obj_meta = []
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            if get_frames:
                frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                frames.append(frame)

            l_frame_meta.append(frame_meta)
            l_obj_meta = []

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                l_obj_meta.append(obj_meta)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            ll_obj_meta.append(l_obj_meta)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        if get_frames:
            probe_fn(frames, batch_meta, l_frame_meta, ll_obj_meta)
        else:
            probe_fn(batch_meta, l_frame_meta, ll_obj_meta)

        return Gst.PadProbeReturn.OK

    def _wrap_probe(self, probe_fn):
        get_frames = "frames" in signature(probe_fn).parameters
        return partial(self._probe_fn_wrapper, probe_fn=probe_fn, get_frames=get_frames)

    @staticmethod
    def _get_static_pad(element, pad_name: str = "sink"):
        pad = element.get_static_pad(pad_name)
        if not pad:
            raise AttributeError(f"Unable to get {pad_name} pad of {element.name}")

        return pad

    def _add_probes(self):
        tiler_sinkpad = self._get_static_pad(self.tiler)

        if self.enable_osd and self.write_osd_analytics:
            tiler_sinkpad.add_probe(Gst.PadProbeType.BUFFER,
                                    self._wrap_probe(self._write_osd_analytics))

        if self.save_crops:
            tiler_sinkpad.add_probe(Gst.PadProbeType.BUFFER, self._wrap_probe(self._save_crops))

    def release(self):
        """Release resources and cleanup."""
        pass

    def run(self):
        # Create an event loop and feed gstreamer bus messages to it
        loop = GObject.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_call, loop)

        # Start streaming
        server = GstRtspServer.RTSPServer.new()
        server.props.service = "%d" % rtsp_port_num
        server.attach(None)

        factory = GstRtspServer.RTSPMediaFactory.new()
        factory.set_launch(
            '( udpsrc name=pay0 port=%d buffer-size=524288 caps="application/x-rtp, media=video, clock-rate=90000, '
            'encoding-name=(string)%s, payload=96 " ) '
            % (updsink_port_num, self.rtsp_codec)
        )
        factory.set_shared(True)
        server.get_mount_points().add_factory("/ds-test", factory)

        self.logger.info(
            "\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n"
            % rtsp_port_num
        )

        # Start play back and listen to events
        self.logger.info("Starting pipeline")
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            loop.run()
        except:
            pass

        self.logger.info("Exiting pipeline")
        self.pipeline.set_state(Gst.State.NULL)
        self.release()
