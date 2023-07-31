import os
import logging

from app.pipeline import Pipeline
from app.config import CONFIGS_DIR, LOGLEVEL

logging.basicConfig(level=LOGLEVEL)


def run_pipeline(num_sources: int, rtsp_urls: str, rtsp_ids: str):
    pipeline = Pipeline(
        num_sources=num_sources,
        rtsp_urls=rtsp_urls,
        rtsp_ids=rtsp_ids,
        pgie_config_path=os.path.join(CONFIGS_DIR, "pgies/pgie.txt"),
        tracker_config_path=os.path.join(CONFIGS_DIR, "trackers/nvdcf.txt")
    )
    pipeline.run()
