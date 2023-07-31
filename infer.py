import os

from app.core import run_pipeline

if __name__ == '__main__':
    sources = int(os.environ.get("sources"))
    # file path (file://...) or RTSP URL (rtsp://...)
    rtsp_urls = str(os.environ.get("rtsp_urls"))
    rtsp_ids = str(os.environ.get("rtsp_ids"))
    run_pipeline(sources, rtsp_urls, rtsp_ids)
