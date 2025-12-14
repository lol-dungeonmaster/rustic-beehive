import requests, time
from enum import Enum
from pathlib import Path

class DepthModel(Enum):
    Small = "vits"
    Base = "vitb"
    Large = "vitl"

from support.api import Api

def init_model(encoder: DepthModel, video_metric_depth: bool = False):
    if not video_metric_depth:
        checkpoints = Path("external/Depth-Anything-V2/checkpoints")
        target_file = f"depth_anything_v2_{encoder.value}.pth"
        url_path = f"Depth-Anything-V2-{encoder.name}"
    else:
        checkpoints = Path("external/Video-Depth-Anything/checkpoints")
        target_file = f"metric_video_depth_anything_{encoder.value}.pth"
        url_path = f"Metric-Video-Depth-Anything-{encoder.name}"

    checkpoints.mkdir(parents=True, exist_ok=True)
    target_path = checkpoints / target_file
    file_present = target_path.is_file()

    if not file_present:
        print(f"init_model: Downloading encoder {encoder.value}, do not interrupt...")
        try:
            url = f"https://huggingface.co/depth-anything/{url_path}/resolve/main/{target_file}"
            data = Api.get(url, as_response=True)
            with open(target_path, "wb") as f:
                for chunk in data.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"init_model: disconnected downloading {encoder.value}, retrying in 15s")
            time.sleep(15)
            init_model(encoder)
        else:
            print(f"init_model: done")
