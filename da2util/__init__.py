import re, requests, subprocess, time
from enum import Enum
from pathlib import Path

def cuda_version() -> str:
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"nvcc failed (code {result.returncode}). "
                f"stderr: {result.stderr.strip()}"
            )
    except FileNotFoundError as e:
        print("cuda_version: nvcc command not found")
    else:    
        # Look for the first occurrence of ``release X.Y`` where X and Y are numbers.
        match = re.search(r"release\s+(\d+)\.(\d+)", result.stdout, re.IGNORECASE)
        if not match:
            raise ValueError("cuda_version: Could not locate CUDA version in nvcc output.")
        major, minor = match.groups()
        return f"{major}{minor}"

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
