import cv2, logging, numpy as np, requests, time, torch
from cv2.typing import MatLike
from enum import Enum
from google import genai
from google.api_core import retry
from google.genai.models import Models
from google.genai import errors
from IPython.display import HTML
from matplotlib import pyplot as plt
from pathlib import Path
from screeninfo import get_monitors
from . import agent

log = logging.getLogger()
logging.getLogger("google_adk.google.adk.models.google_llm").setLevel(logging.WARNING)
logging.getLogger("google_adk.google.adk.runners").setLevel(logging.ERROR)
logging.getLogger("google_adk.google.adk.plugins.plugin_manager").setLevel(logging.WARNING)

from .secret import UserSecretsClient
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

is_retriable = lambda e: (isinstance(e, errors.APIError) and e.code in {429, 503, 500})
Models.generate_images = retry.Retry(predicate=is_retriable)(Models.generate_images)
Models.generate_videos = retry.Retry(predicate=is_retriable)(Models.generate_videos)
Models.generate_content = retry.Retry(predicate=is_retriable)(Models.generate_content)

def side_by_side(img1, img2, width="45%", margin="2%"):
    html = f"""
    <div style="
        display:flex;
        justify-content:center;
        align-items:center;
        gap:{margin};
    ">
        <img src="{img1}" style="width:{width}; max-width:100%; height:auto;">
        <img src="{img2}" style="width:{width}; max-width:100%; height:auto;">
    </div>
    """
    return HTML(html)

def depth_to_heatmap(depth_map):
    # 1. Normalize the depth values to [0, 255]
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    
    # Avoid division by zero if output is constant
    if depth_max - depth_min > 0:
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros(depth_map.shape)

    # 2. Scale to 0-255 and convert to uint8
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    # 3. Apply a colormap (INFERNO is standard for depth, or use JET/VIRIDIS)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    # 4. Convert from BGR (opencv) to RGB for Jupyter/Colab
    return cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

def show_full_width(img):
    # Set dpi based on screen resolution.
    dpi = 100 if get_monitors()[0].height <= 1080 else 144

    # Get image dimensions.
    height, width = img.shape[:2]
    
    # Calculate figure size in inches (width / dpi, height / dpi).
    figsize = width / float(dpi), height / float(dpi)
    
    # Create the figure with the exact size.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Add an axes that fills the whole figure (no white borders).
    ax = fig.add_axes([0, 0, 1, 1])
    
    # Turn off axis labels.
    ax.axis('off')
    
    # Display
    ax.imshow(img)
    plt.show()

class DA2Model(Enum):
    Small = "vits"
    Base = "vitb"
    Large = "vitl"

from .api import Api

def init_model(encoder: DA2Model):
    checkpoints = Path("external/Depth-Anything-V2/checkpoints")
    checkpoints.mkdir(parents=True, exist_ok=True)
    target_path = checkpoints / f"depth_anything_v2_{encoder.value}.pth"
    file_present = target_path.is_file()

    if not file_present:
        url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{encoder.name}/resolve/main/depth_anything_v2_{encoder.value}.pth"
        print(f"init_model: Downloading encoder {encoder.value}, do not interrupt...")
        try:
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

from depth_anything_v2.dpt import DepthAnythingV2

def infer_depth(image: str, encoder: DA2Model = DA2Model.Large) -> MatLike:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"infer_depth: using {device} backend")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    img_mat = cv2.imread(image)
    model = DepthAnythingV2(**model_configs[encoder.value])
    model.load_state_dict(torch.load(f'external/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder.value}.pth', map_location='cpu'))
    model = model.to(device).eval()
    
    return model.infer_image(img_mat) # HxW raw depth map in numpy
