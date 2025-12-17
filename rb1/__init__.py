import cv2, Imath, json, numpy as np, open3d as o3d, OpenEXR, os, pathlib, torch
from cv2.typing import MatLike
from da2util.model import DepthModel, VideoModel
from google import genai
from IPython.display import HTML
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from PIL import Image, PngImagePlugin
from screeninfo import get_monitors
from typing import Any
from . import agent

from support.secret import UserSecretsClient
GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

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

from depth_anything_v2.dpt import DepthAnythingV2
from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def scale_to_uint16(depths: MatLike) -> tuple[MatLike, float, float]:
    min_val = float(depths.min())
    max_val = float(depths.max())
    if max_val == min_val:
        raise ValueError("This depth map has zero dynamic range.")
    scaled = ((depths - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
    return scaled, min_val, max_val

def save_depth_png(depths: MatLike, src_name: str, out_path: str = 'docs/results'):
    scaled, min_val, max_val = scale_to_uint16(depths)
    png_out = Image.fromarray(scaled, mode="I;16") # "I;16" = 16‑bit unsigned integer grayscale
    # Store the scaling parameters in PNG metadata.
    meta = PngImagePlugin.PngInfo()
    meta.add_text(
        "depth_scale",
        json.dumps({"min": min_val, "max": max_val}),
        zip=False
    )
    png_out.save(os.path.join(out_path, os.path.splitext(os.path.basename(src_name))[0] + '_depths.png'), pnginfo=meta)

def load_depth_png(depths_image: str) -> MatLike:
    png_in = Image.open(depths_image) # reads metadata into .info
    scaled = np.array(png_in, dtype=np.float32)
    # Extract scaling parameters.
    meta_json = png_in.info.get("depth_scale")
    if meta_json is None:
        raise KeyError("Missing 'depth_scale' metadata in PNG.")
    params = json.loads(meta_json)
    min_val = float(params["min"])
    max_val = float(params["max"])
    # Recover the original depths.
    depths = scaled / 65535.0 * (max_val - min_val) + min_val
    return depths.astype(np.float32)

def save_depth_lossless(depths: MatLike | NDArray, src_name: str, out_path: str = 'docs/results'):
    # Save the raw depth map.
    np.savez_compressed(os.path.join(out_path, os.path.splitext(os.path.basename(src_name))[0]+'_depths.npz'), depths=depths)
    # Check if depths is a video.
    if isinstance(depths, tuple) and len(depths) == 2:
        export_video_exr(depths, src_name, out_path)
    else:
        export_image_exr(depths, os.path.splitext(os.path.basename(src_name))[0]+'_depths.exr', out_path)

def export_image_exr(depth: MatLike, out_name: str, out_path: str):
    output_exr = f"{out_path}/{out_name}"
    header = OpenEXR.Header(depth.shape[1], depth.shape[0])
    header["channels"] = {
        "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    exr_file = OpenEXR.OutputFile(output_exr, header)
    exr_file.writePixels({"Z": depth.tobytes()})
    exr_file.close()

def export_video_exr(depths: NDArray, src_name: str, out_path: str):
    exr_out = os.path.join(out_path, os.path.splitext(os.path.basename(src_name))[0]+'_depths_exr')
    os.makedirs(exr_out, exist_ok=True)
    for i, depth in enumerate(depths):
        export_image_exr(depth, f"frame_{i:05d}.exr", exr_out)

def infer_image_depth(image: str, encoder: DepthModel = DepthModel.Large) -> MatLike:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'image_depth: using {device} backend')
    # Generate depth prediction.
    model = DepthAnythingV2(**model_configs[encoder.value])
    model.load_state_dict(torch.load(f'external/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder.value}.pth', map_location='cpu'))
    model = model.to(device).eval()
    depths = model.infer_image(cv2.imread(image))
    # Save the depth prediction.
    save_depth_png(depths, src_name=image)
    save_depth_lossless(depths, src_name=image)
    return depths # HxW raw depth map in numpy