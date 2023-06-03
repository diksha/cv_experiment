import os
import shutil
import subprocess  # trunk-ignore(bandit/B404)

import streamlit as st
import yaml
from PIL import Image
from streamlit_cropper import st_cropper

### Set variables
crop_box = None  # trunk-ignore(pylint/C0103)

# Check if 'key' already exists in session_state
# If not, then initialize it
if "crop_box" not in st.session_state:
    st.session_state["crop_box"] = crop_box


### Helpers
def get_fisheye_circle_params(box: dict) -> list:
    """Return fisheye circle params

    Args:
        box (dict): coordinates of top left corner and bounding box shape

    Returns:
        list: fisheye radius and center coordinates
    """
    fisheye_r = max(box["height"], box["width"]) / 2.0
    fisheye_cx = box["width"] / 2.0 + box["left"]
    fisheye_cy = box["height"] / 2.0 + box["top"]
    return fisheye_r, fisheye_cx, fisheye_cy


def fish2persp_exe() -> str:
    """Finds the fish2persp binary

    Raises:
        RuntimeError: if it fails to find the fish2persp exe

    Returns:
        str: path to the fish2persp exe
    """
    lookup_paths = [
        "fish2persp",
        "/opt/voxel/bin/fish2persp",
    ]
    if os.getenv("FISH2PERSP") is not None:
        lookup_paths.append(os.getenv("FISH2PERSP"))

    for path in lookup_paths:
        exe_path = shutil.which(path)
        if exe_path is not None:
            return exe_path

    raise RuntimeError("failed to find fish2persp exe")


def display_value():
    """Displays the dewarped image"""

    # Update the image
    if img_file and crop_box:
        # Save image for processing
        image = Image.open(img_file)
        image.save("./img_out.jpg")

        # Compute fisheye circle params
        fisheye_r, fisheye_cx, fisheye_cy = get_fisheye_circle_params(
            st.session_state.crop_box
        )

        # Dewarp
        cmd = [
            fish2persp_exe(),
            "-w",
            "1280",
            "-h",
            "720",
            "-r",
            str(int(fisheye_r)),
            "-c",
            str(int(fisheye_cx)),
            str(int(fisheye_cy)),
            "-s",
            str(int(st.session_state.fov_slider)),
            "-x",
            str(int(st.session_state.tilt_slider)),
            "-y",
            str(int(st.session_state.roll_slider)),
            "-z",
            str(int(st.session_state.pan_slider)),
            "-t",
            str(int(st.session_state.persp_slider)),
            "-f",
            "./img_out.jpg",
        ]

        print(cmd)
        subprocess.run(cmd, check=True)  # trunk-ignore(bandit/B603)

        # Display image
        st.header("Adjust the FOV, roll, pitch, yaw sliders")

        st.image("./_persp.jpg")

        # Write YAML
        fish2persp_config = {
            "fish2persp_remap": {
                "fish": {
                    "width_pixels": img_width,
                    "height_pixels": img_height,
                    "center_x_pixels": int(fisheye_cx),
                    "center_y_pixels": int(fisheye_cy),
                    "radius_x_pixels": int(fisheye_r),
                    "fov_degrees": int(st.session_state.fov_slider),
                    "tilt_degrees": int(st.session_state.tilt_slider),
                    "roll_degrees": int(st.session_state.roll_slider),
                    "pan_degrees": int(st.session_state.pan_slider),
                },
                "persp": {
                    "width_pixels": 1280,
                    "height_pixels": 720,
                    "fov_degrees": int(st.session_state.persp_slider),
                },
            }
        }

        with open("fish2persp_config.yaml", "w", encoding="utf-8") as outfile:
            yaml.dump(fish2persp_config, outfile, default_flow_style=False)

        with open("fish2persp_config.yaml", "rb") as file:
            st.download_button(
                label="Download Fisheye config as YAML",
                data=file,
                file_name="fish2persp_config.yaml",
                mime="text",
            )
        # Display details
        persp_image = Image.open("./_persp.jpg")
        width, height = persp_image.size
        st.write(f"Width: {width} | Height: {height}")
        st.write(f"Center X: {fisheye_cx} | Center Y: {fisheye_cy}")
        config_str = yaml.dump(fish2persp_config)
        st.code(config_str)

    else:
        st.write("First upload an image!!!")


### GUI Parameters

st.title("Fisheye Dewarping")

# Create sidebar inputs
st.sidebar.header(
    "1. Set the bounding box tightly around the fisheye region. :arrow_down:"
)
uploadImg = st.sidebar.file_uploader(
    label="Upload your image!",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=False,
)

st.sidebar.header(
    "2. Set the bounding box tightly around the fisheye region. :arrow_right:"
)

st.sidebar.header("3. Adjust the FOV, roll, pitch, yaw sliders")

fisheye_fov = st.sidebar.slider(
    "Fisheye FOV",
    180,
    0,
    180,
    1,
    key="fov_slider",
    help="FOV of the camera. Most fisheyes are 180 deg.",
)

perspective_fov = st.sidebar.slider(
    "Perspective FOV",
    0,
    180,
    90,
    1,
    key="persp_slider",
    help="The zoom of the perspective view. A higher FOV number means zooming out.",
)

roll = st.sidebar.slider(
    "Tilt",
    -180,
    180,
    0,
    1,
    key="tilt_slider",
    help="Positive number tilts the camera down. Negative number tilts the camera up.",
)
pitch = st.sidebar.slider(
    "Roll",
    -180,
    180,
    0,
    1,
    key="roll_slider",
    help="Positive number rotates the camera clockwise.",
)
yaw = st.sidebar.slider(
    "Pan",
    -180,
    180,
    0,
    1,
    key="pan_slider",
    help="Positive number pans the camera right. Negative number pans the camera left.",
)
button = st.sidebar.button("Apply Transform", on_click=display_value)


### CROPPER

# Bounding box
st.header(
    "Set the bounding box tightly around the fisheye region. :arrow_down:"
)
SHOULD_UPDATE_REALTIME = True
ASPECT_RATIO = None
BOX_COLOR = "#0000FF"

img_file = uploadImg
if img_file:
    img = Image.open(img_file)
    if not SHOULD_UPDATE_REALTIME:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    crop_box = st_cropper(
        img,
        realtime_update=SHOULD_UPDATE_REALTIME,
        box_color=BOX_COLOR,
        aspect_ratio=ASPECT_RATIO,
        return_type="box",
        key="fisheye_center",
    )

    img_width, img_height = img.size

    st.session_state.crop_box = crop_box
