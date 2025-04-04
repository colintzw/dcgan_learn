import numpy as np
from PIL import Image, ImageOps


def crop_and_resize(image_path, padding, target_size):
    img = Image.open(image_path).convert("RGBA")
    img_arr = np.array(img)

    mask_coords = np.argwhere(img_arr[:, :, 3] > 0)
    y_min, x_min = mask_coords.min(axis=0)
    y_max, x_max = mask_coords.max(axis=0) + 1  # Include last pixel

    # Find current padding on each side
    top_padding = y_min
    left_padding = x_min
    bottom_padding = img_arr.shape[0] - y_max
    right_padding = img_arr.shape[1] - x_max

    # Find the smallest padding (ensuring equal margins)
    min_padding = min(top_padding, left_padding, bottom_padding, right_padding)

    # Crop the image using the smallest padding to balance margins
    cropped = img.crop(
        (
            x_min - min_padding,
            y_min - min_padding,
            x_max + min_padding,
            y_max + min_padding,
        )
    )
    size_wo_pad = (target_size[0] - padding * 2, target_size[1] - padding * 2)
    resized = cropped.resize(size_wo_pad, Image.LANCZOS)

    # Add extra padding to match the desired padding
    resized = ImageOps.expand(
        resized, border=padding, fill=(255, 255, 255, 0)
    )  # White and transparent padding

    return resized
