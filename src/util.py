from PIL import Image
import math


def preprocess_image(image_path, target_size=384):
    """
    Preprocess the image by resizing and center cropping to the target size.

    Args:
        image_path (str): Path to the input image.
        target_size (int): Desired size for the shortest side of the image.

    Returns:
        PIL.Image: Preprocessed image.
    """
    
    image = Image.open(image_path).convert("RGB")

    # Resize while maintaining aspect ratio
    W, H = image.size
    scale = target_size / min(W, H)
    new_W = math.ceil(W * scale)
    new_H = math.ceil(H * scale)
    image = image.resize((new_W, new_H), resample=Image.BICUBIC)

    # Center crop to target size
    W, H = image.size
    left = (W - target_size) // 2
    top = (H - target_size) // 2
    image = image.crop((left, top, left + target_size, top + target_size))

    return image