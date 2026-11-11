"""
Image upload validation and preprocessing (report §12.2 steps 1-2, §21).

Validates by decoding the file (not by trusting the extension), rejects
oversized and decompression-bomb images, fixes EXIF orientation, strips
metadata (e.g. GPS) by re-encoding, and downsizes to the vision model's
working resolution.
"""

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from config import settings

ALLOWED_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "BMP": "image/bmp"}
MAX_PIXELS = 40_000_000
MIN_SIDE = 32


class ImageValidationError(ValueError):
    pass


@dataclass
class PreparedImage:
    data: bytes
    mime: str
    width: int
    height: int
    original_format: str
    original_size: tuple
    resized: bool


def prepare_image(raw_bytes, max_bytes=None, max_side=None):
    max_bytes = max_bytes or settings.MAX_IMAGE_BYTES
    max_side = max_side or settings.MAX_IMAGE_SIDE

    if not raw_bytes:
        raise ImageValidationError("The uploaded image is empty.")
    if len(raw_bytes) > max_bytes:
        raise ImageValidationError(
            f"Image is {len(raw_bytes) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB."
        )

    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(raw_bytes)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as error:
        raise ImageValidationError(f"Not a valid or supported image: {error}") from error

    fmt = (image.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError(
            f"Unsupported image format '{fmt or 'unknown'}'. Use PNG, JPEG, WEBP or BMP."
        )
    original_size = image.size
    if min(original_size) < MIN_SIDE:
        raise ImageValidationError("Image is too small to read.")

    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        background = Image.new("RGB", image.size, "white")
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        image = background
    elif image.mode == "L":
        image = image.convert("RGB")

    resized = False
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        resized = True

    buffer = io.BytesIO()
    # PNG keeps thin lines and small labels in diagrams sharp; photos -> JPEG.
    if fmt == "JPEG":
        image.save(buffer, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"

    return PreparedImage(
        data=buffer.getvalue(), mime=mime, width=image.width, height=image.height,
        original_format=fmt, original_size=original_size, resized=resized,
    )
