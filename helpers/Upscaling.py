"""Image upscaling + sharpening for FetoMorph.

A smooth LANCZOS upscale followed by a sharpness boost and an unsharp mask.
Exposed as reusable functions so the "Upscale Image…" adjustment action can run
it on the current image, and as a folder helper / CLI for batch upscaling.

Because the whole frame (including any burned-in scale bar) is enlarged
uniformly by ``scale``, physical calibration measured on the upscaled image
stays correct — do NOT divide an existing mm/pixel by ``scale`` afterwards.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DEFAULT_SCALE = 4
DEFAULT_SHARPNESS_FACTOR = 1.8
DEFAULT_UNSHARP_RADIUS = 1.0
DEFAULT_UNSHARP_PERCENT = 100
DEFAULT_UNSHARP_THRESHOLD = 2


def upscale_image(
    img: Image.Image,
    scale: int = DEFAULT_SCALE,
    *,
    sharpness_factor: float = DEFAULT_SHARPNESS_FACTOR,
    unsharp_radius: float = DEFAULT_UNSHARP_RADIUS,
    unsharp_percent: int = DEFAULT_UNSHARP_PERCENT,
    unsharp_threshold: int = DEFAULT_UNSHARP_THRESHOLD,
) -> Image.Image:
    """Return a smooth LANCZOS-upscaled + sharpened copy of *img*.

    The image is enlarged ``scale``× with LANCZOS resampling, its sharpness is
    boosted by ``sharpness_factor``, then an unsharp mask is applied for crisp
    edges.
    """
    if scale < 1:
        raise ValueError("scale must be >= 1")
    upscaled = img.resize(
        (img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
    sharpened = ImageEnhance.Sharpness(upscaled).enhance(sharpness_factor)
    return sharpened.filter(
        ImageFilter.UnsharpMask(
            radius=unsharp_radius,
            percent=unsharp_percent,
            threshold=unsharp_threshold,
        )
    )


def upscale_image_file(
    input_path,
    output_path,
    scale: int = DEFAULT_SCALE,
    **kwargs,
) -> Path:
    """Upscale one image file and save it as PNG; returns the output path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as img:
        processed = upscale_image(img.convert("RGBA"), scale, **kwargs)
    processed.save(out)
    return out


def upscale_folder(
    input_folder,
    output_folder,
    scale: int = DEFAULT_SCALE,
    *,
    suffix: str = "_upscaled",
    **kwargs,
) -> list[Path]:
    """Upscale every supported image in *input_folder* into *output_folder*."""
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for image_path in sorted(in_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        out_path = out_dir / f"{image_path.stem}{suffix}.png"
        upscale_image_file(image_path, out_path, scale, **kwargs)
        written.append(out_path)
        print(f"Saved: {out_path}")
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smooth LANCZOS upscale + sharpen a folder of images.")
    parser.add_argument("input", type=Path, help="Input image folder.")
    parser.add_argument("output", type=Path, help="Output folder.")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE,
                        help=f"Upscale factor (default {DEFAULT_SCALE}).")
    args = parser.parse_args()
    upscale_folder(args.input, args.output, args.scale)
