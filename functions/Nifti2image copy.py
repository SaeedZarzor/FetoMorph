from PIL import Image, ImageDraw, ImageFont
import numpy as np

def measure_text(draw, text, font):
    # Preferred (Pillow ≥8.0; textsize removed in ≥10)
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    # Fallback: font bbox
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    # Old Pillow fallback
    return draw.textsize(text, font=font)



def nifti_slice_to_image(inp_path: str, out_path: str,
    BAR_MM=25,
    MM_PER_PX=None,
    REL_BAR_W=0.22,
    MARGIN=24,
    PAD=8,
    LINE_THICK=8,
    FONT_SIZE=32,
    CROP_THRESH=25,
    COLOR_THRESH=15):
#inp_path = "seg.nii_view.png"
#out_path = "image_with_scalebar.png"


    # 1) load
    img = Image.open(inp_path).convert("RGB")
    arr = np.array(img)

    # 2) crop to content (remove dark borders)
    bright = arr.sum(axis=2)  # 0..765
    mask_content = bright > CROP_THRESH
    ys, xs = np.where(mask_content)
    if len(xs) > 0 and len(ys) > 0:   # ✅ corrected condition
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        pad = 10
        y0 = max(0, y0 - pad); y1 = min(arr.shape[0], y1 + pad)
        x0 = max(0, x0 - pad); x1 = min(arr.shape[1], x1 + pad)
        arr = arr[y0:y1, x0:x1, :]

    # 3) remove grayscale + unify colored to blue on white background
    a = arr.astype(np.int16)
    ptp = a.max(axis=2) - a.min(axis=2)
    mask_colored = ptp > COLOR_THRESH

    out = np.ones_like(a) * 255
    out[mask_colored] = np.array([0, 0, 255], dtype=np.int16)
    proc = Image.fromarray(out.astype(np.uint8), "RGB")

    # 4) scalebar
    w, h = proc.size
    draw = ImageDraw.Draw(proc, "RGBA")
    bar_px = int(round(BAR_MM / MM_PER_PX)) if MM_PER_PX else max(40, int(round(w * REL_BAR_W)))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    text = f"{BAR_MM} mm"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    rect_w = bar_px + 2*PAD
    rect_h = max(LINE_THICK, 2) + th + 3*PAD
    rx = w - MARGIN - rect_w
    ry = h - MARGIN - rect_h

    draw.rectangle([rx, ry, rx+rect_w, ry+rect_h], fill=(0, 0, 0, 160))
    x1 = rx + PAD; x2 = x1 + bar_px; y = ry + PAD + LINE_THICK//2
    draw.line([x1, y, x2, y], fill=(255,255,255,255), width=LINE_THICK)
    draw.text((x1 + (bar_px - tw)//2, y + PAD), text, fill=(255,255,255,255), font=font)

    proc.save(out_path)
    print("[Nifti to PNG] The image saved:", out_path)
