from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# ---------------------------------------------------------------------------
# Base configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Kept for local development/debugging.
# Render will load models from Hugging Face instead.
MODELS_DIR = BASE_DIR / "models"

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Hugging Face model repositories
# ---------------------------------------------------------------------------

LINE_MODEL_REPO = "Aryan98999/scanmind-line-ocr"
WORD_MODEL_REPO = "Aryan98999/scanmind-word-ocr"


# ---------------------------------------------------------------------------
# Tuning options for line segmentation
# ---------------------------------------------------------------------------

# Send each line to the model as clean black ink on a white background.
# This removes chalk texture, neighbouring lines and subtitle text from crops.
USE_CLEAN_BINARY_CROPS = True


# Only used when USE_CLEAN_BINARY_CROPS = False.
INVERT_DARK_CROPS = True


# Drop very thin strokes (video subtitles, small UI icons) when the
# handwriting itself is thick (marker / chalk). Set False for thin pen scans.
FILTER_THIN_STROKES = True


# Ignore these top / bottom fractions of the image.
# Example: 0.08 ignores 8%. 0.0 = ignore nothing.
IGNORE_TOP_FRACTION = 0.0
IGNORE_BOTTOM_FRACTION = 0.0


# Horizontal dilation width, in units of median character height.
# Lower -> separate columns stay apart.
# Higher -> broken lines get joined.
JOIN_WIDTH_FACTOR = 2.2


# Fragments on the same row closer than this are merged.
# Example: "*" + "Read operation", "2>" + "ifstream ...".
MERGE_GAP_FACTOR = 2.5


# Final lines narrower than this are dropped.
MIN_LINE_WIDTH_FACTOR = 1.5


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ScanMind OCR API",
    description="Full-page OCR using trained TrOCR models",
    version="3.2.0",
)


# ---------------------------------------------------------------------------
# Global model variables
# ---------------------------------------------------------------------------

word_processor = None
word_model = None

line_processor = None
line_model = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ocr_model(model_name: str):
    """
    Load a ScanMind TrOCR model.

    On Render:
        Models are downloaded automatically from Hugging Face.

    Local development:
        The same code can also access the Hugging Face repositories
        as long as the machine has internet access.
    """

    if model_name == "line_model":
        model_path = LINE_MODEL_REPO

    elif model_name == "word_model":
        model_path = WORD_MODEL_REPO

    else:
        raise ValueError(
            f"Unknown OCR model: {model_name}"
        )

    print(f"Loading OCR model from: {model_path}")

    processor = TrOCRProcessor.from_pretrained(
        model_path
    )

    model = VisionEncoderDecoderModel.from_pretrained(
        model_path
    )

    model.to(DEVICE)
    model.eval()

    print(f"Loaded OCR model successfully: {model_name}")

    return processor, model


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_models():

    global word_processor
    global word_model
    global line_processor
    global line_model

    print("Loading ScanMind OCR models...")

    word_processor, word_model = load_ocr_model(
        "word_model"
    )

    line_processor, line_model = load_ocr_model(
        "line_model"
    )

    print("ScanMind Word OCR model loaded.")
    print("ScanMind Line OCR model loaded.")
    print("All OCR models ready.")


# ---------------------------------------------------------------------------
# Line segmentation helpers
# ---------------------------------------------------------------------------

def _remove_ui_noise(rgb, gray):
    """
    Remove pure-green frame/debug lines drawn over the image.
    """

    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)

    green = (
        (g > 150)
        & (r < 120)
        & (b < 120)
    )

    green = cv2.dilate(
        green.astype(np.uint8) * 255,
        np.ones((5, 5), np.uint8)
    ) > 0

    gray = gray.copy()
    gray[green] = int(np.median(gray))

    clean_rgb = rgb.copy()

    clean_rgb[green] = (
        np.median(
            rgb.reshape(-1, 3),
            axis=0
        ).astype(np.uint8)
    )

    return clean_rgb, gray


def _split_rows(local, line_h):
    """
    Split a blob that holds several touching lines
    at row-projection valleys.

    local: 0/1 mask.

    Returns:
        list of (top, bottom) row ranges.
    """

    h = local.shape[0]

    if h < 1.8 * line_h:
        return [(0, h)]

    rows = local.sum(axis=1).astype(np.float32)

    smooth = np.convolve(
        rows,
        np.ones(5, np.float32) / 5,
        mode="same"
    )

    lo = max(
        1,
        int(line_h * 0.6)
    )

    hi = h - lo

    if hi <= lo:
        return [(0, h)]

    cut = (
        lo
        + int(
            np.argmin(
                smooth[lo:hi]
            )
        )
    )

    top = _split_rows(
        local[:cut],
        line_h
    )

    bottom = [
        (a + cut, b + cut)
        for a, b in _split_rows(
            local[cut:],
            line_h
        )
    ]

    return top + bottom


def _merge_same_line(items, max_gap):
    """
    Merge boxes that sit side by side on the same text row.

    Boxes that overlap horizontally
    are never merged.
    """

    changed = True

    while changed:

        changed = False

        for i in range(len(items)):

            for j in range(i + 1, len(items)):

                a = items[i]["box"]
                b = items[j]["box"]

                x_gap = (
                    max(a[0], b[0])
                    - min(a[2], b[2])
                )

                if x_gap < 0 or x_gap > max_gap:
                    continue

                overlap = (
                    min(a[3], b[3])
                    - max(a[1], b[1])
                )

                min_h = min(
                    a[3] - a[1],
                    b[3] - b[1]
                )

                if min_h <= 0:
                    continue

                if overlap / min_h < 0.5:
                    continue

                items[i] = {
                    "box": (
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                    ),
                    "ids": (
                        items[i]["ids"]
                        | items[j]["ids"]
                    ),
                }

                del items[j]

                changed = True
                break

            if changed:
                break

    return items


# ---------------------------------------------------------------------------
# Line segmentation
# ---------------------------------------------------------------------------

def segment_lines(image: Image.Image):
    """
    Detect text lines in a page with several columns / scattered notes.

    Steps:

    1. Remove green frames, long straight lines,
       tiny specks and thin subtitles.

    2. Join letters and words horizontally
       -> one blob per line fragment.

    3. Split blobs containing touching lines.

    4. Merge fragments belonging to the same row.

    5. Build clean black-on-white crops.

    6. Return crops in reading order.
    """

    rgb = np.array(
        image.convert("RGB")
    )

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray.shape

    rgb, gray = _remove_ui_noise(
        rgb,
        gray
    )

    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Writing -> white, background -> black
    flag = (
        cv2.THRESH_BINARY
        if np.mean(blur) < 127
        else cv2.THRESH_BINARY_INV
    )

    binary = cv2.threshold(
        blur,
        0,
        255,
        flag + cv2.THRESH_OTSU
    )[1]

    if IGNORE_TOP_FRACTION > 0:

        binary[
            :int(
                height * IGNORE_TOP_FRACTION
            ),
            :
        ] = 0

    if IGNORE_BOTTOM_FRACTION > 0:

        binary[
            int(
                height * (
                    1 - IGNORE_BOTTOM_FRACTION
                )
            ):,
            :
        ] = 0

    # Remove long straight lines
    h_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                max(40, width // 4),
                1
            )
        ),
    )

    v_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                1,
                max(40, height // 4)
            )
        ),
    )

    binary = cv2.subtract(
        binary,
        cv2.bitwise_or(
            h_lines,
            v_lines
        )
    )

    # Connected components
    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    min_area = max(
        12,
        int(
            height
            * width
            * 0.00002
        )
    )

    areas = stats[
        :,
        cv2.CC_STAT_AREA
    ]

    candidate = areas >= min_area

    candidate[0] = False

    # Filter thin strokes
    if FILTER_THIN_STROKES and candidate.any():

        dist = cv2.distanceTransform(
            binary,
            cv2.DIST_L2,
            3
        )

        ink = binary > 0

        max_dist = np.zeros(
            n,
            dtype=np.float32
        )

        np.maximum.at(
            max_dist,
            labels[ink],
            dist[ink]
        )

        reference = float(
            np.percentile(
                max_dist[candidate],
                75
            )
        )

        if reference >= 2.5:

            heights = stats[
                :,
                cv2.CC_STAT_HEIGHT
            ]

            typical_h = float(
                np.median(
                    heights[candidate]
                )
            )

            thin = (
                max_dist
                < 0.65 * reference
            )

            small = (
                heights
                < 0.8 * typical_h
            )

            candidate &= ~(
                thin & small
            )

    binary = (
        candidate[labels] * 255
    ).astype(np.uint8)

    if not candidate.any():
        return [
            image.convert("RGB")
        ]

    char_h = float(
        np.median(
            stats[
                candidate,
                cv2.CC_STAT_HEIGHT
            ]
        )
    )

    # Join letters + words
    kw = max(
        15,
        int(
            char_h
            * JOIN_WIDTH_FACTOR
        )
    )

    kh = max(
        1,
        int(
            char_h * 0.15
        )
    )

    joined = cv2.dilate(
        binary,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                kw,
                kh
            )
        ),
    )

    n_blobs, blob_labels, blob_stats, _ = (
        cv2.connectedComponentsWithStats(
            joined,
            connectivity=8
        )
    )

    # One entry per blob
    blobs = []

    for i in range(1, n_blobs):

        x, y, w, h = (
            int(v)
            for v in blob_stats[
                i,
                :4
            ]
        )

        local = (
            (
                blob_labels[
                    y:y + h,
                    x:x + w
                ] == i
            )
            &
            (
                binary[
                    y:y + h,
                    x:x + w
                ] > 0
            )
        ).astype(np.uint8)

        ys, xs = np.nonzero(local)

        if len(xs) == 0:
            continue

        bw = int(
            xs.max()
            - xs.min()
            + 1
        )

        bh = int(
            ys.max()
            - ys.min()
            + 1
        )

        if (
            bh < char_h * 0.5
            or bw < char_h * 0.5
        ):
            continue

        blobs.append(
            (
                i,
                x,
                y,
                local,
                bh
            )
        )

    if not blobs:
        return [
            image.convert("RGB")
        ]

    line_h = float(
        np.median(
            [
                b[4]
                for b in blobs
            ]
        )
    )

    # Split blobs containing touching lines
    items = []

    for i, x, y, local, _ in blobs:

        for top, bottom in _split_rows(
            local,
            line_h
        ):

            part = local[
                top:bottom
            ]

            ys, xs = np.nonzero(
                part
            )

            if len(xs) == 0:
                continue

            box = (
                x + int(xs.min()),
                y + top + int(ys.min()),
                x + int(xs.max()) + 1,
                y + top + int(ys.max()) + 1,
            )

            bw = (
                box[2]
                - box[0]
            )

            bh = (
                box[3]
                - box[1]
            )

            if bh < char_h * 0.5:
                continue

            # Tall, thin strokes are brackets / vertical bars
            if (
                bw < char_h * 1.2
                and bh > 1.5 * bw
            ):
                continue

            items.append(
                {
                    "box": box,
                    "ids": {i}
                }
            )

    if not items:
        return [
            image.convert("RGB")
        ]

    # Merge fragments belonging to same row
    items = _merge_same_line(
        items,
        max_gap=(
            char_h
            * MERGE_GAP_FACTOR
        )
    )

    # Drop leftovers too narrow to be a text line
    items = [
        it
        for it in items
        if (
            it["box"][2]
            - it["box"][0]
        )
        >= char_h
        * MIN_LINE_WIDTH_FACTOR
    ]

    if not items:
        return [
            image.convert("RGB")
        ]

    # Reading order
    items.sort(
        key=lambda it:
            (
                it["box"][1]
                + it["box"][3]
            ) / 2
    )

    rows = []
    current = [items[0]]

    row_h = float(
        np.median(
            [
                it["box"][3]
                - it["box"][1]
                for it in items
            ]
        )
    )

    for it in items[1:]:

        cy = (
            it["box"][1]
            + it["box"][3]
        ) / 2

        prev_cy = np.mean(
            [
                (
                    p["box"][1]
                    + p["box"][3]
                ) / 2
                for p in current
            ]
        )

        if abs(cy - prev_cy) < row_h * 0.6:
            current.append(it)

        else:
            rows.append(current)
            current = [it]

    rows.append(current)

    ordered = [
        it
        for row in rows
        for it in sorted(
            row,
            key=lambda t:
                t["box"][0]
        )
    ]

    # Build crops
    clean_pil = Image.fromarray(rgb)

    pad_x = max(
        6,
        int(char_h * 0.3)
    )

    pad_y = max(
        4,
        int(char_h * 0.2)
    )

    crops = []
    final_boxes = []

    for it in ordered:

        x1, y1, x2, y2 = (
            it["box"]
        )

        x1 = max(
            0,
            x1 - pad_x
        )

        y1 = max(
            0,
            y1 - pad_y
        )

        x2 = min(
            width,
            x2 + pad_x
        )

        y2 = min(
            height,
            y2 + pad_y
        )

        if USE_CLEAN_BINARY_CROPS:

            own = np.isin(
                blob_labels[
                    y1:y2,
                    x1:x2
                ],
                list(it["ids"])
            )

            mask = (
                own
                &
                (
                    binary[
                        y1:y2,
                        x1:x2
                    ] > 0
                )
            )

            canvas = np.full(
                mask.shape,
                255,
                dtype=np.uint8
            )

            canvas[mask] = 0

            canvas = cv2.GaussianBlur(
                canvas,
                (3, 3),
                0
            )

            crop = (
                Image.fromarray(canvas)
                .convert("RGB")
            )

        else:

            crop = clean_pil.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

            if (
                INVERT_DARK_CROPS
                and np.mean(
                    np.array(
                        crop.convert("L")
                    )
                ) < 127
            ):
                crop = ImageOps.invert(
                    crop
                )

        crops.append(crop)

        final_boxes.append(
            (
                x1,
                y1,
                x2,
                y2
            )
        )

    # Debug image
    debug = cv2.cvtColor(
        np.array(
            image.convert("RGB")
        ),
        cv2.COLOR_RGB2BGR
    )

    for k, (
        x1,
        y1,
        x2,
        y2
    ) in enumerate(
        final_boxes,
        1
    ):

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2 - 1, y2 - 1),
            (0, 255, 0),
            2
        )

        cv2.putText(
            debug,
            str(k),
            (
                x1,
                max(12, y1 - 4)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(
        str(
            BASE_DIR
            / "debug_lines.jpg"
        ),
        debug
    )

    return (
        crops
        if crops
        else [
            image.convert("RGB")
        ]
    )


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------

def recognize_line(image: Image.Image):

    pixel_values = line_processor(
        images=image.convert("RGB"),
        return_tensors="pt",
    ).pixel_values.to(DEVICE)

    with torch.inference_mode():

        generated_ids = line_model.generate(
            pixel_values,
            max_new_tokens=128,
        )

    return line_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "device": str(DEVICE),
        "word_model_loaded": (
            word_model is not None
        ),
        "line_model_loaded": (
            line_model is not None
        ),
    }


# ---------------------------------------------------------------------------
# OCR API
# ---------------------------------------------------------------------------

@app.post("/ocr/{model_type}")
async def run_ocr(
    model_type: str,
    file: UploadFile = File(...),
):

    if model_type == "word":

        processor = word_processor
        model = word_model

    elif model_type == "line":

        processor = line_processor
        model = line_model

    else:

        raise HTTPException(
            status_code=400,
            detail=(
                "model_type must be "
                "'word' or 'line'"
            ),
        )

    if (
        processor is None
        or model is None
    ):

        raise HTTPException(
            status_code=503,
            detail="OCR model is not loaded",
        )

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    try:

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a valid "
                "image file"
            ),
        )

    try:

        # ---------------------------------------------------------------
        # Line OCR
        # ---------------------------------------------------------------

        if model_type == "line":

            crops = segment_lines(
                image
            )

            extracted = [
                recognize_line(crop)
                for crop in crops
            ]

            extracted = [
                line
                for line in extracted
                if line
            ]

            return {
                "text": "\n".join(
                    extracted
                ),
                "lines_detected": len(
                    crops
                ),
                "lines_recognized": len(
                    extracted
                ),
                "model": "line",
                "filename": file.filename,
            }

        # ---------------------------------------------------------------
        # Word OCR
        # ---------------------------------------------------------------

        pixel_values = processor(
            images=image,
            return_tensors="pt",
        ).pixel_values.to(DEVICE)

        with torch.inference_mode():

            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=128,
            )

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        return {
            "text": text,
            "model": "word",
            "filename": file.filename,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"OCR inference failed: {str(exc)}"
            ),
        ) from exc