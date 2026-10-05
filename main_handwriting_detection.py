"""ScanMind CV service.

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8001
"""

import base64
import json
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from ocr import run_ocr
from pipeline import process_image, quality_report
from handwriting_detector import detect_handwriting

app = FastAPI(title="ScanMind CV Service")


def read_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return image


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/quality")
async def quality(file: UploadFile = File(...)):
    return quality_report(read_image(await file.read()))


@app.post("/detect-handwriting")
async def detect_handwriting_endpoint(file: UploadFile = File(...)):
    """Classify an image as handwritten, printed, or uncertain."""
    image = read_image(await file.read())
    try:
        return detect_handwriting(image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Handwriting detection failed: {e}",
        )


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    mode: str = Form("scan"),
    corners: Optional[str] = Form(None),
):
    if mode not in ("scan", "gray", "color"):
        raise HTTPException(
            status_code=400,
            detail="mode must be scan, gray, or color",
        )

    image = read_image(await file.read())
    parsed = json.loads(corners) if corners else None
    out = process_image(image, mode=mode, corners=parsed)

    ok, buf = cv2.imencode(".png", out["image"])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")

    return {
        "image_base64": base64.b64encode(buf).decode(),
        "corners": out["corners"],
        "document_found": out["document_found"],
        "quality": out["quality"],
    }


@app.post("/process/image")
async def process_as_image(
    file: UploadFile = File(...),
    mode: str = Form("scan"),
):
    image = read_image(await file.read())
    out = process_image(image, mode=mode)
    ok, buf = cv2.imencode(".png", out["image"])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return Response(content=buf.tobytes(), media_type="image/png")


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    preprocess: bool = Form(True),
    mode: str = Form("scan"),
    lang: str = Form("eng"),
):
    image = read_image(await file.read())

    if preprocess:
        image = process_image(image, mode=mode)["image"]

    try:
        return run_ocr(image, lang=lang)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR failed. Is Tesseract installed? Details: {e}",
        )
