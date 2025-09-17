from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from typing import Optional, List, Dict, Any
import logging
import traceback
import io
from PIL import Image
import pillow_heif
import os
import base64
import json as _json

from batchgrids.vision.image_processor import ImageProcessor
from batchgrids.vision.yolo_segmenter import YOLOSegmenter

logger = logging.getLogger(__name__)

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

router = APIRouter()


def process_uploaded_image(file_content: bytes, max_size: int = 2048) -> np.ndarray:
    """
    Process uploaded image with support for HEIC and automatic resizing.
    
    Args:
        file_content: Raw file bytes
        max_size: Maximum dimension (width or height) for resizing
        
    Returns:
        OpenCV image array (BGR format)
        
    Raises:
        ValueError: If image format is not supported or processing fails
    """
    try:
        # First, try to open with PIL (supports HEIC, WebP, and other formats)
        logger.info("Attempting to decode image with PIL/Pillow")
        pil_image = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if pil_image.mode not in ('RGB', 'BGR'):
            logger.info(f"Converting image from {pil_image.mode} to RGB")
            pil_image = pil_image.convert('RGB')
        
        # Get original dimensions
        original_width, original_height = pil_image.size
        logger.info(f"Original image size: {original_width}x{original_height}")
        
        # Resize if too large
        if max(original_width, original_height) > max_size:
            # Calculate new dimensions maintaining aspect ratio
            if original_width > original_height:
                new_width = max_size
                new_height = int(original_height * max_size / original_width)
            else:
                new_height = max_size  
                new_width = int(original_width * max_size / original_height)
            
            logger.info(f"Resizing image to {new_width}x{new_height}")
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert PIL Image to OpenCV format (BGR)
        image_rgb = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        logger.info(f"Successfully processed image: final size {image_bgr.shape[1]}x{image_bgr.shape[0]}")
        return image_bgr
        
    except Exception as e:
        logger.error(f"PIL processing failed: {e}")
        
        # Fallback to OpenCV decoding (for standard JPEG/PNG)
        logger.info("Falling back to OpenCV decoding")
        nparr = np.frombuffer(file_content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Unable to decode image - unsupported format or corrupted file")
        
        # Resize if too large
        h, w = image.shape[:2]
        if max(w, h) > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            
            logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        logger.info(f"Successfully processed image with OpenCV: final size {image.shape[1]}x{image.shape[0]}")
        return image


@router.post("/process_manual_scale")
async def process_manual_scale(
    file: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
    distance_mm: float = Form(...),
    orig_w: Optional[float] = Form(None),
    orig_h: Optional[float] = Form(None),
):
    """Process an image using manual scale picked from two points on a ruler.

    Returns SVG outline and basic dimensions.
    """
    try:
        logger.info(f"Processing image: {file.filename}, scale: {distance_mm}mm, points: ({x1},{y1})-({x2},{y2})")
        
        if not file.filename or distance_mm <= 0:
            logger.error(f"Invalid input: filename={file.filename}, distance_mm={distance_mm}")
            raise HTTPException(status_code=400, detail="Invalid input")

        contents = await file.read()
        logger.info(f"Read {len(contents)} bytes from uploaded file")
        
        # Use enhanced image processing with HEIC support and auto-resizing
        try:
            image = process_uploaded_image(contents, max_size=2048)
        except ValueError as e:
            logger.error(f"Failed to process image: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        logger.info(f"Processed image shape: {image.shape}")

        # If original client image size provided, rescale points into processed coordinates
        try:
            if orig_w and orig_h and image is not None:
                sx = float(image.shape[1]) / float(orig_w)
                sy = float(image.shape[0]) / float(orig_h)
                x1, y1 = float(x1) * sx, float(y1) * sy
                x2, y2 = float(x2) * sx, float(y2) * sy
        except Exception:
            pass

        # Pixel distance between clicked points
        px = float(np.hypot(x2 - x1, y2 - y1))
        if px <= 0:
            logger.error(f"Zero-length scale selection: px={px}")
            raise HTTPException(status_code=400, detail="Zero-length scale selection")

        px_per_mm = px / float(distance_mm)
        logger.info(f"Calculated scale: {px_per_mm:.3f} px/mm")

        processor = ImageProcessor()
        result = processor.process_image(image, px_per_mm_override=px_per_mm)
        
        logger.info(f"Processing result: success={result.success}, needs_review={result.needs_review}")
        
        if not result.success:
            logger.error(f"Processing failed: {result.error_message}")
            raise HTTPException(status_code=500, detail=result.error_message or "Processing failed")

        data = {
            "px_per_mm": px_per_mm,
            "needs_review": result.needs_review,
            "review_reason": result.review_reason,
            "dimensions": result.tool_dimensions or {},
            "svg": result.svg_outline or "",
        }
        logger.info("Successfully processed image and returning results")
        return JSONResponse(data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_manual_scale: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/interactive_tool_outline")
async def interactive_tool_outline(
    file: UploadFile = File(...),
    points: str = Form(...),  # JSON string: [{"x": int, "y": int}, ...] foreground seeds
    bg_points: Optional[str] = Form(None),  # optional JSON string for background seeds
    prev_mask: Optional[str] = Form(None),  # optional base64 PNG of previous binary mask
    roi: Optional[str] = Form(None),        # optional JSON: {x,y,w,h}
    yolo_mask: Optional[str] = Form(None),  # optional base64 PNG of YOLO mask
    smoothing_level: Optional[float] = Form(None),  # 0-100 smoothing control
    offset_mm: Optional[float] = Form(None),        # outline offset in mm (can be negative)
    px_per_mm: Optional[float] = Form(None),
):
    """Detect tool outline by seeding a segmentation with user clicks.

    Strategy:
    1) Build a GrabCut trimap from user foreground seeds (and optional background seeds), plus border as background.
    2) Run GrabCut to obtain a clean foreground mask.
    3) Post-process with morphology and contour selection guided by seed inclusion.
    4) Convert to polygon, dimensions and SVG.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        contents = await file.read()
        image = process_uploaded_image(contents, max_size=2048)

        # Parse seed points
        try:
            import json as _json
            seed_list = _json.loads(points)
            seeds = [(int(p["x"]), int(p["y"])) for p in seed_list]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid points payload")

        if not seeds:
            raise HTTPException(status_code=400, detail="At least one seed point required")

        # Parse optional ROI
        rx = ry = 0
        rw, rh = image.shape[1], image.shape[0]
        if roi:
            try:
                roi_obj = _json.loads(roi)
                rx = max(0, int(roi_obj.get("x", 0)))
                ry = max(0, int(roi_obj.get("y", 0)))
                rw = int(roi_obj.get("w", image.shape[1] - rx))
                rh = int(roi_obj.get("h", image.shape[0] - ry))
                # clamp
                rw = max(1, min(rw, image.shape[1] - rx))
                rh = max(1, min(rh, image.shape[0] - ry))
            except Exception:
                rx = ry = 0
                rw, rh = image.shape[1], image.shape[0]

        # Crop to ROI for processing
        roi_img = image[ry:ry+rh, rx:rx+rw].copy()
        h, w = roi_img.shape[:2]
        grab_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

        # Border as definite background (thin frame)
        border = max(2, int(0.01 * min(h, w)))
        grab_mask[:border, :] = cv2.GC_BGD
        grab_mask[-border:, :] = cv2.GC_BGD
        grab_mask[:, :border] = cv2.GC_BGD
        grab_mask[:, -border:] = cv2.GC_BGD

        # If previous mask provided, bias towards prior foreground
        if prev_mask:
            import base64
            try:
                pm_bytes = base64.b64decode(prev_mask)
                pm_np = np.frombuffer(pm_bytes, np.uint8)
                pm_img = cv2.imdecode(pm_np, cv2.IMREAD_GRAYSCALE)
                if pm_img is not None:
                    pm_img = (pm_img > 127).astype(np.uint8)
                    # Crop prior mask to ROI if full-size
                    if pm_img.shape[:2] != (h, w):
                        pm_img = pm_img[ry:ry+rh, rx:rx+rw]
                    # Dilate slightly to encourage continuity
                    pm_img = cv2.morphologyEx(pm_img, cv2.MORPH_DILATE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                              iterations=1)
                    grab_mask[pm_img > 0] = cv2.GC_PR_FGD
            except Exception as _e:
                logger.warning(f"Failed to decode prev_mask: {_e}")

        # If YOLO mask provided, mark as probable foreground in ROI
        if yolo_mask:
            try:
                ym_bytes = base64.b64decode(yolo_mask)
                ym_np = np.frombuffer(ym_bytes, np.uint8)
                ym_img = cv2.imdecode(ym_np, cv2.IMREAD_GRAYSCALE)
                if ym_img is not None:
                    if ym_img.shape[:2] != (h, w):
                        ym_img = ym_img[ry:ry+rh, rx:rx+rw]
                    grab_mask[ym_img > 127] = cv2.GC_PR_FGD
            except Exception as _e:
                logger.warning(f"Failed to decode yolo_mask: {_e}")

        # Seed disks
        seed_radius = max(3, int(0.01 * min(h, w)))
        for (sx, sy) in seeds:
            # shift into ROI
            sx_r, sy_r = int(sx - rx), int(sy - ry)
            if 0 <= sx_r < w and 0 <= sy_r < h:
                cv2.circle(grab_mask, (sx_r, sy_r), seed_radius, cv2.GC_FGD, thickness=-1)

        # Optional background seeds
        if bg_points:
            try:
                import json as _json
                bg_list = _json.loads(bg_points)
                for bp in bg_list:
                    bx, by = int(bp["x"]) - rx, int(bp["y"]) - ry
                    if 0 <= bx < w and 0 <= by < h:
                        cv2.circle(grab_mask, (bx, by) , seed_radius, cv2.GC_BGD, thickness=-1)
            except Exception:
                pass

        # Run GrabCut with mask initialization
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(roi_img, grab_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        except Exception as e:
            logger.warning(f"GrabCut failed, falling back to edges: {e}")
            # Fallback to edges-only path
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img.copy()
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(gray, 30, 100)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            edges = cv2.dilate(edges, kernel, iterations=2)
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise HTTPException(status_code=422, detail="No contours detected")
            # Pick closest to first seed, or largest if no seeds
            if seeds:
                sx, sy = seeds[0]
                sx_r, sy_r = float(sx - rx), float(sy - ry)
                cnt = min(contours, key=lambda c: abs(cv2.pointPolygonTest(c, (sx_r, sy_r), True)))
            else:
                cnt = max(contours, key=cv2.contourArea)
            local_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(local_mask, [cnt], -1, 1, thickness=cv2.FILLED)
        else:
            # Build binary mask from GrabCut result
            local_mask = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

            # Keep region that best matches seeds (if multiple blobs)
            contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                local_mask = np.zeros((h, w), dtype=np.uint8)
                if not seeds:
                    best = max(contours, key=cv2.contourArea)
                    cv2.drawContours(local_mask, [best], -1, 1, thickness=cv2.FILLED)
                else:
                    best = None
                    best_score = -1e9
                    for cnt in contours:
                        score = 0.0
                        area = max(1.0, cv2.contourArea(cnt))
                        # Prefer contours containing seeds and with reasonable size
                        for (sx, sy) in seeds:
                            v = cv2.pointPolygonTest(cnt, (float(sx - rx), float(sy - ry)), True)
                            if v >= 0:
                                score += 5.0
                            else:
                                score -= min(5.0, abs(v) / 10.0)
                        score += min(5.0, area / (h * w * 0.05))  # mild area prior
                        if score > best_score:
                            best_score = score
                            best = cnt
                    if best is not None:
                        cv2.drawContours(local_mask, [best], -1, 1, thickness=cv2.FILLED)

        # Map local_mask back to full image coordinates
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask[ry:ry+rh, rx:rx+rw] = (local_mask > 0).astype(np.uint8)

        # Use YOLOSegmenter helpers (no model load) to create outline and SVG
        seg = YOLOSegmenter(load_model=False)
        # Apply optional smoothing override
        try:
            lvl = float(smoothing_level) if smoothing_level is not None else None
        except Exception:
            lvl = None
        if lvl is not None:
            # Map smoothing level to parameters
            try:
                from math import isfinite
                if isfinite(lvl):
                    if lvl < 0: lvl = 0
                    if lvl > 100: lvl = 100
                    # Use conservative ranges to avoid oversmoothing and contour loss
                    seg.simplify_epsilon_ratio = 0.001 + (lvl / 100.0) * 0.007  # 0.001..0.008
                    seg.smooth_buffer_px = int(round((lvl / 100.0) * 6.0))      # 0..6 px
                    seg.smooth_buffer_max_px = 8                                # cap rounding radius
                    seg.chaikin_iterations = 0 if lvl < 50 else 1               # add one pass at higher levels
                    seg.max_outline_points = 400
                    seg.svg_use_smooth_curves = False if lvl == 0 else True
            except Exception:
                pass
        pxmm = float(px_per_mm) if px_per_mm and float(px_per_mm) > 0 else 1.0
        tool = seg.extract_tool_outline(mask, pxmm)
        outline = tool.get("outline", [])
        dims = tool.get("dimensions", {})
        # Apply offset if requested
        try:
            off = float(offset_mm) if offset_mm is not None else 0.0
        except Exception:
            off = 0.0
        offset_outline = seg._offset_outline(outline, off, pxmm) if outline and abs(off) > 1e-6 else outline
        svg = seg.create_svg_outline(offset_outline, dims, pxmm) if offset_outline else ""

        # Encode mask as base64 PNG for iterative refinement on client
        try:
            png_ok, png_buf = cv2.imencode('.png', (mask * 255).astype(np.uint8))
            mask_png_b64 = None
            if png_ok:
                import base64 as _b64
                mask_png_b64 = _b64.b64encode(png_buf.tobytes()).decode('ascii')
        except Exception:
            mask_png_b64 = None

        return JSONResponse({
            "outline": outline,
            "dimensions": dims,
            "svg": svg,
            "px_per_mm": pxmm,
            "mask_png": mask_png_b64,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "offset_outline": offset_outline,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interactive selection failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def _b64_image(image: np.ndarray) -> str:
    """Encode BGR image to base64 JPEG data URI for vision models."""
    try:
        ok, buf = cv2.imencode('.jpg', image)
        if not ok:
            return ""
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


ALLOWED_TOOL_LABELS: List[str] = [
    "side cutters","diagonal cutters","needle-nose pliers","lineman pliers","slip-joint pliers",
    "adjustable wrench","combination wrench","socket wrench","ratchet",
    "screwdriver flat","screwdriver phillips","box cutter","utility knife",
    "tape measure","chisel","hammer","allen key","hex key","torx driver",
    "file","hacksaw","snips","tin snips","pipe wrench","vise grip",
    "wire stripper","caliper"
]


def _ask_ai_for_tools(image: np.ndarray) -> list[str]:
    """Call OpenAI Vision to list tools in the photo as short class names.

    Returns a list like ["pliers", "screwdriver"]. Falls back to [].
    Requires OPENAI_API_KEY and openai package installed.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.info("OPENAI_API_KEY not set; skipping AI guidance")
        return []
    try:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=api_key)
            image_url = _b64_image(image)
            if not image_url:
                return []
            prompt = (
                "You are an expert identifying physical workshop tools for segmentation.\n"
                "Rules:\n"
                "- Return ONLY a JSON array of short tool names, lowercase (e.g., [\"pliers\", \"adjustable wrench\"]).\n"
                "- Include only concrete tools in the frame.\n"
                "- EXCLUDE: apriltags, rulers, cutting mats, backgrounds, shadows, hands, text, logos, boxes, paper.\n"
                "- Prefer specific types when obvious (\"needle-nose pliers\", \"philips screwdriver\").\n"
                "- If no tools, return []."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You return only valid JSON arrays of strings."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                temperature=0,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            text = resp.choices[0].message.content.strip() if resp and resp.choices else ""
            try:
                arr = _json.loads(text)
                if isinstance(arr, list):
                    return [str(x).strip().lower() for x in arr if isinstance(x, (str, int, float))]
            except Exception:
                # Try to recover array between brackets
                import re
                m = re.search(r"\[(.*?)\]", text, re.S)
                if m:
                    arr_text = f"[{m.group(1)}]"
                    arr = _json.loads(arr_text)
                    if isinstance(arr, list):
                        return [str(x).strip().lower() for x in arr]
        except Exception as e:
            logger.warning(f"AI guidance failed: {e}")
            return []
    except Exception:
        return []
    return []


def _ask_ai_for_tool_boxes(image: np.ndarray) -> List[Dict[str, Any]]:
    """Ask AI for tool-only labels with normalized boxes.

    Returns list of {label:str, conf:float, box:[x0,y0,x1,y1]} with 0-1 normalized coords.
    Filters labels to ALLOWED_TOOL_LABELS. Returns [] on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)
        image_url = _b64_image(image)
        if not image_url:
            return []
        allowed = ", ".join([f'"{lbl}"' for lbl in ALLOWED_TOOL_LABELS])
        prompt = (
            "Identify ONLY workshop hand tools in this image. Follow these rules:\n"
            "- Allowed labels: [" + allowed + "].\n"
            "- Exclude everything else (people, animals, vehicles, rulers, AprilTags, mats, backgrounds, shadows, logos, boxes, paper).\n"
            "- Return only this JSON (no extra text):\n"
            '{"tools":[{"label":"<one of allowed labels>","conf":0.00,"box":[x0,y0,x1,y1]}]}\n'
            "- box is normalized 0–1 (x0<x1, y0<y1), tight around each tool.\n"
            "- conf is 0–1. If no tools, return {\"tools\":[]}.")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in identifying workshop hand tools. Return only valid JSON."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        text = resp.choices[0].message.content.strip() if resp and resp.choices else ""
        data = _json.loads(text)
        tools = data.get("tools", []) if isinstance(data, dict) else []
        out: List[Dict[str, Any]] = []
        for t in tools:
            try:
                label = str(t.get("label", "")).strip().lower()
                if label not in ALLOWED_TOOL_LABELS:
                    continue
                conf = float(t.get("conf", 0.0))
                box = t.get("box", [])
                if not (isinstance(box, list) and len(box) == 4):
                    continue
                x0, y0, x1, y1 = [float(v) for v in box]
                if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                    continue
                out.append({"label": label, "conf": conf, "box": [x0, y0, x1, y1]})
            except Exception:
                continue
        return out
    except Exception as e:
        logger.warning(f"AI box guidance failed: {e}")
        return []


def _refine_with_grabcut(image: np.ndarray, yolo_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Refine a YOLO mask with GrabCut constrained to the bbox ROI.

    Returns a binary mask the same size as image (0/1).
    """
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))
    y2 = max(y1 + 1, min(y2, h_img))

    # Pad ROI a bit to capture edges
    pad = int(max(5, 0.02 * max(w_img, h_img)))
    rx = max(0, x1 - pad)
    ry = max(0, y1 - pad)
    rw = min(w_img, x2 + pad) - rx
    rh = min(h_img, y2 + pad) - ry

    roi_img = image[ry:ry+rh, rx:rx+rw]
    roi_mask = yolo_mask[ry:ry+rh, rx:rx+rw].astype(np.uint8)

    # Build grabcut mask (probable bg default)
    gc_mask = np.full((rh, rw), cv2.GC_PR_BGD, dtype=np.uint8)
    # Border as definite background
    border = max(2, int(0.01 * min(rh, rw)))
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD
    # YOLO mask as PR_FGD if present, otherwise build a weak prior from intensity
    if np.any(roi_mask > 0):
        gc_mask[roi_mask > 0] = cv2.GC_PR_FGD
    else:
        # Otsu threshold to get a rough foreground in ROI
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img.copy()
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Pick the larger of th / inverted th as FG prior
        if np.sum(th > 0) < (th.size // 2):
            th = cv2.bitwise_not(th)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        gc_mask[th > 0] = cv2.GC_PR_FGD

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi_img, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        local = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
        # Clean up
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, k, iterations=2)
        local = cv2.morphologyEx(local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    except Exception:
        # Fallback: cleaned YOLO mask in ROI
        local = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Map back to full image
    out = np.zeros((h_img, w_img), dtype=np.uint8)
    out[ry:ry+rh, rx:rx+rw] = (local > 0).astype(np.uint8)
    return out


def _match_hints(class_name: str, hints: list[str]) -> bool:
    cn = (class_name or "").lower()
    for h in hints:
        h = h.lower()
        if h in cn or cn in h:
            return True
        # simple synonyms
        if h.startswith("needle") and "pliers" in cn:
            return True
        if h in ("utility knife", "box cutter") and ("knife" in cn or "cutter" in cn):
            return True
    return False


@router.post("/ai_guided_detection")
async def ai_guided_detection(
    file: UploadFile = File(...),
    px_per_mm: Optional[float] = Form(None),
    use_ai: bool = Form(True),
    smoothing_level: Optional[float] = Form(None),
    offset_mm: Optional[float] = Form(None),
):
    """Detect tools using AI guidance (ChatGPT Vision) to prioritize classes, then YOLO segmentation.

    - Returns AI hints, matched detections (outline, dims, bbox, class, confidence), and px_per_mm.
    - Scale can use AprilTags (if present) via ImageProcessor or supplied px_per_mm.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        contents = await file.read()
        image = process_uploaded_image(contents, max_size=2048)

        # Determine scale
        pxmm_override = float(px_per_mm) if px_per_mm and float(px_per_mm) > 0 else None
        processor = ImageProcessor()
        result = processor.process_image(image, px_per_mm_override=pxmm_override)

        used_pxmm = result.calibration.px_per_mm if (result.calibration and result.calibration.px_per_mm and result.calibration.px_per_mm > 0) else (pxmm_override or 1.0)

        # Ask AI for tool boxes (tool-only taxonomy)
        hints: list[str] = []
        ai_boxes: List[Dict[str, Any]] = []
        if use_ai:
            ai_boxes = _ask_ai_for_tool_boxes(image)
            hints = [t["label"] for t in ai_boxes]

        # Run YOLO segmentation (for mask priors); tolerate failure
        try:
            if result.yolo_prediction is None:
                from batchgrids.vision.yolo_segmenter import YOLOSegmenter
                yolo = YOLOSegmenter()
                ypred = yolo.segment_image(image)
            else:
                ypred = result.yolo_prediction
        except Exception:
            ypred = None

        # Filter detections by AI hints if any
        detections = []
        from batchgrids.vision.yolo_segmenter import YOLOSegmenter as _SEG
        seg = _SEG(load_model=False)
        # Apply optional smoothing override
        try:
            lvl = float(smoothing_level) if smoothing_level is not None else None
        except Exception:
            lvl = None
        if lvl is not None:
            try:
                from math import isfinite
                if isfinite(lvl):
                    if lvl < 0: lvl = 0
                    if lvl > 100: lvl = 100
                    seg.simplify_epsilon_ratio = 0.001 + (lvl / 100.0) * 0.007
                    seg.smooth_buffer_px = int(round((lvl / 100.0) * 6.0))
                    seg.smooth_buffer_max_px = 8
                    seg.chaikin_iterations = 0 if lvl < 50 else 1
                    seg.max_outline_points = 400
                    seg.svg_use_smooth_curves = False if lvl == 0 else True
            except Exception:
                pass
        if ai_boxes:
            H, W = image.shape[:2]
            # Build a simple YOLO prior mask per ROI if available
            yolo_masks: List[np.ndarray] = []
            if ypred is not None:
                try:
                    for d in ypred.detections:
                        yolo_masks.append(d.mask.astype(np.uint8))
                except Exception:
                    yolo_masks = []
            for t in ai_boxes:
                x0, y0, x1, y1 = t["box"]
                bb = (int(x0*W), int(y0*H), int(x1*W), int(y1*H))
                # Build a prior mask from YOLO detections that intersect ROI
                prior = np.zeros((H, W), dtype=np.uint8)
                if yolo_masks:
                    rx0, ry0, rx1, ry1 = bb
                    for m in yolo_masks:
                        # quick overlap test via any nonzero in ROI
                        roi_m = m[ry0:ry1, rx0:rx1]
                        if roi_m.size > 0 and np.any(roi_m > 0):
                            prior = np.maximum(prior, m.astype(np.uint8))
                # If no prior, start with empty; _refine_with_grabcut will use ROI bbox
                refined = _refine_with_grabcut(image, prior, bb)
                tool = seg.extract_tool_outline(refined, used_pxmm)
                outline = tool.get("outline", [])
                dims = tool.get("dimensions", {})
                # Offset handling
                try:
                    off = float(offset_mm) if offset_mm is not None else 0.0
                except Exception:
                    off = 0.0
                offset_outline = seg._offset_outline(outline, off, used_pxmm) if outline and abs(off) > 1e-6 else outline
                svg = seg.create_svg_outline(offset_outline, dims, used_pxmm) if offset_outline else ""
                detections.append({
                    "class_name": t.get("label", "tool"),
                    "confidence": float(t.get("conf", 0.0)),
                    "bbox": [bb[0], bb[1], bb[2], bb[3]],
                    "outline": outline,
                    "offset_outline": offset_outline,
                    "dimensions": dims,
                    "svg": svg,
                })
        elif ypred is not None:
            for det in ypred.detections:
                # No AI boxes: keep previous behavior but still refine masks
                refined = _refine_with_grabcut(image, det.mask.astype(np.uint8), det.bbox)
                tool = seg.extract_tool_outline(refined, used_pxmm)
                outline = tool.get("outline", [])
                dims = tool.get("dimensions", {})
                try:
                    off = float(offset_mm) if offset_mm is not None else 0.0
                except Exception:
                    off = 0.0
                offset_outline = seg._offset_outline(outline, off, used_pxmm) if outline and abs(off) > 1e-6 else outline
                svg = seg.create_svg_outline(offset_outline, dims, used_pxmm) if offset_outline else ""
                detections.append({
                    "class_name": det.class_name,
                    "confidence": float(det.confidence),
                    "bbox": det.bbox,
                    "outline": outline,
                    "offset_outline": offset_outline,
                    "dimensions": dims,
                    "svg": svg,
                })

        # Determine top detection for returning a mask useful for ROI fusion
        top_det = None
        if detections:
            # match by confidence among matched ones
            matched_src = [d for d in ypred.detections if _match_hints(d.class_name, hints)] if hints else list(ypred.detections)
            if matched_src:
                top_det = max(matched_src, key=lambda d: d.confidence)
        elif ypred.detections:
            top_det = max(ypred.detections, key=lambda d: d.confidence)

        top_mask_b64 = None
        if top_det is not None and top_det.mask is not None:
            try:
                # Provide refined mask as guidance for ROI-based flows
                m_ref = _refine_with_grabcut(image, top_det.mask.astype(np.uint8), top_det.bbox)
                m = (m_ref.astype(np.uint8) * 255)
                ok, buf = cv2.imencode('.png', m)
                if ok:
                    top_mask_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            except Exception:
                top_mask_b64 = None

        detections.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)

        return JSONResponse({
            "px_per_mm": used_pxmm,
            "hints": hints,
            "detections": detections,
            "top_mask_png": top_mask_b64,
            "needs_review": result.needs_review,
            "review_reason": result.review_reason,
            "image_width": int(ypred.image_shape[1]) if ypred and ypred.image_shape else int(image.shape[1]),
            "image_height": int(ypred.image_shape[0]) if ypred and ypred.image_shape else int(image.shape[0]),
        })

    except HTTPException:
        raise
    except Exception as e:
        # Fall back to YOLO-only path instead of 500 to keep UX smooth
        try:
            logger.error(f"AI-guided detection failed; falling back to YOLO-only. Error: {e}")
            # Use the already-read image from the main try block above
            # Don't re-read the file as FastAPI UploadFile can't be read twice
            pxmm_override = float(px_per_mm) if px_per_mm and float(px_per_mm) > 0 else None
            processor = ImageProcessor()
            res = processor.process_image(image, px_per_mm_override=pxmm_override)
            used_pxmm = res.calibration.px_per_mm if (res.calibration and res.calibration.px_per_mm and res.calibration.px_per_mm > 0) else (pxmm_override or 1.0)
            from batchgrids.vision.yolo_segmenter import YOLOSegmenter as _SEG
            seg = _SEG(load_model=False)
            dets = []
            if res.yolo_prediction:
                for det in res.yolo_prediction.detections:
                    mask_ref = _refine_with_grabcut(image, det.mask.astype(np.uint8), det.bbox)
                    tool = seg.extract_tool_outline(mask_ref, used_pxmm)
                    outline = tool.get("outline", [])
                    dims = tool.get("dimensions", {})
                    # Optional offset
                    try:
                        off = float(offset_mm) if offset_mm is not None else 0.0
                    except Exception:
                        off = 0.0
                    offset_outline = outline
                    if outline and abs(off) > 1e-6:
                        offset_outline = seg._offset_outline(outline, off, used_pxmm)
                    svg = seg.create_svg_outline(offset_outline, dims, used_pxmm) if offset_outline else ""
                    dets.append({
                        "class_name": det.class_name,
                        "confidence": float(det.confidence),
                        "bbox": det.bbox,
                        "outline": outline,
                        "offset_outline": offset_outline,
                        "dimensions": dims,
                        "svg": svg,
                    })
            dets.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
            return JSONResponse({
                "px_per_mm": used_pxmm,
                "hints": [],
                "detections": dets,
                "top_mask_png": None,
                "needs_review": res.needs_review,
                "review_reason": res.review_reason,
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
            })
        except Exception as e2:
            logger.error(f"YOLO-only fallback failed: {e2}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="AI and fallback segmentation failed")


@router.post("/full_detection")
async def full_detection(
    file: UploadFile = File(...),
    px_per_mm: Optional[float] = Form(None),
):
    """Run the full detection pipeline and return all detected tools.

    - Uses AprilTag calibration if available; otherwise uses provided px_per_mm, else 1.0.
    - Returns list of detections with outline, dims, bbox, class and confidence.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        contents = await file.read()
        image = process_uploaded_image(contents, max_size=2048)

        processor = ImageProcessor()
        pxmm_override = float(px_per_mm) if px_per_mm and float(px_per_mm) > 0 else None
        result = processor.process_image(image, px_per_mm_override=pxmm_override)

        if result.yolo_prediction is None:
            return JSONResponse({"detections": [], "px_per_mm": result.calibration.px_per_mm or (pxmm_override or 1.0)})

        used_pxmm = result.calibration.px_per_mm if (result.calibration and result.calibration.px_per_mm and result.calibration.px_per_mm > 0) else (pxmm_override or 1.0)

        seg = YOLOSegmenter(load_model=False)
        dets = []
        for det in result.yolo_prediction.detections:
            tool = seg.extract_tool_outline(det.mask.astype(np.uint8), used_pxmm)
            outline = tool.get("outline", [])
            dims = tool.get("dimensions", {})
            svg = seg.create_svg_outline(outline, dims, used_pxmm) if outline else ""
            dets.append({
                "class_name": det.class_name,
                "confidence": float(det.confidence),
                "bbox": det.bbox,
                "outline": outline,
                "dimensions": dims,
                "svg": svg,
            })

        # Sort by confidence descending
        dets.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)

        return JSONResponse({
            "px_per_mm": used_pxmm,
            "detections": dets,
            "needs_review": result.needs_review,
            "review_reason": result.review_reason,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full detection failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
