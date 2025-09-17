#!/usr/bin/env python3
"""Generate printable AprilTag images and an optional PDF sheet.

Uses OpenCV's aruco module to render the AprilTag 36h11 family.
Requires opencv-contrib-python.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
    import reportlab
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import mm
except Exception:
    reportlab = None

from PIL import Image, ImageDraw, ImageFont


FAMILY_MAP = {
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def parse_ids(ids_arg: str, range_arg: Optional[str]) -> List[int]:
    if ids_arg:
        ids = []
        for tok in ids_arg.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                ids.extend(list(range(int(a), int(b) + 1)))
            else:
                ids.append(int(tok))
        return sorted(set(ids))
    if range_arg:
        a, b = range_arg.split("-", 1)
        return list(range(int(a), int(b) + 1))
    # Default: first 4 tags used by our mat
    return [0, 1, 2, 3]


def mm_to_px(mm_val: float, dpi: int) -> int:
    return int(round(mm_val * dpi / 25.4))


def generate_tag_image(tag_id: int, size_px: int, family_name: str) -> np.ndarray:
    if family_name not in FAMILY_MAP:
        raise ValueError(f"Unsupported family: {family_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(FAMILY_MAP[family_name])
    # Generate marker image (API varies by OpenCV version)
    size_px = int(size_px)
    tag_id = int(tag_id)
    tag = None
    if hasattr(cv2.aruco, "drawMarker"):
        tag = cv2.aruco.drawMarker(dictionary, tag_id, size_px)
    elif hasattr(cv2.aruco, "generateImageMarker"):
        tag = cv2.aruco.generateImageMarker(dictionary, tag_id, size_px)
    else:
        raise RuntimeError("OpenCV aruco module lacks drawMarker/generateImageMarker. Update opencv-contrib-python.")
    return tag


def save_tag_png(img: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def build_pdf_sheet(
    out_pdf: Path,
    tag_paths: List[Tuple[int, Path]],
    page_size: str = "letter",
    margin_mm: float = 10.0,
    spacing_mm: float = 10.0,
    tags_per_row: int = 4,
):
    if reportlab is None:
        raise RuntimeError("reportlab is required for PDF output. Install 'reportlab'.")

    if page_size.lower() == "letter":
        page_w, page_h = letter
    elif page_size.lower() == "a4":
        page_w, page_h = A4
    else:
        raise ValueError("page_size must be 'letter' or 'a4'")

    c = canvas.Canvas(str(out_pdf), pagesize=(page_w, page_h))
    x = margin_mm * mm
    y = page_h - margin_mm * mm
    spacing = spacing_mm * mm

    count_in_row = 0
    for tag_id, path in tag_paths:
        # Load image to get size in pixels, then scale 1:1 pixel-to-point (72dpi) is fine for layout
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h_px, w_px = img.shape[:2]
        # Convert to RGB and save to a temporary PNG that reportlab can use
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        tmp_path = str(path)  # reuse same file path
        cv2.imwrite(tmp_path, rgb)

        # Place image; reportlab expects width/height in points
        # Assume 300 DPI images: 1px = 72/300 points
        scale = 72.0 / 300.0
        w_pt = w_px * scale
        h_pt = h_px * scale

        if x + w_pt > page_w - margin_mm * mm:
            # new row
            x = margin_mm * mm
            y -= (h_pt + spacing)
            count_in_row = 0
            if y - h_pt < margin_mm * mm:
                c.showPage()
                y = page_h - margin_mm * mm

        c.drawImage(tmp_path, x, y - h_pt, width=w_pt, height=h_pt)
        c.drawString(x, y - h_pt - 10, f"Tag {tag_id}")

        x += w_pt + spacing
        count_in_row += 1
        if count_in_row >= tags_per_row:
            x = margin_mm * mm
            y -= (h_pt + spacing)
            count_in_row = 0
            if y - h_pt < margin_mm * mm:
                c.showPage()
                y = page_h - margin_mm * mm

    c.save()


def build_pdf_sheet_pillow(
    out_pdf: Path,
    tag_paths: List[Tuple[int, Path]],
    page_size: str = "letter",
    margin_mm: float = 10.0,
    spacing_mm: float = 10.0,
    tags_per_row: int = 4,
    dpi: int = 300,
):
    # Define page sizes in mm
    if page_size.lower() == "letter":
        page_w_mm, page_h_mm = 215.9, 279.4
    elif page_size.lower() == "a4":
        page_w_mm, page_h_mm = 210.0, 297.0
    else:
        raise ValueError("page_size must be 'letter' or 'a4'")

    def mm_to_px_local(mm_val: float) -> int:
        return int(round(mm_val * dpi / 25.4))

    page_w_px = mm_to_px_local(page_w_mm)
    page_h_px = mm_to_px_local(page_h_mm)
    margin_px = mm_to_px_local(margin_mm)
    spacing_px = mm_to_px_local(spacing_mm)

    pages: List[Image.Image] = []
    draw_objs: List[ImageDraw.ImageDraw] = []

    def new_page() -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (page_w_px, page_h_px), color=(255, 255, 255))
        drw = ImageDraw.Draw(img)
        pages.append(img)
        draw_objs.append(drw)
        return img, drw

    page, draw = new_page()
    x = margin_px
    y = margin_px
    font = ImageFont.load_default()

    count_in_row = 0
    for tag_id, path in tag_paths:
        try:
            tag_img = Image.open(str(path)).convert("RGB")
        except Exception:
            continue
        w_px, h_px = tag_img.size
        label = "Tag %s" % tag_id
        # Measure text robustly across Pillow versions
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            label_w, label_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                label_w, label_h = font.getsize(label)  # type: ignore[attr-defined]
            except Exception:
                label_w, label_h = (len(label) * 6, 12)

        # Wrap to next row if width exceeded
        if x + w_px > page_w_px - margin_px:
            x = margin_px
            y += h_px + spacing_px + label_h + 4
            count_in_row = 0

        # New page if height exceeded
        if y + h_px + label_h + 4 > page_h_px - margin_px:
            page, draw = new_page()
            x = margin_px
            y = margin_px
            count_in_row = 0

        page.paste(tag_img, (x, y))
        draw.text((x, y + h_px + 2), label, fill=(0, 0, 0), font=font)

        x += w_px + spacing_px
        count_in_row += 1
        if count_in_row >= tags_per_row:
            x = margin_px
            y += h_px + spacing_px + label_h + 4
            count_in_row = 0

    # Save as multi-page PDF
    if not pages:
        return
    first, rest = pages[0], pages[1:]
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    first.save(str(out_pdf), "PDF", resolution=dpi, save_all=True, append_images=rest)


def main():
    ap = argparse.ArgumentParser(description="Generate AprilTag images for printing.")
    ap.add_argument("--ids", help="Comma-separated IDs and/or ranges (e.g., '0-3,10,12')")
    ap.add_argument("--range", dest="range_ids", help="ID range (e.g., '0-19')")
    ap.add_argument("--family", default="APRILTAG_36h11", choices=list(FAMILY_MAP.keys()), help="Tag family")
    ap.add_argument("--size-mm", type=float, default=40.0, help="Tag size (inner black square) in mm")
    ap.add_argument("--dpi", type=int, default=300, help="Output DPI for PNGs")
    ap.add_argument("--output-dir", default="output/apriltags", help="Output directory for PNGs")
    ap.add_argument("--pdf", help="Optional PDF sheet output path")
    ap.add_argument("--page-size", default="letter", choices=["letter", "a4"], help="PDF page size")
    ap.add_argument("--margin-mm", type=float, default=10.0, help="PDF margin in mm")
    ap.add_argument("--spacing-mm", type=float, default=10.0, help="PDF spacing in mm")
    ap.add_argument("--tags-per-row", type=int, default=4, help="Tags per row in PDF")

    args = ap.parse_args()

    ids = parse_ids(args.ids or "", args.range_ids)
    size_px = mm_to_px(args.size_mm, args.dpi)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(ids)} AprilTag(s) in family {args.family} at {args.size_mm}mm, {args.dpi} DPI...")
    tag_paths: List[Tuple[int, Path]] = []
    for tid in ids:
        img = generate_tag_image(tid, size_px, args.family)
        # Add 1-module white border around for printing clarity
        border = int(round(size_px * 0.15))
        img_b = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
        out_path = out_dir / f"apriltag_{args.family.lower()}_{tid}.png"
        save_tag_png(img_b, out_path)
        tag_paths.append((tid, out_path))
    print(f"Saved PNGs to {out_dir}")

    if args.pdf:
        out_pdf = Path(args.pdf)
        print(f"Building PDF sheet: {out_pdf}")
        built = False
        if reportlab is not None:
            try:
                build_pdf_sheet(out_pdf, tag_paths, page_size=args.page_size, margin_mm=args.margin_mm,
                                spacing_mm=args.spacing_mm, tags_per_row=args.tags_per_row)
                print(f"Saved PDF: {out_pdf}")
                built = True
            except Exception as e:
                print(f"ReportLab PDF generation failed ({e}); trying Pillow-based PDF...")
        if not built:
            try:
                build_pdf_sheet_pillow(out_pdf, tag_paths, page_size=args.page_size, margin_mm=args.margin_mm,
                                       spacing_mm=args.spacing_mm, tags_per_row=args.tags_per_row, dpi=args.dpi)
                print(f"Saved PDF (Pillow): {out_pdf}")
                built = True
            except Exception as e:
                print(f"PDF generation failed ({e}); PNGs are available in {Path(args.output_dir)}")

    print("Done.")


if __name__ == "__main__":
    raise SystemExit(main())
