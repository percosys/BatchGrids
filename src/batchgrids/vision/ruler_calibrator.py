import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from batchgrids.config import settings


@dataclass
class RulerCalibration:
    px_per_mm: float
    is_valid: bool
    orientation: str  # 'horizontal' or 'vertical'
    units: str  # 'mm' or 'cm' or 'inch-derived'
    quality_score: float
    error_message: Optional[str] = None
    roi_bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


class RulerCalibrator:
    """Estimate px/mm from an image containing a ruler.

    Approach: project image intensity along X and Y, detect periodic peaks via autocorrelation,
    prefer a period consistent with 10 mm (1 cm) tick spacing, else fall back to 1 mm or 1 inch.
    """

    def __init__(self):
        self.expected_units = settings.ruler_expected_units  # auto|cm|inch
        self.cm_peak_range = (settings.ruler_cm_peak_px_min, settings.ruler_cm_peak_px_max)
        self.mm_peak_range = (settings.ruler_mm_peak_px_min, settings.ruler_mm_peak_px_max)

    def calibrate(self, image: np.ndarray) -> RulerCalibration:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # Edge-enhance tick marks
            edges = cv2.Canny(gray, 50, 150)
            # Strengthen thin marks
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=1)

            # Projections
            proj_x = edges.sum(axis=0).astype(np.float32)  # along columns (horizontal axis)
            proj_y = edges.sum(axis=1).astype(np.float32)  # along rows (vertical axis)

            # Normalize, remove DC, and smooth safely in 1D
            def preprocess(sig: np.ndarray) -> np.ndarray:
                s = sig.astype(np.float32)
                mx = float(s.max()) if s.size else 0.0
                if mx > 0:
                    s = s / mx
                s = s - float(s.mean())
                # Gaussian 1D smoothing via numpy convolution
                ksize = 21  # odd length
                sigma = 4.0
                x = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
                g = np.exp(-(x * x) / (2.0 * sigma * sigma))
                g /= g.sum()
                s = np.convolve(s, g, mode='same')
                return s

            px = preprocess(proj_x)
            py = preprocess(proj_y)

            # Autocorrelation
            def autocorr(sig: np.ndarray) -> np.ndarray:
                n = len(sig)
                f = np.fft.rfft(sig, n=2*n)
                ac = np.fft.irfft(f * np.conj(f))[:n]
                ac = ac / (np.arange(n, 0, -1))
                return ac

            acx = autocorr(px)
            acy = autocorr(py)

            # Find dominant period in reasonable ranges
            def peak_in_range(ac: np.ndarray, r: Tuple[int, int]) -> Tuple[int, float]:
                start = max(2, r[0])
                end = min(len(ac) - 1, r[1])
                if end <= start:
                    return -1, 0.0
                idx = np.argmax(ac[start:end]) + start
                return int(idx), float(ac[idx])

            # Prefer 1 cm spacing if possible
            cm_period_x, score_cmx = peak_in_range(acx, self.cm_peak_range)
            cm_period_y, score_cmy = peak_in_range(acy, self.cm_peak_range)
            mm_period_x, score_mmx = peak_in_range(acx, self.mm_peak_range)
            mm_period_y, score_mmy = peak_in_range(acy, self.mm_peak_range)

            # Choose best orientation and unit
            candidates = []
            if cm_period_x > 0:
                candidates.append((score_cmx, 'horizontal', 'cm', cm_period_x))
            if cm_period_y > 0:
                candidates.append((score_cmy, 'vertical', 'cm', cm_period_y))
            if mm_period_x > 0:
                candidates.append((score_mmx, 'horizontal', 'mm', mm_period_x))
            if mm_period_y > 0:
                candidates.append((score_mmy, 'vertical', 'mm', mm_period_y))

            if not candidates:
                return RulerCalibration(0.0, False, 'horizontal', 'mm', 0.0, 'No periodic ticks detected')

            # Respect expected_units preference
            if self.expected_units in ('cm', 'inch'):
                preferred = [c for c in candidates if c[2] == ('cm' if self.expected_units == 'cm' else 'inch')]
                if preferred:
                    candidates = preferred

            # Pick highest score
            candidates.sort(key=lambda t: t[0], reverse=True)
            top = candidates[0]
            score, orientation, unit, period_px = top

            if unit == 'cm':
                px_per_mm = float(period_px) / 10.0
            elif unit == 'mm':
                px_per_mm = float(period_px)
            else:
                # inch case if implemented; convert to mm
                px_per_mm = float(period_px) / 25.4
                unit = 'inch-derived'

            # Sanity check reasonable range
            if not (0.5 <= px_per_mm <= 100.0):
                return RulerCalibration(0.0, False, orientation, unit, score, 'Unreasonable px/mm estimate')

            return RulerCalibration(px_per_mm=px_per_mm, is_valid=True, orientation=orientation,
                                    units=unit, quality_score=score)

        except Exception as e:
            return RulerCalibration(0.0, False, 'horizontal', 'mm', 0.0, str(e))

    # --- ROI-based refinement ---
    def calibrate_best_roi(self, image: np.ndarray) -> RulerCalibration:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
            h, w = gray.shape[:2]
            # Emphasize vertical lines (ticks for horizontal ruler) and horizontal lines
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            ax = np.abs(sobelx)
            ay = np.abs(sobely)

            # Row/column strengths
            row_strength_x = ax.mean(axis=1)
            col_strength_y = ay.mean(axis=0)

            # Smooth
            # Smooth with 1D Gaussian
            def smooth1d(v: np.ndarray) -> np.ndarray:
                ksize = 31
                sigma = 6.0
                xg = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
                g = np.exp(-(xg * xg) / (2.0 * sigma * sigma))
                g /= g.sum()
                return np.convolve(v.astype(np.float32), g, mode='same')

            row_strength_x = smooth1d(row_strength_x)
            col_strength_y = smooth1d(col_strength_y)

            # Window size ~ 20% of dimension
            win_h = max(30, int(0.2 * h))
            win_w = max(30, int(0.2 * w))

            # Best row window for vertical lines (horizontal ruler)
            best_y = 0
            best_val_y = -1
            csum = np.cumsum(row_strength_x)
            for y in range(0, h - win_h):
                val = csum[y + win_h] - csum[y]
                if val > best_val_y:
                    best_val_y = val
                    best_y = y

            # Best column window for horizontal lines (vertical ruler)
            best_x = 0
            best_val_x = -1
            csumx = np.cumsum(col_strength_y)
            for x in range(0, w - win_w):
                val = csumx[x + win_w] - csumx[x]
                if val > best_val_x:
                    best_val_x = val
                    best_x = x

            # Two candidate ROIs: horizontal band (y band) and vertical band (x band)
            candidates = []
            # Horizontal band ROI (assume ticks vertical): full width, band in y
            y0 = max(0, best_y - win_h // 4)
            y1 = min(h, best_y + win_h + win_h // 4)
            roi_h = image[y0:y1, :]
            cal_h = self.calibrate(roi_h)
            if cal_h.is_valid:
                cal_h.roi_bbox = (0, y0, w, y1 - y0)
                candidates.append(cal_h)

            # Vertical band ROI (assume ticks horizontal): full height, band in x
            x0 = max(0, best_x - win_w // 4)
            x1 = min(w, best_x + win_w + win_w // 4)
            roi_v = image[:, x0:x1]
            cal_v = self.calibrate(roi_v)
            if cal_v.is_valid:
                cal_v.roi_bbox = (x0, 0, x1 - x0, h)
                candidates.append(cal_v)

            if not candidates:
                return self.calibrate(image)

            candidates.sort(key=lambda c: c.quality_score, reverse=True)
            return candidates[0]
        except Exception as e:
            # Fallback to global calibration
            rc = self.calibrate(image)
            rc.error_message = (rc.error_message or "") + f" | ROI refine error: {e}"
            return rc

    # --- Hough-based oriented calibration ---
    def calibrate_hough(self, image: np.ndarray) -> RulerCalibration:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
            edges = cv2.Canny(gray, 80, 180)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=min(gray.shape[:2]) // 4, maxLineGap=10)
            if lines is None or len(lines) == 0:
                return self.calibrate_best_roi(image)

            best: Optional[RulerCalibration] = None
            for l in lines[:50]:
                x1, y1, x2, y2 = l[0]
                dx, dy = x2 - x1, y2 - y1
                length = np.hypot(dx, dy)
                if length < min(gray.shape[:2]) * 0.3:
                    continue
                angle = np.degrees(np.arctan2(dy, dx))
                # Rotate image so this line is horizontal
                M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
                rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
                # Transform the line coordinates
                pts = np.array([[x1, y1, 1], [x2, y2, 1]], dtype=np.float32).T
                rot_pts = (M @ pts).T.astype(int)
                rx1, ry1 = rot_pts[0, 0], rot_pts[0, 1]
                rx2, ry2 = rot_pts[1, 0], rot_pts[1, 1]
                cy = max(0, min(rot.shape[0] - 1, (ry1 + ry2) // 2))
                band_h = max(10, int(rot.shape[0] * 0.05))
                y0 = max(0, cy - band_h // 2)
                y1b = min(rot.shape[0], cy + band_h // 2)
                band = rot[y0:y1b, :]
                # 1D signal along x
                sig = band.sum(axis=0).astype(np.float32)

                # Preprocess and autocorrelate
                s = sig.copy()
                mx = float(s.max()) if s.size else 0.0
                if mx > 0:
                    s /= mx
                s = s - float(s.mean())
                ksize = 31
                sigma = 6.0
                xg = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
                g = np.exp(-(xg * xg) / (2.0 * sigma * sigma))
                g /= g.sum()
                s = np.convolve(s, g, mode='same')
                # autocorr
                n = len(s)
                f = np.fft.rfft(s, n=2*n)
                ac = np.fft.irfft(f * np.conj(f))[:n]
                ac = ac / (np.arange(n, 0, -1))

                # Search for 1 cm then 1 mm periods
                cm_period, cm_score = self._best_peak_in_range(ac, self.cm_peak_range)
                mm_period, mm_score = self._best_peak_in_range(ac, self.mm_peak_range)
                pick_unit = None
                pick_period = None
                pick_score = 0.0
                if cm_period > 0 and cm_score >= mm_score:
                    pick_unit = 'cm'
                    pick_period = cm_period
                    pick_score = cm_score
                elif mm_period > 0:
                    pick_unit = 'mm'
                    pick_period = mm_period
                    pick_score = mm_score
                if pick_period is None:
                    continue

                px_per_mm = (pick_period / 10.0) if pick_unit == 'cm' else float(pick_period)
                if not (0.5 <= px_per_mm <= 100.0):
                    continue
                rc = RulerCalibration(px_per_mm=px_per_mm, is_valid=True, orientation='horizontal', units=pick_unit,
                                      quality_score=float(pick_score), roi_bbox=(0, y0, rot.shape[1], y1b - y0))
                if best is None or rc.quality_score > best.quality_score:
                    best = rc

            if best:
                return best
            return self.calibrate_best_roi(image)
        except Exception as e:
            rc = self.calibrate_best_roi(image)
            rc.error_message = (rc.error_message or "") + f" | Hough error: {e}"
            return rc

    def _best_peak_in_range(self, ac: np.ndarray, r: Tuple[int, int]) -> Tuple[int, float]:
        start = max(2, r[0])
        end = min(len(ac) - 1, r[1])
        if end <= start:
            return -1, 0.0
        idx = int(np.argmax(ac[start:end]) + start)
        return idx, float(ac[idx])
