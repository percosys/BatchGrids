
# User Stories & Validation Tests — BatchGrids

## Epics
1. Capture & Calibration
2. Identification & Review
3. Tool Library (CRUD & Search)
4. Grouping & Gridfinity Packing
5. Inventory & Location Lookup
6. Exports (On-Demand)
7. Labels & Indexes (PDF)
8. Admin & Taxonomy
9. Reliability & Performance

---

## 1. Capture & Calibration
**US1.1** Upload tool photo with AprilTags  
- As a user, I want to upload a photo on the scan mat so the system can scale correctly.  
- **AC:** With IDs 0,1,2,3 visible, `/images/process` returns `px_per_mm`, rectified image.

**US1.2** Block bad photos  
- **AC:** Missing ≥2 tags → 400 error.

**US1.3** Rectified preview  
- **AC:** Preview shows corrected geometry; “Scale Verified” badge if ruler checks pass.

---

## 2. Identification & Review
**US2.1** YOLO segmentation & classification  
- **AC:** Outline mask stored, class predicted with confidence.

**US2.2** Low-confidence review  
- **AC:** If `p_top1 < 0.8`, system prompts user to confirm/correct.

**US2.3** Manual outline adjustment  
- **AC:** User can tweak outline; new version saved, history preserved.

---

## 3. Tool Library
**US3.1** Create tool record  
- **AC:** Auto-create tool record after processing with image, dims, outline.

**US3.2** Search by name/class/dimensions  
- **AC:** Queries return in <1s for 10k records.

**US3.3** Edit metadata  
- **AC:** PATCH `/tools/{id}` updates name, notes.

---

## 4. Grouping & Packing
**US4.1** Create a bin (tile)  
- **AC:** User defines bin (x,y,z units); system computes usable mm.

**US4.2** Auto-pack tools into bin  
- **AC:** Returns placements; overflow tools spilled into new bin.

**US4.3** Manual placement override  
- **AC:** Dragging tools updates assignments; overlaps prevented.

---

## 5. Inventory & Lookup
**US5.1** Define drawers  
- **AC:** POST `/drawers` creates drawers with unique codes.

**US5.2** Assign bin to drawer/tile code  
- **AC:** PATCH `/bins/{id}` with `{drawer_id, label_code}` works; uniqueness enforced.

**US5.3** Tool lookup  
- **AC:** Query returns drawer/tile location in <1s.

---

## 6. Exports
**US6.1** Export foam (SVG/DXF)  
- **AC:** Correct mm scale; path loads in Inkscape/laser software.

**US6.2** Export bins (STL/STEP/3MF)  
- **AC:** STL printable, STEP editable, 3MF slicer-ready.

**US6.3** Version history  
- **AC:** Past exports listed with timestamps.

---

## 7. Labels & Indexes
**US7.1** Drawer label (PDF)  
- **AC:** Printable with drawer name/code.

**US7.2** Tile labels (PDF)  
- **AC:** Page with tile codes T1..Tn.

**US7.3** Drawer index sheet  
- **AC:** Lists tools under each tile.

---

## 8. Admin & Taxonomy
**US8.1** Manage class list  
- **AC:** Admin adds/edits classes; synonyms searchable.

**US8.2** Active learning export  
- **AC:** `/trainer/export-dataset` returns COCO dataset for low-confidence images.

---

## 9. Reliability & Performance
**US9.1** Processing speed  
- **AC:** 95th percentile image process <5s.

**US9.2** Export speed  
- **AC:** 2D exports <2s; 3D <30s.

**US9.3** Dimensional accuracy  
- **AC:** L/W within ±1 mm vs caliper.

**US9.4** Search latency  
- **AC:** <1s for 10k tools.

---

## Validation Tests

### A. Capture
- Valid photo returns px_per_mm.
- Missing tags → 400 error.
- Scale check square matches ruler.

### B. Identification
- YOLO detects known tools correctly.
- Low confidence → review prompt.

### C. Library
- Record auto-created.
- Search by class/dim works.

### D. Packing
- Tools auto-packed without overlap.
- Overflow creates new bin.

### E. Inventory
- Drawers/tiles created with unique codes.
- Lookup returns correct location.

### F. Exports
- SVG/DXF print to scale.
- STL imports in slicer.
- STEP opens in Fusion/FreeCAD.
- 3MF opens in PrusaSlicer/Cura.

### G. Labels/Indexes
- Drawer label PDF prints.
- Tile label sheet correct.
- Index PDF legible.

### H. NFR Tests
- Search latency <1s with 10k tools.
- Dimensional error <±1 mm vs caliper.
