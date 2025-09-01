
# Business Requirements Document (BRD) — BatchGrids

## 1. Purpose
BatchGrids is a tool management and organization application that lets users:
1. Capture images of individual tools on a calibrated scan mat.
2. Automatically identify tool type and outline using AprilTag calibration + YOLO segmentation.
3. Store tool metadata, geometry, and classifications in a persistent database.
4. Group tools and auto-arrange them into storage layouts (e.g., Gridfinity bins, foam shadow boards).
5. Generate **on-demand exports** for fabrication (SVG, DXF, STL, STEP, 3MF).
6. Maintain an **inventory lookup service** so users can locate tools by drawer/tile.
7. Print **text-based labels and indexes** for drawers and tiles.

---

## 2. Goals & Objectives
- **Digitization:** Create a digital library of tools with accurate dimensions and metadata.
- **Automation:** Reduce manual input via AI segmentation and classification.
- **Customization:** Enable users to create custom storage solutions automatically.
- **Inventory:** Provide quick lookup (tool → drawer/tile location).
- **Fabrication:** Support multiple export formats for CNC, laser, 3D printing, and CAD workflows.
- **Simplicity:** Keep onboarding and usage simple (print mat, snap photo, confirm label, export).

---

## 3. Scope

### In Scope (MVP + Extensions)
- AprilTag-based scan mat calibration (IDs 0–3 in corners).
- YOLO segmentation for outline + classification with confidence scores.
- Tool library database with search by class, name, dimensions.
- Grouping and Gridfinity packing (row/column → MaxRects upgrade).
- Inventory management (drawers, bins/tiles).
- Labels & Index generation (PDF text).
- On-demand export (SVG, DXF, STL, STEP, 3MF).

### Out of Scope (Future)
- Multi-user collaboration & sharing.
- Check-in/out workflows.
- IoT tagging (RFID/BLE).
- Barcode/QR scanning.

---

## 4. Stakeholders
- **Primary Users:** Makers, hobbyists, small shops.
- **Business Stakeholders:** Product owner, partners (laser/3D printing shops).
- **Future Customers:** Enterprises with large tool inventories.

---

## 5. Functional Requirements

### Tool Capture & Calibration
- Users capture one tool per photo with AprilTags visible.
- System validates presence of all required tags.
- Homography rectification converts trapezoid → true rectangle.
- px_per_mm scale computed.

### Tool Identification
- YOLO segmentation produces outline mask + class prediction.
- System stores geometric measurements (L, W, optional T).
- Predictions with confidence <0.8 require user review.

### Tool Database
- Each tool stores: images, masks, SVG outline, dims, class, notes, and location.
- Search by class, text, or dimension filters.
- Edit metadata (rename, notes).

### Grouping & Packing
- Group tools into categories (e.g., screwdrivers, sockets).
- Auto-pack into bins based on available area.
- Handle overflow with new bins.
- Allow manual placement override.

### Inventory & Lookup
- Users define drawers with unique codes (e.g., D1).
- Assign bins/tiles to drawers (e.g., T1..Tn).
- Each tool mapped to a bin/tile automatically or manually.
- Search returns drawer/tile location in <1s.

### Labels & Indexes
- Generate drawer label PDFs (plain text).
- Generate tile labels (codes T1..Tn).
- Generate drawer index sheets:
  ```
  Drawer D1
  T1: Hammer, Pliers
  T2: Screwdrivers
  T3: 10mm Socket, 12mm Socket
  ```
- Export full inventory index as CSV.

### Exporting (On-Demand)
- Supported formats:
  - Foam inserts: SVG, DXF.
  - Gridfinity bins: STL, STEP, 3MF.
- User requests specific format(s).
- Export job runs asynchronously; file stored in object storage.
- Export records stored with metadata, tool snapshot, versioning.

---

## 6. Non-Functional Requirements
- **Accuracy:** ≤ ±1 mm error vs caliper truth.
- **Performance:** Process image <5s; SVG/DXF export <2s; 3D export <30s.
- **Search speed:** <1s for ≥10k tools.
- **Storage:** Exports only generated on demand; old exports prunable.
- **Compatibility:** STEP imports into Fusion/FreeCAD; 3MF loads in slicers.

---

## 7. Data Model

### Tables
- **users**: id, email, name, created_at
- **classes**: id, name, synonyms[]
- **tools**: id, user_id, class_id, canonical_name, dims_mm, notes, shape_polygon, created_at
- **images**: id, tool_id, user_id, uri, px_per_mm, homography, bbox, mask_uri, status, created_at
- **predictions**: id, image_id, model, top1, p_top1, topk, reviewed, created_at
- **drawers**: id, user_id, name, label_code, created_at
- **grid_bins**: id, user_id, drawer_id, name, x_units, y_units, z_units, inner_clearance_mm, wall_mm, label_code
- **assignments**: bin_id, tool_id, x, y, w, h, rotation
- **exports**: id, user_id, bin_id, format, uri, version, created_at

---

## 8. Risks & Mitigations
- **Bad photos:** enforce tag validation + user guidance overlay.
- **Misclassification:** review flow + active learning loop.
- **Packing complexity:** start with row packing; upgrade to MaxRects.
- **Label/index drift:** regenerate often with timestamps.

---

## 9. Success Criteria
- User can digitize ≥20 tools in a session.
- ≥80% classification accuracy.
- Exports usable in slicers and CAD (correct scale).
- Search returns correct location <1s.
- Labels/indexes printable and legible.
