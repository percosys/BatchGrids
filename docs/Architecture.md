
# BatchGrids Architecture (Stub)

## High-level components
- **Clients:** Mobile app (React Native), Web app (Next.js)
- **API:** FastAPI (auth, CRUD, orchestration)
- **Workers:** Celery + Redis (vision, exports)
- **ML:** YOLO-seg (Ultralytics, ONNX)
- **Data:** Postgres (with PostGIS), S3-compatible storage
- **Exports:** SVG/DXF (2D), STL/STEP/3MF (3D)
- **Infra:** Docker Compose (dev), Terraform + ECS/K8s (prod)

## Data Flow
1. User uploads tool image.
2. API stores image → queues Celery job.
3. Worker: detect AprilTags → rectify → YOLO-seg → outline → dims.
4. Save tool record, SVG outline, metadata.
5. User groups tools into bins → auto-pack.
6. On export request, worker generates chosen file format → S3.
7. API returns download link.

## Next steps
- Add diagrams (system context, sequence flows, DB schema).
- Detail security & auth model.
