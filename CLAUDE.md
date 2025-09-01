# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**BatchGrids** is a tool management application that digitizes physical tools through computer vision and organizes them into custom storage solutions. The project is currently in the planning/design phase with comprehensive documentation but no implementation yet.

## Architecture

Based on `docs/Architecture.md`, the system follows a multi-tier architecture:

- **Clients:** Mobile app (React Native), Web app (Next.js)
- **API:** FastAPI backend (authentication, CRUD, orchestration)
- **Workers:** Celery + Redis for background processing (computer vision, exports)
- **ML Pipeline:** YOLO segmentation with ONNX runtime for tool detection
- **Data:** PostgreSQL with PostGIS for spatial data, S3-compatible object storage
- **Exports:** Multiple format support (SVG/DXF for 2D, STL/STEP/3MF for 3D)
- **Infrastructure:** Docker Compose for development, Terraform + ECS/K8s for production

## Core Workflow

1. User captures tool images on AprilTag-calibrated scan mat
2. API stores images and queues Celery processing jobs
3. Worker pipeline: AprilTag detection → image rectification → YOLO segmentation → outline extraction → dimension calculation
4. Tools are stored with metadata, SVG outlines, and classifications
5. Users group tools into bins with auto-packing algorithms
6. On-demand export generation for various fabrication formats
7. Inventory lookup system maps tools to physical storage locations

## Key Components

### Computer Vision Pipeline
- AprilTag calibration for perspective correction and scale reference
- YOLO segmentation for tool outline detection and classification
- Homography transformation for accurate measurements
- Confidence-based review workflows for low-accuracy predictions

### Data Model
Key entities include: users, classes, tools, images, predictions, drawers, grid_bins, assignments, exports. Full schema documented in `docs/BRD_Tooltracepp.md`.

### Storage and Packing
- Gridfinity-compatible bin generation
- Auto-packing algorithms (starting with row packing, upgrading to MaxRects)
- Custom storage layout support

### Export System
- On-demand file generation (not pre-computed)
- Multiple format support for different fabrication workflows
- Versioned exports with metadata tracking

## Development Notes

This project is currently in the documentation and planning phase. No implementation exists yet, so development commands, test frameworks, and build systems have not been established. When implementation begins, refer to the architecture documents for technology stack decisions and the BRD for functional requirements.