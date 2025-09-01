# BatchGrids Computer Vision Pipeline

## Overview

The BatchGrids computer vision pipeline automatically processes tool images to extract precise dimensions and create digital tool records for organization. The system uses AprilTag calibration for accurate measurements and YOLO segmentation for tool detection.

## Pipeline Components

### 1. AprilTag Calibration (`apriltag_detector.py`)
- **Purpose**: Provides accurate scale and perspective correction
- **Process**: 
  - Detects AprilTags with IDs 0-3 at image corners
  - Computes homography transformation matrix
  - Calculates pixels-per-millimeter scale factor
- **Requirements**: Tags must be exactly 200mm apart
- **Output**: Rectified image with known scale

### 2. YOLO Segmentation (`yolo_segmenter.py`) 
- **Purpose**: AI-powered tool detection and classification
- **Process**:
  - Loads YOLOv8 segmentation model
  - Detects tools with confidence scores
  - Generates precise binary masks
- **Quality Control**: Flags low confidence (<0.8) for human review
- **Output**: Tool class, confidence, bounding box, segmentation mask

### 3. Dimension Extraction
- **Purpose**: Convert pixel measurements to real-world dimensions
- **Process**:
  - Extracts tool outline from segmentation mask
  - Calculates length, width, area in millimeters
  - Simplifies polygon for efficient storage
- **Accuracy**: ±1mm precision target
- **Output**: Dimensional data and simplified polygon

### 4. SVG Generation
- **Purpose**: Create vector outlines for fabrication
- **Process**:
  - Converts pixel coordinates to millimeter coordinates
  - Generates SVG with precise path data
  - Includes metadata and scale information
- **Compatible**: Laser cutters, CNC machines, CAM software
- **Output**: Scalable vector graphics file

## Workflow

```
📸 Image Upload
     ↓
🏷️ AprilTag Detection & Calibration
     ↓  
📐 Perspective Rectification
     ↓
🤖 YOLO Segmentation
     ↓
📏 Dimension Extraction
     ↓
💾 Database Storage & SVG Generation
```

## API Integration

The computer vision pipeline is integrated with the FastAPI backend:

- `POST /images/upload` - Upload tool images
- `POST /images/{id}/process` - Trigger processing  
- `GET /images/{id}` - Get processing results
- `GET /images/{id}/prediction` - Get YOLO predictions
- `GET /images/{id}/download` - Download processed images

## Quality Control

The system automatically flags images that need human review:

- **Low Confidence**: YOLO confidence < 0.8
- **Multiple Detections**: More than one tool detected
- **Calibration Failure**: Missing or invalid AprilTags
- **Processing Errors**: Technical failures

## File Outputs

### Tool Record (Database)
```sql
tools: {
  id: 42,
  canonical_name: "screwdriver",
  dims_mm: {"length": 185.3, "width": 12.7, "area": 1247.5},
  shape_polygon: {"type": "Polygon", "coordinates": [[...]]}
}
```

### SVG Outline (Fabrication)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     viewBox="0 0 185.3 12.7" width="185.3mm" height="12.7mm">
  <path d="M 2.1 6.3 L 178.9 6.3 ..." stroke="black" stroke-width="0.1mm"/>
</svg>
```

### Processing Metadata
```json
{
  "calibration": {"px_per_mm": 4.2, "tags_detected": [0,1,2,3]},
  "segmentation": {"confidence": 0.94, "class": "screwdriver"},
  "processing": {"time_ms": 3200, "needs_review": false}
}
```

## Testing

### CLI Tools

Generate calibration mat:
```bash
batchgrids-generate-mat --output calibration_mat.png --size 300 --dpi 300
```

Test vision pipeline:
```bash
batchgrids-test-vision image.jpg --output ./results --show --save-steps
```

### Docker Testing

Run pipeline in container:
```bash
docker-compose run api python src/batchgrids/cli/test_vision.py test_image.jpg
```

## Performance Targets

- **Processing Speed**: <5 seconds per image
- **Accuracy**: ±1mm dimensional precision
- **Reliability**: >95% successful processing rate
- **Throughput**: 20+ tools per session

## Calibration Mat Requirements

For accurate measurements, print the calibration mat with:
- **Exact size**: No scaling, print at actual size
- **High quality**: Matte paper, sharp black/white contrast
- **Proper tags**: AprilTags 0-3 at corners, 200mm spacing
- **Verification**: Measure tag spacing with ruler

## Production Deployment

The computer vision pipeline runs as Celery background tasks:
- Async processing prevents API blocking
- Status tracking through database
- Error handling and retry logic
- Debug visualization for troubleshooting
- S3 storage for images and results

## Future Enhancements

- Custom YOLO model training for specific tool types
- Multi-tool detection and segmentation
- 3D dimension estimation from single images
- Active learning for improving classification accuracy
- Batch processing for multiple tools per image