# BatchGrids

BatchGrids is a tool management and organization application that digitizes physical tools through computer vision and organizes them into custom storage solutions.

## Features

- **Computer Vision Pipeline**: AprilTag calibration + YOLO segmentation for automatic tool detection
- **Digital Tool Library**: Store tool metadata, dimensions, and classifications
- **Smart Organization**: Auto-pack tools into Gridfinity bins and custom storage layouts
- **Multi-Format Exports**: Generate files for CNC, laser cutting, 3D printing (SVG, DXF, STL, STEP, 3MF)
- **Inventory Management**: Quick lookup system to locate tools by drawer/tile
- **Label Generation**: Printable labels and inventory indexes

## Technology Stack

- **Backend**: FastAPI with PostgreSQL + PostGIS
- **Background Processing**: Celery + Redis
- **Computer Vision**: OpenCV, Ultralytics YOLO, ONNX Runtime
- **Storage**: S3-compatible object storage (MinIO)
- **Infrastructure**: Docker Compose for development

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/percosys/BatchGrids.git
   cd BatchGrids
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration if needed
   ```

3. **Start services with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

4. **Install dependencies (for local development):**
   ```bash
   pip install -e ".[dev]"
   ```

5. **Run database migrations:**
   ```bash
   alembic upgrade head
   ```

6. **Access the application:**
   - API Documentation: http://localhost:8000/docs
   - MinIO Console: http://localhost:9001 (minioadmin / minioadmin123)

### Development Commands

```bash
# Start all services
docker-compose up

# Run API only (requires external DB/Redis)
uvicorn batchgrids.api:app --reload

# Run Celery worker
celery -A batchgrids.worker worker --loglevel=info

# Database migrations
alembic revision --autogenerate -m "Description"
alembic upgrade head

# Code formatting and linting
black .
ruff --fix .
mypy src/

# Run tests
pytest
```

## Architecture

### Core Components

1. **API Server** (`src/batchgrids/api/`): FastAPI application with authentication and CRUD endpoints
2. **Database Models** (`src/batchgrids/models/`): SQLAlchemy models for tools, images, bins, etc.
3. **Worker Tasks** (`src/batchgrids/worker/`): Celery tasks for image processing and exports
4. **Configuration** (`src/batchgrids/config.py`): Environment-based configuration management

### Data Flow

1. User uploads tool image → API stores in S3 and queues processing job
2. Worker detects AprilTags → rectifies image → runs YOLO segmentation
3. Tool record created with dimensions, outline, and classification
4. User groups tools into bins → auto-packing algorithm assigns positions
5. Export requests generate files asynchronously → stored in S3 with download links

### Database Schema

- **users**: User accounts and authentication
- **classes**: Tool classifications with synonyms
- **tools**: Tool records with dimensions and geometry
- **images**: Original photos and processing metadata
- **predictions**: YOLO classification results
- **drawers**: Physical storage containers
- **grid_bins**: Gridfinity bins with dimensions
- **assignments**: Tool placements within bins
- **exports**: Generated file records with download URLs

## API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/token` - Login and get access token
- `GET /auth/me` - Get current user info

### Images & Processing
- `POST /images/upload` - Upload tool image
- `POST /images/{id}/process` - Trigger image processing
- `GET /images/{id}` - Get image processing results

### Tool Management
- `GET /tools` - List tools with search/filtering
- `GET /tools/{id}` - Get tool details
- `PATCH /tools/{id}` - Update tool metadata

### Organization
- `POST /bins` - Create new Gridfinity bin
- `POST /bins/{id}/autofill` - Auto-pack tools into bin
- `POST /drawers` - Create storage drawer
- `GET /lookup?query=...` - Look up tool locations

### Exports
- `POST /bins/{id}/export` - Generate export file
- `GET /bins/{id}/exports` - Get export history

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run linting and type checking
5. Submit a pull request

## Documentation

- [Business Requirements Document](docs/BRD_BatchGrids.md)
- [User Stories & Validation Tests](docs/UserStories_BatchGrids.md)
- [System Architecture](docs/Architecture.md)
- [API Specification](docs/API_Spec.md)

## License

MIT License - see LICENSE file for details.