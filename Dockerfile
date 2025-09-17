FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy sources and metadata
COPY src/ ./src/
COPY web/ ./web/
COPY pyproject.toml ./
COPY README.md ./

ENV PIP_NO_CACHE_DIR=1
# Base runtime deps - no AprilTag dependency for simplified deployment
RUN pip install \
    numpy==1.24.4 \
    opencv-python-headless==4.8.1.78 \
    pillow==10.1.0 \
    ultralytics>=8.3.190 \
    pillow-heif>=0.13.0 \
    uvicorn[standard]>=0.24.0

# Install project (brings FastAPI, Pydantic, etc.)
RUN pip install .

# App env
ENV PYTHONPATH=/app/src
ENV DATABASE_URL="sqlite:///./batchgrids.db"
ENV SKIP_EXTERNAL_SERVICES=true
ENV IS_DEVELOPMENT=false
EXPOSE 8000

# Create output dir
RUN mkdir -p /app/output

# Start FastAPI app
CMD ["uvicorn", "batchgrids.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
