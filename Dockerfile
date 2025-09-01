FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgl1 \
    libglu1-mesa \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY src/ ./src/

# Install essential Python dependencies for computer vision
RUN pip install --no-cache-dir \
    opencv-python==4.8.1.78 \
    numpy==1.24.4 \
    ultralytics==8.0.200 \
    pillow==10.1.0

# Install AprilTag (may need to be built)
RUN pip install --no-cache-dir apriltag || echo "AprilTag install failed, will use fallback"

# Create output directory
RUN mkdir -p /app/output

# Default command
CMD ["python", "-c", "print('BatchGrids CV Container Ready')"]