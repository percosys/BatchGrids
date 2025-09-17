from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

    # Database
    database_url: str = Field(
        default="postgresql://batchgrids:batchgrids_dev@localhost:5432/batchgrids",
        description="PostgreSQL database URL"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for Celery"
    )
    
    # S3 Storage
    s3_endpoint_url: str = Field(
        default="http://localhost:9000",
        description="S3-compatible storage endpoint"
    )
    s3_access_key: str = Field(default="minioadmin", description="S3 access key")
    s3_secret_key: str = Field(default="minioadmin123", description="S3 secret key")
    s3_bucket_name: str = Field(default="batchgrids", description="S3 bucket name")
    
    # API
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, description="JWT token expiration")
    
    # Environment
    environment: str = Field(default="development", description="Environment (development/production)")
    debug: bool = Field(default=True, description="Debug mode")
    
    # ML Models
    yolo_model_path: str = Field(
        default="yolov8n-seg.pt",
        description="Path to YOLO segmentation model"
    )
    yolo_confidence_threshold: float = Field(
        default=0.8,
        description="YOLO confidence threshold for auto-accept"
    )
    yolo_inference_confidence: float = Field(
        default=0.3,
        description="YOLO inference confidence - lower for better tool detection"
    )
    # Segmentation/mask tuning
    yolo_mask_threshold: float = Field(
        default=0.35,
        description="Probability threshold to binarize YOLO masks"
    )
    yolo_min_area_fraction: float = Field(
        default=0.0005,
        description="Discard masks smaller than this fraction of the image area"
    )
    yolo_morph_kernel_px: int = Field(
        default=5,
        description="Kernel size (px) for morphological open/close on masks"
    )
    yolo_morph_iterations: int = Field(
        default=1,
        description="Iterations for morphological operations"
    )
    outline_simplify_epsilon_ratio: float = Field(
        default=0.006,
        description="RDP simplify epsilon as a ratio of contour perimeter (higher=smoother)"
    )
    outline_smooth_buffer_px: int = Field(
        default=6,
        description="Base pixel radius for buffer-in/buffer-out smoothing (dynamic with perimeter)"
    )
    yolo_refine_shrink_px: int = Field(
        default=0,
        description="Extra erosion in pixels to tighten the final mask (0=off)"
    )
    svg_use_smooth_curves: bool = Field(
        default=True,
        description="Generate smooth cubic Bezier path instead of straight segments"
    )
    outline_chaikin_iterations: int = Field(
        default=1,
        description="Extra curve smoothing via Chaikin iterations before Bezier conversion"
    )
    outline_max_points: int = Field(
        default=400,
        description="Cap on outline points prior to Bezier conversion to keep SVG manageable"
    )
    outline_smooth_buffer_max_px: int = Field(
        default=10,
        description="Upper cap in pixels for corner rounding radius"
    )
    
    # Processing
    max_image_size: int = Field(default=2048, description="Maximum image dimension for processing")
    apriltag_family: str = Field(default="tag36h11", description="AprilTag family")

    # Ruler-based calibration (fallback)
    use_ruler_fallback: bool = Field(default=False, description="Try to estimate scale from a ruler if AprilTags fail")
    ruler_expected_units: str = Field(default="auto", description="auto|cm|inch for ruler calibration")
    ruler_cm_peak_px_min: int = Field(default=60, description="Min expected pixels for 1 cm tick spacing")
    ruler_cm_peak_px_max: int = Field(default=400, description="Max expected pixels for 1 cm tick spacing")
    ruler_mm_peak_px_min: int = Field(default=6, description="Min expected pixels for 1 mm tick spacing")
    ruler_mm_peak_px_max: int = Field(default=60, description="Max expected pixels for 1 mm tick spacing")
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


settings = Settings()
