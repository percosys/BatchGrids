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
    
    # Processing
    max_image_size: int = Field(default=2048, description="Maximum image dimension for processing")
    apriltag_family: str = Field(default="tag36h11", description="AprilTag family")
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


settings = Settings()