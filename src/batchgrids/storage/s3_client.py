import boto3
import cv2
import numpy as np
from typing import Optional, Dict, Any
from io import BytesIO
from pathlib import Path
import uuid
import logging
from datetime import datetime, timedelta

from batchgrids.config import settings

logger = logging.getLogger(__name__)


class S3Client:
    """S3-compatible storage client for images and files."""
    
    def __init__(self):
        self.endpoint_url = settings.s3_endpoint_url
        self.access_key = settings.s3_access_key
        self.secret_key = settings.s3_secret_key
        self.bucket_name = settings.s3_bucket_name
        
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='us-east-1'  # MinIO default
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except self.client.exceptions.NoSuchBucket:
            logger.info(f"Creating bucket: {self.bucket_name}")
            self.client.create_bucket(Bucket=self.bucket_name)
        except Exception as e:
            logger.warning(f"Could not check/create bucket: {e}")
    
    def upload_image(self, image: np.ndarray, key: str, 
                    content_type: str = "image/jpeg") -> str:
        """Upload OpenCV image to S3."""
        # Encode image
        if content_type == "image/jpeg":
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif content_type == "image/png":
            _, buffer = cv2.imencode('.png', image)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Upload to S3
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=buffer.tobytes(),
            ContentType=content_type
        )
        
        return self._get_object_url(key)
    
    def upload_file(self, file_path: str, key: str, 
                   content_type: Optional[str] = None) -> str:
        """Upload file to S3."""
        if content_type is None:
            # Guess content type from extension
            suffix = Path(file_path).suffix.lower()
            content_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.svg': 'image/svg+xml',
                '.dxf': 'application/dxf',
                '.stl': 'application/sla',
                '.step': 'application/step',
                '.3mf': 'application/vnd.ms-package.3dmanufacturing-3dmodel+xml',
                '.pdf': 'application/pdf'
            }
            content_type = content_type_map.get(suffix, 'application/octet-stream')
        
        with open(file_path, 'rb') as f:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=f.read(),
                ContentType=content_type
            )
        
        return self._get_object_url(key)
    
    def upload_bytes(self, data: bytes, key: str, 
                    content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to S3."""
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type
        )
        
        return self._get_object_url(key)
    
    def upload_string(self, content: str, key: str, 
                     content_type: str = "text/plain") -> str:
        """Upload string content to S3."""
        return self.upload_bytes(content.encode('utf-8'), key, content_type)
    
    def download_image(self, key: str) -> np.ndarray:
        """Download image from S3 as OpenCV image."""
        response = self.client.get_object(Bucket=self.bucket_name, Key=key)
        image_data = response['Body'].read()
        
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Could not decode image from S3: {key}")
        
        return image
    
    def download_bytes(self, key: str) -> bytes:
        """Download file as bytes from S3."""
        response = self.client.get_object(Bucket=self.bucket_name, Key=key)
        return response['Body'].read()
    
    def download_string(self, key: str) -> str:
        """Download file as string from S3."""
        return self.download_bytes(key).decode('utf-8')
    
    def delete_object(self, key: str) -> bool:
        """Delete object from S3."""
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete object {key}: {e}")
            return False
    
    def object_exists(self, key: str) -> bool:
        """Check if object exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except self.client.exceptions.NoSuchKey:
            return False
        except Exception:
            return False
    
    def get_object_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get object metadata from S3."""
        try:
            response = self.client.head_object(Bucket=self.bucket_name, Key=key)
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType'),
                'etag': response['ETag'].strip('"')
            }
        except Exception:
            return None
    
    def generate_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """Generate presigned URL for downloading."""
        try:
            return self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expiration
            )
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for {key}: {e}")
            return self._get_object_url(key)
    
    def _get_object_url(self, key: str) -> str:
        """Get direct object URL (for MinIO/development)."""
        return f"{self.endpoint_url}/{self.bucket_name}/{key}"
    
    def generate_unique_key(self, prefix: str, extension: str) -> str:
        """Generate unique key for object storage."""
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        unique_id = str(uuid.uuid4())
        return f"{prefix}/{timestamp}/{unique_id}{extension}"
    
    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> list:
        """List objects with given prefix."""
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            return response.get('Contents', [])
        except Exception as e:
            logger.error(f"Failed to list objects with prefix {prefix}: {e}")
            return []


# Global instance
s3_client = S3Client()