"""
BatchGrids Intelligent Pipeline

A modular, intelligent pipeline for gridfinity object detection and layout generation.

Components:
- ImagePreprocessor: Scale detection and image enhancement
- IntelligentYOLOEPrompting: Dynamic vocabulary generation
- ObjectDatabase: Object storage and management
- LayoutManager: Multi-object layout optimization
- IntegratedPipeline: Complete pipeline orchestration

Author: BatchGrids Pipeline Team
"""

from .preprocessor import ImagePreprocessor, PreprocessingResult, ScaleCalibration, ObjectHint
from .intelligent_yoloe import IntelligentYOLOEPrompting, YOLOEStrategy
from .object_storage import (
    ObjectDatabase, DetectedObject, ObjectTransform, 
    GridfinityParameters, LayoutManager, GridfinityLayout
)
from .integrated_pipeline import IntegratedPipeline

__all__ = [
    # Core pipeline
    'IntegratedPipeline',
    
    # Preprocessing
    'ImagePreprocessor', 'PreprocessingResult', 'ScaleCalibration', 'ObjectHint',
    
    # Intelligent YOLOE
    'IntelligentYOLOEPrompting', 'YOLOEStrategy',
    
    # Object storage
    'ObjectDatabase', 'DetectedObject', 'ObjectTransform', 
    'GridfinityParameters', 'LayoutManager', 'GridfinityLayout',
]

__version__ = "1.0.0"
__author__ = "BatchGrids Pipeline Team"