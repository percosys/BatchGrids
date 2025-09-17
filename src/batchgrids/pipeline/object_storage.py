"""
Object Storage System

Manages detected objects with their properties, transformations,
and gridfinity parameters for multi-object layouts.

Author: BatchGrids Pipeline
"""

import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import cv2
from pathlib import Path


@dataclass
class ObjectTransform:
    """Object transformation parameters"""
    rotation_degrees: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    scale: float = 1.0


@dataclass  
class GridfinityParameters:
    """Gridfinity-specific parameters"""
    clearance_mm: float = 1.5
    depth_mm: float = 7.0  # Standard gridfinity depth
    fillet_radius_mm: float = 1.0
    wall_thickness_mm: float = 1.2
    requires_custom_depth: bool = False


@dataclass
class DetectedObject:
    """Complete detected object with all properties"""
    
    # Identification
    object_id: str
    object_type: str
    confidence: float
    
    # Geometry
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray  # Binary mask
    contour: np.ndarray  # Simplified contour points
    
    # Physical properties
    dimensions_mm: Tuple[float, float]  # width, height in mm
    area_mm2: float
    perimeter_mm: float
    
    # Processing metadata
    detection_method: str
    processing_steps: List[str]
    quality_score: float  # 0-1, overall detection quality
    
    # Gridfinity properties
    transform: ObjectTransform
    gridfinity_params: GridfinityParameters
    
    # Storage
    timestamp: str
    source_image_hash: str


class ObjectDatabase:
    """Database for managing detected objects"""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.objects_dir = self.storage_dir / \"objects\"
        self.layouts_dir = self.storage_dir / \"layouts\"
        self.objects_dir.mkdir(exist_ok=True)
        self.layouts_dir.mkdir(exist_ok=True)
        
        self.objects: Dict[str, DetectedObject] = {}
        self._load_existing_objects()
    
    def add_object(self, obj: DetectedObject) -> str:
        \"\"\"Add object to database\"\"\"
        
        # Generate unique ID if not provided
        if not obj.object_id:
            obj.object_id = str(uuid.uuid4())
        
        # Store in memory
        self.objects[obj.object_id] = obj
        
        # Persist to disk
        self._save_object(obj)
        
        print(f\"💾 Stored object: {obj.object_id} ({obj.object_type})\"\")\")
        return obj.object_id
    
    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        \"\"\"Retrieve object by ID\"\"\"
        return self.objects.get(object_id)
    
    def list_objects(self, object_type: Optional[str] = None) -> List[DetectedObject]:
        \"\"\"List all objects, optionally filtered by type\"\"\"
        if object_type:
            return [obj for obj in self.objects.values() if obj.object_type == object_type]
        return list(self.objects.values())
    
    def update_object_transform(self, object_id: str, transform: ObjectTransform) -> bool:
        \"\"\"Update object transformation\"\"\"
        if object_id in self.objects:
            self.objects[object_id].transform = transform
            self._save_object(self.objects[object_id])
            print(f\"🔄 Updated transform for {object_id}\"\")\")
            return True
        return False
    
    def delete_object(self, object_id: str) -> bool:
        \"\"\"Delete object from database\"\"\"
        if object_id in self.objects:
            del self.objects[object_id]
            
            # Remove from disk
            object_file = self.objects_dir / f\"{object_id}.json\"
            mask_file = self.objects_dir / f\"{object_id}_mask.png\"
            contour_file = self.objects_dir / f\"{object_id}_contour.npy\"
            
            for file in [object_file, mask_file, contour_file]:
                if file.exists():
                    file.unlink()
            
            print(f\"🗑️ Deleted object: {object_id}\"\")\")
            return True
        return False
    
    def _save_object(self, obj: DetectedObject):
        \"\"\"Save object to disk\"\"\"
        
        # Save metadata (without numpy arrays)
        metadata = asdict(obj)
        metadata.pop('mask', None)
        metadata.pop('contour', None)
        
        # Save metadata as JSON
        with open(self.objects_dir / f\"{obj.object_id}.json\", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save mask as PNG
        cv2.imwrite(str(self.objects_dir / f\"{obj.object_id}_mask.png\"), obj.mask)
        
        # Save contour as numpy array
        np.save(self.objects_dir / f\"{obj.object_id}_contour.npy\", obj.contour)
    
    def _load_existing_objects(self):
        \"\"\"Load existing objects from disk\"\"\"
        
        for json_file in self.objects_dir.glob(\"*.json\"):
            try:
                object_id = json_file.stem
                
                # Load metadata
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                # Load mask
                mask_file = self.objects_dir / f\"{object_id}_mask.png\"
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) if mask_file.exists() else np.array([])
                
                # Load contour
                contour_file = self.objects_dir / f\"{object_id}_contour.npy\"
                contour = np.load(contour_file) if contour_file.exists() else np.array([])
                
                # Reconstruct object
                obj = DetectedObject(
                    object_id=metadata['object_id'],
                    object_type=metadata['object_type'],
                    confidence=metadata['confidence'],
                    bbox=tuple(metadata['bbox']),
                    mask=mask,
                    contour=contour,
                    dimensions_mm=tuple(metadata['dimensions_mm']),
                    area_mm2=metadata['area_mm2'],
                    perimeter_mm=metadata['perimeter_mm'],
                    detection_method=metadata['detection_method'],
                    processing_steps=metadata['processing_steps'],
                    quality_score=metadata['quality_score'],
                    transform=ObjectTransform(**metadata['transform']),
                    gridfinity_params=GridfinityParameters(**metadata['gridfinity_params']),
                    timestamp=metadata['timestamp'],
                    source_image_hash=metadata['source_image_hash']
                )
                
                self.objects[object_id] = obj
                
            except Exception as e:
                print(f\"⚠️ Failed to load object {json_file}: {e}\"\")\")


@dataclass
class GridfinityLayout:
    \"\"\"Multi-object gridfinity layout\"\"\"
    
    layout_id: str
    name: str
    object_ids: List[str]
    grid_size: Tuple[int, int]  # width, height in grid units
    object_placements: Dict[str, Tuple[float, float, float]]  # object_id -> (x, y, rotation)
    total_area_mm2: float
    estimated_print_time_hours: float
    created_timestamp: str


class LayoutManager:
    \"\"\"Manages multi-object gridfinity layouts\"\"\"
    
    def __init__(self, object_db: ObjectDatabase):
        self.object_db = object_db
        self.layouts: Dict[str, GridfinityLayout] = {}
    
    def create_layout(self, name: str, object_ids: List[str]) -> GridfinityLayout:
        \"\"\"Create new multi-object layout\"\"\"
        
        layout_id = str(uuid.uuid4())
        
        # Get objects
        objects = [self.object_db.get_object(oid) for oid in object_ids]
        objects = [obj for obj in objects if obj is not None]
        
        if not objects:
            raise ValueError(\"No valid objects provided for layout\")
        
        # Calculate optimal arrangement
        placements = self._calculate_optimal_placement(objects)
        
        # Estimate grid size needed
        grid_size = self._estimate_grid_size(objects, placements)
        
        # Calculate total area
        total_area = sum(obj.area_mm2 for obj in objects)
        
        # Estimate print time (rough calculation)
        estimated_print_time = self._estimate_print_time(objects, grid_size)
        
        layout = GridfinityLayout(
            layout_id=layout_id,
            name=name,
            object_ids=object_ids,
            grid_size=grid_size,
            object_placements=placements,
            total_area_mm2=total_area,
            estimated_print_time_hours=estimated_print_time,
            created_timestamp=datetime.now().isoformat()
        )
        
        self.layouts[layout_id] = layout
        
        print(f\"🏗️ Created layout '{name}': {len(objects)} objects, {grid_size[0]}x{grid_size[1]} grid\")
        
        return layout
    
    def _calculate_optimal_placement(self, objects: List[DetectedObject]) -> Dict[str, Tuple[float, float, float]]:
        \"\"\"Calculate optimal placement for objects\"\"\"
        
        placements = {}
        
        # Simple placement algorithm - can be made more sophisticated
        current_x = 0
        current_y = 0
        row_height = 0
        
        for obj in objects:
            # Consider rotation for better packing
            width_mm, height_mm = obj.dimensions_mm
            
            # Try both orientations and pick more space-efficient
            if width_mm > height_mm and current_x > 0:
                # Rotate 90 degrees if it fits better
                rotation = 90.0
                obj_width, obj_height = height_mm, width_mm
            else:
                rotation = 0.0
                obj_width, obj_height = width_mm, height_mm
            
            placements[obj.object_id] = (current_x, current_y, rotation)
            
            # Update positions for next object
            current_x += obj_width + 5  # 5mm spacing
            row_height = max(row_height, obj_height)
            
            # Start new row if needed (simplified)
            if current_x > 200:  # Max row width
                current_x = 0
                current_y += row_height + 5
                row_height = 0
        
        return placements
    
    def _estimate_grid_size(self, objects: List[DetectedObject], placements: Dict[str, Tuple[float, float, float]]) -> Tuple[int, int]:
        \"\"\"Estimate required grid size\"\"\"
        
        if not placements:
            return (1, 1)
        
        # Find bounding box of all placements
        max_x = max_y = 0
        
        for obj in objects:
            if obj.object_id in placements:
                x, y, rotation = placements[obj.object_id]
                width_mm, height_mm = obj.dimensions_mm
                
                # Account for rotation
                if abs(rotation - 90) < 45:  # Rotated
                    obj_width, obj_height = height_mm, width_mm
                else:
                    obj_width, obj_height = width_mm, height_mm
                
                max_x = max(max_x, x + obj_width)
                max_y = max(max_y, y + obj_height)
        
        # Convert to grid units (42mm per unit)
        grid_width = int(np.ceil(max_x / 42))
        grid_height = int(np.ceil(max_y / 42))
        
        return (max(1, grid_width), max(1, grid_height))
    
    def _estimate_print_time(self, objects: List[DetectedObject], grid_size: Tuple[int, int]) -> float:
        \"\"\"Rough estimate of print time in hours\"\"\"
        
        # Very rough estimation based on area and complexity
        total_area_mm2 = sum(obj.area_mm2 for obj in objects)
        base_area_mm2 = grid_size[0] * grid_size[1] * 42 * 42  # Grid base area
        
        # Assume ~0.2mm layer height, ~50mm/s print speed
        # This is a very rough approximation
        estimated_hours = (total_area_mm2 + base_area_mm2) / (50 * 60 * 60 * 0.2)
        
        return max(0.5, estimated_hours)  # Minimum 30 minutes