"""
Intelligent YOLOE Prompting System

Dynamically generates YOLOE vocabulary based on preprocessor findings
and manages object-specific detection strategies.

Author: BatchGrids Pipeline
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

from .preprocessor import ObjectHint, PreprocessingResult


@dataclass
class YOLOEStrategy:
    """YOLOE detection strategy for specific object types"""
    vocabulary: List[str]
    confidence_threshold: float
    iou_threshold: float
    description: str


class IntelligentYOLOEPrompting:
    """Intelligently generate YOLOE prompts based on preprocessing"""
    
    def __init__(self):
        # Object-specific vocabularies
        self.vocabularies = {
            'elongated_tool': [
                'knife', 'kitchen knife', 'chef knife', 'paring knife', 'utility knife',
                'screwdriver', 'flathead screwdriver', 'phillips screwdriver',
                'allen wrench', 'hex key', 'torx driver',
                'file', 'metal file', 'hand file',
                'chisel', 'wood chisel', 'metal chisel',
                'awl', 'ice pick', 'leather punch',
                'ruler', 'measuring stick', 'straightedge'
            ],
            
            'compact_tool': [
                'pliers', 'needle nose pliers', 'flush cutting pliers',
                'wire cutters', 'side cutters', 'diagonal cutters',
                'crimping pliers', 'locking pliers', 'vice grips',
                'tin snips', 'aviation snips',
                'multimeter', 'digital caliper', 'micrometer',
                'small hammer', 'tack hammer', 'ball peen hammer'
            ],
            
            'medium_tool': [
                'wrench', 'adjustable wrench', 'crescent wrench',
                'box wrench', 'open end wrench', 'combination wrench',
                'socket wrench', 'ratchet handle',
                'hammer', 'claw hammer', 'framing hammer',
                'saw handle', 'hand saw', 'hacksaw',
                'level', 'torpedo level', 'spirit level'
            ],
            
            'electronics': [
                'circuit board', 'PCB', 'microcontroller',
                'resistor', 'capacitor', 'transistor',
                'integrated circuit', 'IC', 'chip',
                'connector', 'header pins', 'jumper wires',
                'breadboard', 'prototype board',
                'multimeter probe', 'test lead'
            ],
            
            'fasteners': [
                'screw', 'bolt', 'nut', 'washer',
                'hex bolt', 'machine screw', 'wood screw',
                'socket head screw', 'button head screw',
                'wing nut', 'hex nut', 'lock nut',
                'flat washer', 'lock washer', 'spring washer'
            ]
        }
        
        # Fallback general vocabulary
        self.general_vocabulary = [
            'tool', 'hand tool', 'cutting tool', 'measuring tool',
            'mechanical tool', 'precision tool', 'workshop tool',
            'hardware', 'metal object', 'precision instrument'
        ]
    
    def generate_strategy(self, preprocessing_result: PreprocessingResult) -> YOLOEStrategy:
        """
        Generate YOLOE detection strategy based on preprocessing results
        
        Args:
            preprocessing_result: Results from image preprocessing
            
        Returns:
            YOLOEStrategy with optimized vocabulary and parameters
        """
        
        print("🧠 INTELLIGENT YOLOE: Generating detection strategy...")
        
        # Analyze object hints
        vocabulary = set()
        confidence_adjustments = []
        
        for hint in preprocessing_result.object_hints:
            object_type = hint.object_type
            
            if object_type in self.vocabularies:
                # Add specific vocabulary for detected object type
                specific_vocab = self.vocabularies[object_type]
                vocabulary.update(specific_vocab)
                
                print(f"  📝 Added {len(specific_vocab)} terms for '{object_type}'")
                
                # Adjust confidence based on object characteristics
                if hint.confidence > 0.8:
                    confidence_adjustments.append(0.01)  # Very confident, lower threshold
                elif hint.confidence > 0.6:
                    confidence_adjustments.append(0.02)  # Moderately confident
                else:
                    confidence_adjustments.append(0.03)  # Less confident, higher threshold
        
        # Add general vocabulary if no specific objects detected
        if not vocabulary:
            vocabulary.update(self.general_vocabulary)
            print(f"  📝 Using general vocabulary ({len(self.general_vocabulary)} terms)\"")
        
        # Add complementary vocabularies for better coverage
        vocabulary.update(self._get_complementary_vocabulary(list(vocabulary)))
        
        # Calculate optimal detection parameters
        base_confidence = 0.005
        if confidence_adjustments:
            confidence_threshold = base_confidence + np.mean(confidence_adjustments)
        else:
            confidence_threshold = 0.01  # Default for general detection
        
        # IOU threshold based on expected object density
        num_objects = len(preprocessing_result.object_hints)
        if num_objects > 3:
            iou_threshold = 0.2  # Lower for crowded scenes
        else:
            iou_threshold = 0.15  # Standard for single/few objects
        
        strategy = YOLOEStrategy(
            vocabulary=list(vocabulary),
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            description=f"Optimized for {num_objects} detected objects"
        )
        
        print(f"  ✅ Strategy: {len(strategy.vocabulary)} terms, conf={confidence_threshold:.3f}, iou={iou_threshold:.2f}")
        
        return strategy
    
    def _get_complementary_vocabulary(self, primary_vocab: List[str]) -> List[str]:
        \"\"\"Add complementary terms to improve detection coverage\"\"\"
        
        complementary = []
        
        # If we have cutting tools, add related terms
        cutting_terms = ['knife', 'cutter', 'blade', 'cutting']
        if any(term in ' '.join(primary_vocab).lower() for term in cutting_terms):
            complementary.extend([
                'blade', 'cutting edge', 'sharp tool',
                'kitchen utensil', 'culinary tool'
            ])
        
        # If we have electronics terms, add related
        electronics_terms = ['circuit', 'electronic', 'digital']
        if any(term in ' '.join(primary_vocab).lower() for term in electronics_terms):
            complementary.extend([
                'electronic component', 'electronic device',
                'measurement device', 'test equipment'
            ])
        
        # If we have hand tools, add variations
        tool_terms = ['tool', 'wrench', 'pliers', 'screwdriver']
        if any(term in ' '.join(primary_vocab).lower() for term in tool_terms):
            complementary.extend([
                'hand tool', 'workshop tool', 'mechanical tool',
                'precision tool', 'maintenance tool'
            ])
        
        return complementary
    
    def refine_detection(self, 
                        strategy: YOLOEStrategy,
                        detection_results: List[Dict],
                        preprocessing_result: PreprocessingResult) -> YOLOEStrategy:
        \"\"\"
        Refine strategy based on initial detection results
        
        Args:
            strategy: Current YOLOE strategy
            detection_results: Results from initial YOLOE run
            preprocessing_result: Original preprocessing results
            
        Returns:
            Refined YOLOEStrategy
        \"\"\"
        
        print(\"🔄 REFINING: Adjusting strategy based on results...\")
        
        # Analyze detection quality
        if not detection_results:
            # No detections - lower confidence threshold
            new_confidence = max(0.001, strategy.confidence_threshold * 0.5)
            print(f"  📉 No detections, lowering confidence: {strategy.confidence_threshold:.3f} → {new_confidence:.3f}")
            
            return YOLOEStrategy(
                vocabulary=strategy.vocabulary + self.general_vocabulary,
                confidence_threshold=new_confidence,
                iou_threshold=strategy.iou_threshold,
                description=\"Refined for better detection coverage\"
            )
        
        # Too many low-confidence detections - raise threshold
        low_conf_count = sum(1 for result in detection_results if result.get('confidence', 1.0) < 0.1)
        if low_conf_count > len(detection_results) * 0.7:
            new_confidence = min(0.05, strategy.confidence_threshold * 2.0)
            print(f"  📈 Too many low-conf detections, raising threshold: {strategy.confidence_threshold:.3f} → {new_confidence:.3f}")
            
            return YOLOEStrategy(
                vocabulary=strategy.vocabulary,
                confidence_threshold=new_confidence,
                iou_threshold=strategy.iou_threshold,
                description=\"Refined for higher precision\"
            )
        
        # Strategy seems good
        print(\"  ✅ Strategy performing well, no changes needed\")
        return strategy