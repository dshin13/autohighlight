"""
FastAPI wrapper for AutoHighlight video summarizer.
Provides endpoints for video analysis and summarization.
Designed for integration with BallQ_beta pipeline.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, List
import numpy as np
from pydantic import BaseModel, Field

# AutoHighlight imports
from utils.event_detector import videoscan
from utils.annotator import Annotator


class HighlightConfig(BaseModel):
    """Configuration for video highlighting"""
    model_path: str = Field(default="models/default.hdf5", description="Path to the model file")
    sampling_rate: int = Field(default=75, description="Inference sampling rate in frames")
    classes: List[int] = Field(default=[1, 2], description="Classes to highlight")
    thresholds: List[float] = Field(default=[0.9, 0.95], description="Thresholds per class")
    highlight_duration: Optional[int] = Field(default=None, description="Highlight duration in seconds")


class HighlightResult(BaseModel):
    """Result from video highlighting"""
    success: bool
    output_path: str
    events_count: int
    events: Optional[List[dict]] = None
    message: str


class AutoHighlightAPI:
    """API wrapper for AutoHighlight functionality"""

    def __init__(self, model_path: str = "models/default.hdf5"):
        """Initialize AutoHighlight API
        
        Args:
            model_path: Path to the pre-trained model
        """
        self.model_path = model_path
        self.default_classes = [1, 2]
        self.default_thresholds = [0.9, 0.95]

    def analyze_video(
        self,
        source_path: str,
        model_path: Optional[str] = None,
        sampling_rate: int = 75,
        verbose: bool = False
    ) -> np.ndarray:
        """Perform inference on video file
        
        Args:
            source_path: Path to input video file
            model_path: Path to model (uses default if None)
            sampling_rate: Frame sampling interval
            verbose: Print progress
            
        Returns:
            Numpy array of class probabilities
        """
        model = model_path or self.model_path
        if not os.path.exists(model):
            raise FileNotFoundError(f"Model not found: {model}")
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Video not found: {source_path}")
            
        return videoscan(source_path, model, sampling_rate, verbose=verbose)

    def extract_highlights(
        self,
        inference_output: np.ndarray,
        source_path: str,
        classes: Optional[List[int]] = None,
        thresholds: Optional[List[float]] = None,
        highlight_duration: Optional[int] = None
    ) -> tuple[Annotator, List[dict]]:
        """Extract highlights from inference output
        
        Args:
            inference_output: Output from analyze_video
            source_path: Path to input video (for FPS extraction)
            classes: Classes to highlight
            thresholds: Thresholds per class
            highlight_duration: Duration in seconds
            
        Returns:
            Tuple of (Annotator object, list of events)
        """
        classes = classes or self.default_classes
        thresholds = thresholds or self.default_thresholds
        
        if len(classes) != len(thresholds):
            raise ValueError("Classes and thresholds must have same length")
        
        anno = Annotator(inference_output, classes, thresholds)
        anno.get_fps(source_path)
        anno.extract_timestamp()
        
        # Format events for API response
        events = [
            {
                "timestamp": float(event[0]),
                "class": int(event[1]),
                "probability": float(event[2])
            }
            for event in anno.events_sec
        ]
        
        return anno, events

    def summarize_video(
        self,
        source_path: str,
        output_path: str,
        annotator: Annotator
    ) -> bool:
        """Generate summary video
        
        Args:
            source_path: Path to input video
            output_path: Path to output video
            annotator: Annotator object with extracted events
            
        Returns:
            True if successful
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        annotator.summarize(source_path, output_path)
        return os.path.exists(output_path)

    def process_video(
        self,
        source_path: str,
        output_path: str,
        config: Optional[HighlightConfig] = None
    ) -> HighlightResult:
        """End-to-end video processing
        
        Args:
            source_path: Path to input video
            output_path: Path to output video
            config: Configuration options
            
        Returns:
            HighlightResult with status and details
        """
        config = config or HighlightConfig()
        
        try:
            # Analyze video
            inference_output = self.analyze_video(
                source_path,
                model_path=config.model_path,
                sampling_rate=config.sampling_rate
            )
            
            # Extract highlights
            annotator, events = self.extract_highlights(
                inference_output,
                source_path,
                classes=config.classes,
                thresholds=config.thresholds
            )
            
            # Generate summary
            self.summarize_video(source_path, output_path, annotator)
            
            return HighlightResult(
                success=True,
                output_path=output_path,
                events_count=len(events),
                events=events,
                message=f"Successfully processed video with {len(events)} highlights"
            )
        except Exception as e:
            return HighlightResult(
                success=False,
                output_path="",
                events_count=0,
                message=f"Error processing video: {str(e)}"
            )


# Singleton instance for use in FastAPI app
_autohighlight_api: Optional[AutoHighlightAPI] = None


def get_autohighlight_api(model_path: str = "models/default.hdf5") -> AutoHighlightAPI:
    """Get or create AutoHighlight API instance"""
    global _autohighlight_api
    if _autohighlight_api is None:
        _autohighlight_api = AutoHighlightAPI(model_path)
    return _autohighlight_api
