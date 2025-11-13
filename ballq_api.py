"""
FastAPI integration module for BallQ_beta pipeline.
Provides REST API endpoints for AutoHighlight functionality.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse
import tempfile
import os
from pathlib import Path
from typing import Optional, List
import shutil

from fastapi_wrapper import get_autohighlight_api, HighlightConfig, HighlightResult


# Create router for submodule integration
router = APIRouter(prefix="/autohighlight", tags=["autohighlight"])


@router.post("/analyze", response_model=HighlightResult)
async def analyze_video(
    file: UploadFile = File(...),
    model_path: str = Query("models/default.hdf5"),
    sampling_rate: int = Query(75),
    classes: str = Query("1,2"),
    thresholds: str = Query("0.9,0.95"),
):
    """Analyze video and extract highlights
    
    Args:
        file: Video file to analyze
        model_path: Path to model file
        sampling_rate: Inference sampling rate
        classes: Comma-separated class indices
        thresholds: Comma-separated threshold values
        
    Returns:
        HighlightResult with events and summary statistics
    """
    try:
        # Parse query parameters
        class_list = [int(c.strip()) for c in classes.split(",")]
        threshold_list = [float(t.strip()) for t in thresholds.split(",")]
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Get API instance
            api = get_autohighlight_api(model_path)
            
            # Analyze video
            inference_output = api.analyze_video(
                tmp_path,
                model_path=model_path,
                sampling_rate=sampling_rate
            )
            
            # Extract highlights
            annotator, events = api.extract_highlights(
                inference_output,
                tmp_path,
                classes=class_list,
                thresholds=threshold_list
            )
            
            return HighlightResult(
                success=True,
                output_path="",
                events_count=len(events),
                events=events,
                message=f"Found {len(events)} highlight events"
            )
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/summarize", response_model=HighlightResult)
async def summarize_video(
    file: UploadFile = File(...),
    output_filename: str = Query("summary.mp4"),
    model_path: str = Query("models/default.hdf5"),
    sampling_rate: int = Query(75),
    classes: str = Query("1,2"),
    thresholds: str = Query("0.9,0.95"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Analyze video and generate summary video
    
    Args:
        file: Video file to summarize
        output_filename: Name for output video
        model_path: Path to model file
        sampling_rate: Inference sampling rate
        classes: Comma-separated class indices
        thresholds: Comma-separated threshold values
        background_tasks: Background tasks handler
        
    Returns:
        HighlightResult with output path and event details
    """
    try:
        # Parse query parameters
        class_list = [int(c.strip()) for c in classes.split(",")]
        threshold_list = [float(t.strip()) for t in thresholds.split(",")]
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            content = await file.read()
            tmp_input.write(content)
            tmp_input_path = tmp_input.name
        
        # Create output path in temp directory
        tmp_output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        try:
            # Get API instance
            api = get_autohighlight_api(model_path)
            
            # Process video
            config = HighlightConfig(
                model_path=model_path,
                sampling_rate=sampling_rate,
                classes=class_list,
                thresholds=threshold_list
            )
            
            result = api.process_video(tmp_input_path, tmp_output_path, config)
            
            if result.success:
                # Schedule cleanup of temp files
                background_tasks.add_task(cleanup_temp_files, tmp_input_path, tmp_output_path)
                result.output_path = tmp_output_path
            else:
                # Clean up on failure
                if os.path.exists(tmp_input_path):
                    os.unlink(tmp_input_path)
            
            return result
            
        except Exception as e:
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)
            raise HTTPException(status_code=400, detail=str(e))
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/download/{task_id}")
async def download_video(task_id: str):
    """Download processed video
    
    Args:
        task_id: Identifier for the processed video
        
    Returns:
        Video file for download
    """
    file_path = os.path.join(tempfile.gettempdir(), f"{task_id}.mp4")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(file_path, filename=f"{task_id}.mp4")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "autohighlight"}


def cleanup_temp_files(input_path: str, output_path: str):
    """Clean up temporary files"""
    try:
        if os.path.exists(input_path):
            os.unlink(input_path)
    except:
        pass
