# BallQ_beta Integration Guide

## Overview

AutoHighlight has been prepared for seamless integration with BallQ_beta as a FastAPI-compatible submodule on the `autohighlights` branch.

## Files Added for Integration

### Core FastAPI Integration Files

1. **fastapi_wrapper.py**
   - `AutoHighlightAPI` class: Main wrapper around AutoHighlight functionality
   - `HighlightConfig` model: Pydantic configuration class
   - `HighlightResult` model: Result response model
   - `get_autohighlight_api()`: Singleton instance getter

2. **ballq_api.py**
   - FastAPI `router` object with endpoints ready for inclusion in BallQ_beta
   - `POST /autohighlight/analyze`: Analyze video and extract highlights
   - `POST /autohighlight/summarize`: Generate summary video
   - `GET /autohighlight/download/{task_id}`: Download processed video
   - `GET /autohighlight/health`: Health check endpoint

3. **setup.py**
   - `configure_autohighlight()`: Initialize AutoHighlight in your app
   - `get_autohighlight_router()`: Get the FastAPI router
   - `setup_autohighlight_path()`: Configure Python path
   - `create_app_with_autohighlight()`: Create standalone FastAPI app

### Documentation Files

- **BALLQ_INTEGRATION.md**: Complete integration instructions
- **requirements_fastapi.txt**: Additional dependencies needed
- **setup_ballq.sh**: Automated setup script

## Quick Integration Steps

### Step 1: Add as Submodule to BallQ_beta

```bash
cd /path/to/ballq_beta
git submodule add -b autohighlights https://github.com/dannywhite908/autohighlight.git services/autohighlight
cd services/autohighlight
git checkout autohighlights
cd ../..
git add .gitmodules services/autohighlight
git commit -m "Add AutoHighlight submodule for video summarization"
git push
```

### Step 2: Install Dependencies

```bash
pip install -r services/autohighlight/requirements_fastapi.txt
```

### Step 3: Integrate with FastAPI App

In your BallQ_beta main application file:

```python
from fastapi import FastAPI
import sys
from pathlib import Path

# Add submodule to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "autohighlight"))

# Import and include router
from ballq_api import router as autohighlight_router

app = FastAPI()

# Include AutoHighlight routes
app.include_router(autohighlight_router)

# ... rest of your BallQ_beta routes
```

### Step 4: Test Integration

```bash
# Test health endpoint
curl http://localhost:8000/autohighlight/health

# Test with a video file
curl -X POST "http://localhost:8000/autohighlight/analyze" \
  -F "file=@video.mp4" \
  -F "sampling_rate=75" \
  -F "classes=1,2" \
  -F "thresholds=0.9,0.95"
```

## API Reference

### POST /autohighlight/analyze

Analyze a video and extract highlight events.

**Parameters:**
- `file` (required): Video file (multipart form data)
- `model_path` (optional): Path to model file (default: `models/default.hdf5`)
- `sampling_rate` (optional): Inference sampling rate (default: 75)
- `classes` (optional): Comma-separated class indices (default: `1,2`)
- `thresholds` (optional): Comma-separated threshold values (default: `0.9,0.95`)

**Response:**
```json
{
  "success": true,
  "output_path": "",
  "events_count": 42,
  "events": [
    {
      "timestamp": 12.5,
      "class": 1,
      "probability": 0.92
    }
  ],
  "message": "Found 42 highlight events"
}
```

### POST /autohighlight/summarize

Analyze video and generate a summary video with highlighted moments.

**Parameters:**
- `file` (required): Video file
- `output_filename` (optional): Output video filename (default: `summary.mp4`)
- `model_path` (optional): Path to model file
- `sampling_rate` (optional): Inference sampling rate
- `classes` (optional): Comma-separated class indices
- `thresholds` (optional): Comma-separated threshold values

**Response:**
```json
{
  "success": true,
  "output_path": "/tmp/summary.mp4",
  "events_count": 42,
  "events": [...],
  "message": "Successfully processed video with 42 highlights"
}
```

### GET /autohighlight/download/{task_id}

Download a processed video file.

**Parameters:**
- `task_id`: Filename of the processed video (without extension)

**Response:** Video file

### GET /autohighlight/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "autohighlight"
}
```

## Python API Usage

If you want to use AutoHighlight programmatically within BallQ_beta:

```python
from autohighlight.fastapi_wrapper import get_autohighlight_api, HighlightConfig

# Get API instance
api = get_autohighlight_api("models/default.hdf5")

# Analyze video
inference_output = api.analyze_video("path/to/video.mp4", sampling_rate=75)

# Extract highlights
annotator, events = api.extract_highlights(
    inference_output,
    "path/to/video.mp4",
    classes=[1, 2],
    thresholds=[0.9, 0.95]
)

# Generate summary
api.summarize_video("path/to/video.mp4", "output/summary.mp4", annotator)

# Or use process_video for one-step processing
config = HighlightConfig(
    model_path="models/default.hdf5",
    sampling_rate=75,
    classes=[1, 2],
    thresholds=[0.9, 0.95]
)
result = api.process_video("input.mp4", "output.mp4", config)
```

## Configuration

### Environment Variables

Optionally set these in your BallQ_beta deployment:

```bash
AUTOHIGHLIGHT_MODEL_PATH=models/default.hdf5
AUTOHIGHLIGHT_SAMPLING_RATE=75
AUTOHIGHLIGHT_DEFAULT_CLASSES=1,2
AUTOHIGHLIGHT_DEFAULT_THRESHOLDS=0.9,0.95
```

### Custom Model Files

To use a different model:

1. Place the `.hdf5` file in `services/autohighlight/models/`
2. Reference it in API calls:
   ```python
   api.analyze_video("video.mp4", model_path="models/custom_model.hdf5")
   ```

## Performance Tuning

### Sampling Rate
- **Higher values** (100+): Faster processing, fewer samples
- **Lower values** (25-50): More detailed analysis, slower processing
- **Default (75)**: Good balance for soccer matches

### Class Selection
- Customize based on your use case
- Default `[1, 2]` for soccer: goals and cards
- Add/remove class indices as needed

### Threshold Tuning
- **Higher thresholds** (0.95+): Only very confident predictions
- **Lower thresholds** (0.7-0.8): More highlights detected
- Adjust per-class for domain-specific tuning

## Troubleshooting

### Model Not Found Error
```
FileNotFoundError: Model not found: models/default.hdf5
```
**Solution:** Verify the model file exists and path is correct relative to the working directory.

### Import Errors
```
ImportError: No module named 'utils'
```
**Solution:** Ensure sys.path includes the autohighlight directory as shown in Step 3.

### CORS Issues with BallQ_beta
Add CORS middleware to your FastAPI app:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Memory Issues with Large Videos
- Process videos in chunks
- Reduce sampling_rate to process fewer frames
- Increase time between inferences

## Testing

Create a test script to verify integration:

```python
# test_autohighlight_integration.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "services" / "autohighlight"))

from fastapi.testclient import TestClient
from ballq_api import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

client = TestClient(app)

def test_health():
    response = client.get("/autohighlight/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

if __name__ == "__main__":
    test_health()
    print("✓ AutoHighlight integration test passed")
```

Run with:
```bash
python test_autohighlight_integration.py
```

## Branch Information

- **Branch Name:** `autohighlights`
- **Parent Branch:** `master`
- **Repository:** https://github.com/dannywhite908/autohighlight.git
- **Status:** Ready for production use as BallQ_beta submodule

## Support

For issues or questions:
1. Check BALLQ_INTEGRATION.md for setup details
2. Review API documentation above
3. Check AutoHighlight original documentation in README.md
4. Open issues on the autohighlight repository

## License

AutoHighlight code and contents are released under MIT License.
Kinetics-pretrained weights are released by DeepMind under Apache 2.0 License.
