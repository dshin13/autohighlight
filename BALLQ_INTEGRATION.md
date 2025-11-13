# AutoHighlight Integration Configuration for BallQ_beta

## Submodule Path
Path in BallQ_beta: `services/autohighlight` or `modules/autohighlight`

## Setup Instructions

### 1. Adding AutoHighlight as a submodule to BallQ_beta:

```bash
cd /path/to/ballq_beta
git submodule add https://github.com/dannywhite908/autohighlight.git services/autohighlight
cd services/autohighlight
git checkout autohighlights
cd ../..
git add .gitmodules services/autohighlight
git commit -m "Add AutoHighlight submodule on autohighlights branch"
```

### 2. FastAPI Integration

The `autohighlight` service provides the following imports for your FastAPI application:

```python
# In your BallQ_beta main FastAPI app (e.g., main.py or app.py):
from fastapi import FastAPI
from autohighlight.ballq_api import router as autohighlight_router

app = FastAPI()

# Include the AutoHighlight router
app.include_router(autohighlight_router)
```

### 3. API Endpoints

Once integrated, the following endpoints are available:

- `POST /autohighlight/analyze` - Analyze video and extract highlights
- `POST /autohighlight/summarize` - Generate summary video
- `GET /autohighlight/download/{task_id}` - Download processed video
- `GET /autohighlight/health` - Health check

### 4. Example Usage

```python
# Using the Python API directly
from autohighlight.fastapi_wrapper import get_autohighlight_api, HighlightConfig

api = get_autohighlight_api("models/default.hdf5")

# Analyze video
inference_output = api.analyze_video("video.mp4", sampling_rate=75)

# Extract highlights
annotator, events = api.extract_highlights(
    inference_output,
    "video.mp4",
    classes=[1, 2],
    thresholds=[0.9, 0.95]
)

# Generate summary
api.summarize_video("video.mp4", "summary.mp4", annotator)
```

### 5. Dependencies

The wrapper requires these additional dependencies:
- `fastapi>=0.100.0`
- `pydantic>=2.0`
- `python-multipart>=0.0.6` (for file uploads)

Add to your BallQ_beta requirements.txt

### 6. Model Files

Ensure the model file exists at:
`autohighlight/models/default.hdf5`

For custom models, specify the path in the API calls or configuration.

### 7. Environment Variables (Optional)

You can set these environment variables in your BallQ_beta deployment:

```bash
AUTOHIGHLIGHT_MODEL_PATH=models/default.hdf5
AUTOHIGHLIGHT_SAMPLING_RATE=75
AUTOHIGHLIGHT_DEFAULT_CLASSES=1,2
AUTOHIGHLIGHT_DEFAULT_THRESHOLDS=0.9,0.95
```

## Branch Information

- Branch name: `autohighlights`
- Purpose: FastAPI-compatible version for BallQ_beta integration
- Parent branch: `master`
- Status: Ready for submodule integration

## Testing the Integration

```bash
# From BallQ_beta root:
cd services/autohighlight

# Test the API wrapper
python -m pytest tests/test_fastapi_wrapper.py -v

# Test with FastAPI dev server
uvicorn ballq_api:app --reload --port 8001
```

## Troubleshooting

### ImportError: No module named 'utils'
- Ensure Python path includes the autohighlight directory
- In BallQ_beta, add to your app initialization:
  ```python
  import sys
  sys.path.insert(0, str(Path(__file__).parent / "services" / "autohighlight"))
  ```

### Model file not found
- Verify `models/default.hdf5` exists in the autohighlight directory
- Check file permissions
- Use absolute paths if relative paths fail

### Performance Issues
- Adjust `sampling_rate` parameter (higher = faster but less accurate)
- Use GPU acceleration if available (Keras/TensorFlow configuration)
- Process videos asynchronously using background tasks

## Contact & Support

For issues specific to AutoHighlight integration with BallQ_beta, refer to:
- AutoHighlight original docs: `autohighlight/README.md`
- BallQ_beta integration: See main BallQ_beta documentation
