# AutoHighlight BallQ_beta Integration Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         BallQ_beta (Main App)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              FastAPI Application (main.py)              │  │
│  │                                                          │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │    Other BallQ_beta Routers & Endpoints       │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  │                        │                                │  │
│  │                        ▼                                │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │   AutoHighlight Router (included from         │   │  │
│  │  │   services/autohighlight/ballq_api.py)        │   │  │
│  │  │                                                 │   │  │
│  │  │   /autohighlight/analyze    ────────┐         │   │  │
│  │  │   /autohighlight/summarize  ────────┼─────┐   │   │  │
│  │  │   /autohighlight/download   ────────┤     │   │   │  │
│  │  │   /autohighlight/health     ────────┤     │   │   │  │
│  │  └──────────────────────────────────────┼─────┼───────┘   │  │
│  │                                         │     │            │  │
│  └─────────────────────────────────────────┼─────┼────────────┘  │
│                                            │     │               │
└────────────────────────────────────────────┼─────┼───────────────┘
                                             │     │
                    ┌────────────────────────┘     └──────────────┐
                    │                                              │
                    ▼                                              ▼
         ┌──────────────────────────┐        ┌────────────────────────────┐
         │ services/autohighlight   │        │  Video Processing & Output │
         │ (Submodule Branch:       │        │                            │
         │  autohighlights)         │        │  • Temp files              │
         │                          │        │  • Processed videos        │
         │  ┌────────────────────┐  │        │  • Highlight events       │
         │  │  fastapi_wrapper   │  │        └────────────────────────────┘
         │  │  - AutoHighlightAPI│  │
         │  │  - HighlightConfig │  │
         │  │  - HighlightResult │  │
         │  └────────────────────┘  │
         │           │              │
         │           ▼              │
         │  ┌────────────────────┐  │
         │  │   ballq_api.py     │  │
         │  │   (FastAPI Router) │  │
         │  └────────────────────┘  │
         │           │              │
         │           ▼              │
         │  ┌────────────────────┐  │
         │  │     setup.py       │  │
         │  │ (Configuration)    │  │
         │  └────────────────────┘  │
         │           │              │
         │           ▼              │
         │  ┌────────────────────┐  │
         │  │  Original modules  │  │
         │  │  - utils/          │  │
         │  │  - models/         │  │
         │  │  - autohighlight   │  │
         │  └────────────────────┘  │
         │                          │
         └──────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────┐
        │   Keras/TensorFlow       │
        │   + OpenCV Video         │
        │   + I3D Model            │
        │                          │
        │  models/default.hdf5     │
        └──────────────────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │   Video Analysis        │
         │   & Summarization       │
         └──────────────────────────┘
```

## Data Flow

### Analyze Endpoint
```
Client Request (Video File)
        │
        ▼
POST /autohighlight/analyze
        │
        ├─→ Save temp file
        │
        ├─→ videoscan() - Model inference
        │    • Load model
        │    • Extract frames
        │    • Predict classes
        │
        ├─→ Annotator - Extract highlights
        │    • Get FPS
        │    • Extract timestamps
        │
        └─→ Return JSON Response
             {
               success: true,
               events_count: N,
               events: [...]
             }
```

### Summarize Endpoint
```
Client Request (Video File)
        │
        ▼
POST /autohighlight/summarize
        │
        ├─→ Save temp input file
        │
        ├─→ videoscan() - Model inference
        │
        ├─→ Annotator - Extract highlights
        │
        ├─→ Annotator.summarize() - Generate summary video
        │    • Identify highlight clips
        │    • Extract clips
        │    • Concatenate into summary
        │
        ├─→ Save output file
        │
        └─→ Return JSON Response
             {
               success: true,
               output_path: "/tmp/summary.mp4",
               events_count: N
             }
```

## Deployment Structure

```
ballq_beta/
├── main.py                          # Main FastAPI app
├── requirements.txt                 # Base dependencies
├── services/
│   └── autohighlight/              # Submodule (autohighlights branch)
│       ├── models/
│       │   └── default.hdf5         # Pre-trained model
│       ├── utils/
│       │   ├── event_detector.py
│       │   ├── annotator.py
│       │   └── ...
│       ├── fastapi_wrapper.py       # API Wrapper
│       ├── ballq_api.py             # FastAPI Router
│       ├── setup.py                 # Configuration
│       ├── requirements_fastapi.txt # Integration dependencies
│       └── README.md                # Documentation
└── docker/
    └── Dockerfile                   # Include services/* in PYTHONPATH
```

## Class Relationships

```
┌─────────────────────────────────────┐
│        HighlightConfig              │
│  (Pydantic Model - Config holder)   │
├─────────────────────────────────────┤
│ + model_path: str                   │
│ + sampling_rate: int                │
│ + classes: List[int]                │
│ + thresholds: List[float]           │
└──────────┬──────────────────────────┘
           │
           │ used by
           ▼
┌─────────────────────────────────────┐
│      AutoHighlightAPI               │
│   (Main wrapper class)              │
├─────────────────────────────────────┤
│ - model_path: str                   │
│ + analyze_video()                   │
│ + extract_highlights()              │
│ + summarize_video()                 │
│ + process_video()                   │
└──────────┬──────────────────────────┘
           │
           │ uses
           ├────────────────┬──────────────────┐
           ▼                ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  videoscan   │  │  Annotator   │  │ file ops     │
    │ (inference)  │  │(highlighting)│  │(temp files)  │
    └──────────────┘  └──────────────┘  └──────────────┘

           │ returns
           ▼
┌─────────────────────────────────────┐
│       HighlightResult               │
│ (Pydantic Model - Response)         │
├─────────────────────────────────────┤
│ + success: bool                     │
│ + output_path: str                  │
│ + events_count: int                 │
│ + events: List[dict]                │
│ + message: str                      │
└─────────────────────────────────────┘
```

## Integration Points

### 1. Router Inclusion
```python
# In BallQ_beta main.py
from autohighlight.ballq_api import router
app.include_router(router)
```

### 2. Direct API Usage
```python
# In BallQ_beta services
from autohighlight.fastapi_wrapper import get_autohighlight_api
api = get_autohighlight_api()
result = api.process_video(...)
```

### 3. Configuration
```python
# In BallQ_beta initialization
from autohighlight.setup import configure_autohighlight
configure_autohighlight(model_path="custom_model.hdf5")
```

## Environment Variables

```bash
# Optional environment configuration
AUTOHIGHLIGHT_MODEL_PATH=models/default.hdf5
AUTOHIGHLIGHT_SAMPLING_RATE=75
AUTOHIGHLIGHT_DEFAULT_CLASSES=1,2
AUTOHIGHLIGHT_DEFAULT_THRESHOLDS=0.9,0.95
```

## Performance Characteristics

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Analyze (5min video) | 30-60s | 2-4GB | Depends on sampling_rate |
| Summarize (5min video) | 45-90s | 3-5GB | Includes video generation |
| Health Check | <10ms | <1MB | Instant response |

## Error Handling

```
Request
  │
  ├─ File validation
  │   ├─ success → continue
  │   └─ error → HTTP 400
  │
  ├─ Model loading
  │   ├─ success → continue
  │   └─ error → HTTP 400
  │
  ├─ Video processing
  │   ├─ success → continue
  │   └─ error → HTTP 400 (cleanup temp files)
  │
  └─ Response → HighlightResult
```

## Branch Strategy

```
autohighlight repository
├── master (original code)
│   └── autohighlights (BallQ_beta integration branch)
│       ├── fastapi_wrapper.py
│       ├── ballq_api.py
│       ├── setup.py
│       ├── BALLQ_INTEGRATION.md
│       └── INTEGRATION_GUIDE.md
```

---

**This architecture ensures:**
- ✅ Clean separation of concerns
- ✅ Easy integration as submodule
- ✅ FastAPI compatibility
- ✅ Configurable and extensible
- ✅ Production-ready error handling
- ✅ Performance optimized
