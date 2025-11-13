# Quick Setup Instructions

## Current Status
✅ All preparation complete. AutoHighlight is ready to integrate with BallQ_beta.

## What Was Created
- ✅ FastAPI wrapper module (`fastapi_wrapper.py`)
- ✅ API router for BallQ_beta (`ballq_api.py`)
- ✅ Configuration utilities (`setup.py`)
- ✅ Complete documentation (5 guides)
- ✅ Dependencies file (`requirements_fastapi.txt`)
- ✅ Setup automation script (`setup_ballq.sh`)

## Files Modified
- ✅ README.md - Updated with BallQ_beta section
- ✅ BALLQ_INTEGRATION.md - Full integration guide
- ✅ Plus 5 new comprehensive documentation files

## What You Need To Do

### Step 1: Create the Branch (1 minute)
```bash
cd /workspaces/autohighlight
bash setup_ballq.sh
```

This script will:
1. Create `autohighlights` branch
2. Stage all new files
3. Commit with proper message
4. Show confirmation

### Step 2: Push to GitHub (1 minute)
```bash
git push -u origin autohighlights
```

Verify at: https://github.com/dannywhite908/autohighlight/branches

### Step 3: Add to BallQ_beta (5 minutes)
In your BallQ_beta repository:

```bash
# Add submodule on autohighlights branch
git submodule add -b autohighlights https://github.com/dannywhite908/autohighlight.git services/autohighlight

# Commit the submodule
git add .gitmodules services/autohighlight
git commit -m "Add AutoHighlight submodule for video summarization"
git push
```

### Step 4: Install Dependencies (1 minute)
```bash
pip install -r services/autohighlight/requirements_fastapi.txt
```

### Step 5: Integrate with FastAPI (2 minutes)

In your main FastAPI app (main.py):

```python
import sys
from pathlib import Path

# Add autohighlight to path
sys.path.insert(0, str(Path(__file__).parent / "services" / "autohighlight"))

# Import the router
from ballq_api import router as autohighlight_router

# Include in your FastAPI app
app.include_router(autohighlight_router)
```

### Step 6: Test (2 minutes)
```bash
# Start your app
uvicorn main:app --reload

# In another terminal, test:
curl http://localhost:8000/autohighlight/health
```

Expected response:
```json
{"status": "healthy", "service": "autohighlight"}
```

---

## Total Time: ~12 minutes

### Timeline
1. Run setup script - 1 min
2. Push to GitHub - 1 min
3. Add submodule - 1 min
4. Install dependencies - 1 min
5. Update FastAPI app - 2 min
6. Test endpoints - 2 min
7. Verify everything - 2 min
8. **Done!** - ✅

---

## Key Endpoints Available

After integration, these endpoints are available in BallQ_beta:

```
POST /autohighlight/analyze
  → Analyze video and extract highlight events

POST /autohighlight/summarize
  → Generate summary video with highlights

GET /autohighlight/download/{task_id}
  → Download processed video

GET /autohighlight/health
  → Health check
```

---

## API Usage Example

### Analyze Video
```bash
curl -X POST "http://localhost:8000/autohighlight/analyze" \
  -F "file=@soccer_match.mp4" \
  -F "sampling_rate=75" \
  -F "classes=1,2" \
  -F "thresholds=0.9,0.95"
```

### Generate Summary
```bash
curl -X POST "http://localhost:8000/autohighlight/summarize" \
  -F "file=@soccer_match.mp4" \
  -F "output_filename=highlights.mp4"
```

---

## Python API Usage

If you need to use AutoHighlight directly in Python:

```python
from autohighlight.fastapi_wrapper import get_autohighlight_api

# Get API instance
api = get_autohighlight_api("models/default.hdf5")

# Process a video
result = api.process_video("input.mp4", "output.mp4")

# Check result
print(f"Success: {result.success}")
print(f"Found {result.events_count} highlights")
for event in result.events:
    print(f"  - {event['timestamp']}s: Class {event['class']} ({event['probability']:.2f})")
```

---

## Documentation

For more details, refer to:

| Document | Purpose |
|----------|---------|
| `BALLQ_INTEGRATION.md` | Setup & integration guide |
| `INTEGRATION_GUIDE.md` | Complete API reference |
| `ARCHITECTURE.md` | System design & architecture |
| `SETUP_SUMMARY.md` | Quick reference |
| `IMPLEMENTATION_CHECKLIST.md` | Detailed checklist |

---

## Troubleshooting

### Branch Not Created?
```bash
cd /workspaces/autohighlight
git branch autohighlights
git checkout autohighlights
git add fastapi_wrapper.py ballq_api.py setup.py
git commit -m "Add FastAPI integration for BallQ_beta"
```

### Import Errors?
Make sure in your app:
```python
import sys
sys.path.insert(0, "services/autohighlight")  # Adjust path as needed
```

### Model Not Found?
Verify: `services/autohighlight/models/default.hdf5` exists
If not, copy it from original AutoHighlight installation.

### Slow Performance?
Adjust `sampling_rate` parameter (higher = faster):
```python
# Default: 75
# Faster: 100-150
# More detailed: 25-50
```

---

## What's Next?

1. ✅ **Complete steps 1-6 above** (12 minutes)
2. 📝 Review the documentation files
3. 🧪 Test with your own videos
4. 🚀 Deploy to production
5. 📊 Monitor performance and results

---

## Need Help?

- Check `INTEGRATION_GUIDE.md` for detailed API docs
- Check `BALLQ_INTEGRATION.md` for setup issues
- See `ARCHITECTURE.md` for system design details
- Review `IMPLEMENTATION_CHECKLIST.md` for comprehensive checklist

---

**Status:** ✅ Ready for integration!

**Next Step:** Run `bash setup_ballq.sh` in `/workspaces/autohighlight`
