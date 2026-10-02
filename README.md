# AI-Based Art Restoration Using Deep Learning

A local web application that accepts damaged or degraded artworks/photographs and produces AI-enhanced restorations using Real-ESRGAN for general super-resolution and GFPGAN for face restoration.

## Features

- **General Image Restoration** — Real-ESRGAN x4 super-resolution and enhancement
- **Face Restoration** — GFPGAN v1.4 for facial detail recovery (when faces are detected)
- **Drag & Drop Upload** — Modern file upload with drag-and-drop and click-to-browse
- **Before/After Comparison** — Interactive slider to compare original vs. restored image
- **Format Support** — JPG, JPEG, PNG, WEBP (max 20 MB)
- **Client & Server Validation** — File type, size, MIME type, corruption detection
- **CPU & GPU Support** — Automatic CUDA detection with CPU fallback
- **Download** — Download the restored image directly from the browser
- **Responsive Design** — Works on desktop and mobile devices
- **Security** — Filename sanitization, path traversal prevention, upload isolation

## Architecture

```
Upload → Validate → Decode/Preprocess → Real-ESRGAN (4x SR)
    → Detect Faces → GFPGAN (if faces found) → Postprocess → Save → Display
```

```
art-restoration/
├── app.py                          # Flask application factory
├── config.py                       # Configuration (paths, limits, device)
├── requirements.txt                # Python dependencies
├── models/
│   ├── model_manager.py            # Lazy-loading model manager
│   ├── realesrgan_model.py         # Real-ESRGAN wrapper
│   ├── gfpgan_model.py             # GFPGAN wrapper
│   └── rrdbnet_arch.py             # RRDBNet architecture (local)
├── pipeline/
│   ├── restoration_pipeline.py     # Main pipeline orchestrator
│   ├── image_utils.py              # Validation, I/O utilities
│   ├── preprocessing.py            # Image preprocessing
│   ├── face_processing.py          # Face detection
│   └── postprocessing.py           # Output processing
├── routes/
│   ├── web.py                      # Homepage route
│   └── api.py                      # REST API endpoints
├── templates/
│   └── index.html                  # Single-page frontend
├── static/
│   ├── css/style.css               # Dark museum-themed styles
│   └── js/app.js                   # Client-side logic
├── tests/
│   ├── test_validation.py          # Unit tests (20 tests)
│   └── test_api.py                 # API endpoint tests (12 tests)
├── weights/                        # Model weights (auto-downloaded)
├── uploads/                        # Temporary upload storage
└── outputs/                        # Restored images
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask |
| AI Models | PyTorch, Real-ESRGAN, GFPGAN |
| Image Processing | OpenCV, Pillow, NumPy |
| Frontend | HTML5, CSS3 (custom properties), Vanilla JavaScript |
| Testing | pytest |

## Models

### Real-ESRGAN

- **Model**: RealESRGAN_x4plus (RRDBNet backbone)
- **Scale**: 4x super-resolution
- **Weights**: `RealESRGAN_x4plus.pth` (~64 MB)
- **Source**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- Tile-based processing (256px tiles on CPU, 512px on GPU)
- Uses pretrained weights — no training from scratch

### GFPGAN

- **Model**: GFPGANv1.4 (clean architecture)
- **Weights**: `GFPGANv1.4.pth` (~333 MB)
- **Source**: [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)
- Face detection + restoration with background preservation
- Falls back gracefully if no faces detected

> **Note**: AI-generated facial details are plausible reconstructions, not guaranteed historical truth.

## Installation

### Prerequisites

- Python 3.10+ (tested with 3.12 and 3.13)
- pip

### Steps

```bash
# 1. Navigate to the project directory
cd "Art Restoration/art-restoration"

# 2. Install dependencies
pip install flask python-dotenv werkzeug Pillow numpy opencv-python-headless pytest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install realesrgan gfpgan facexlib --no-deps
pip install scipy tqdm requests

# 3. If basicsr fails to install (known issue with Python 3.12+),
#    run the shim creation script:
python create_basicsr_shim.py
python fix_stylegan_shim.py

# 4. Download model weights (~400 MB total)
python scripts/setup_models.py
# Or they will auto-download on first restore request
```

## Running the Application

```bash
python app.py
```

The server starts at: **http://127.0.0.1:5000**

## API Documentation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Homepage |
| `POST` | `/api/restore` | Upload and restore image (multipart: field `image`) |
| `GET` | `/api/result/<id>` | View restored image |
| `GET` | `/api/download/<id>` | Download restored image |
| `GET` | `/api/health` | Health check |

### Example: Restore an image

```bash
curl -X POST -F "image=@photo.jpg" http://127.0.0.1:5000/api/restore
```

Response:
```json
{
  "success": true,
  "output_id": "abc123-...",
  "message": "Image restored successfully",
  "details": {
    "has_faces": false,
    "device": "cpu",
    "processing_time": 45.2
  }
}
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run just validation/preprocessing tests
python -m pytest tests/test_validation.py -v

# Run API tests
python -m pytest tests/test_api.py -v
```

## CPU / GPU Requirements

- **CPU**: Works on any modern CPU. Processing a 200×300 image takes ~30-120 seconds.
- **GPU**: CUDA-capable NVIDIA GPU recommended for faster processing. Automatically uses GPU when available with half-precision.
- **RAM**: ~2 GB minimum for model loading
- **Disk**: ~500 MB for model weights

## Configuration

Environment variables (`.env` file):

```
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=change-me-in-production
```

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `basicsr` fails to install | Run `python create_basicsr_shim.py` and `python fix_stylegan_shim.py` |
| `CascadeClassifier` not found | Expected with OpenCV 5.0 — the app uses a fallback face detector |
| Slow restoration | Normal on CPU — GPU with CUDA is 10-50x faster |
| Out of memory | The app automatically reduces tile size on OOM |
| Model download fails | Download weights manually and place in `weights/` directory |

## Limitations

- CPU inference is slow (30-120+ seconds per image depending on size)
- Maximum input resolution: 4096×4096 (larger images are downscaled)
- Minimum input resolution: 16×16
- GFPGAN face restoration produces plausible reconstructions, not verified truth
- No batch processing (one image at a time)
- The `basicsr` package cannot be installed normally on Python 3.12+ due to a build bug — a compatibility shim is provided

## Security Considerations

- Filenames are sanitized with UUID prefixes
- Path traversal is blocked at the Flask routing level
- Upload size limited to 20 MB
- Uploaded files are deleted after processing
- No user-uploaded content is executed
- Secret key loaded from environment variable

## Future Improvements

- Batch processing support
- Progress streaming via WebSocket/SSE
- Additional restoration models (colorization, inpainting)
- Docker containerization
- Result caching
- User accounts and history

