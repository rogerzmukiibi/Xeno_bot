# Docker Build Instructions 

## Dockerfile Optimizations

```dockerfile
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt
```

## Build Progress

The Docker build is currently in progress. Here's what it's doing:

1. ✅ Loading base Python 3.13-slim image
2. ✅ Setting working directory and environment variables
3. ⏳ Installing system dependencies (build-essential, curl)
4. ⏳ Installing Python dependencies (this may take 10-15 minutes due to large packages like PyTorch ~900MB)
5. Copying project files
6. Creating necessary directories
7. Exposing port 7860
8. Setting health check

## How to Use

### Start the Docker Container

Once the build completes, start with:

```bash
docker-compose up -d
```

### Access the Application

- Open browser to `http://localhost:7860`

### View Logs

```bash
docker-compose logs -f xeno-bot
```

### Stop the Container

```bash
docker-compose down
```

### Build Directly Without docker-compose

```bash
docker build -t xeno-bot:latest .
docker run -p 7860:7860 \
  -e GEMINI_API_KEY="your-api-key" \
  -e GOOGLE_SHEETS_CREDENTIALS='{"type": "service_account", ...}' \
  xeno-bot:latest
```

## Environment Variables Required

The container needs these environment variables set in `.env`:

```
GEMINI_API_KEY=your-google-gemini-api-key
GOOGLE_SHEETS_CREDENTIALS={"type": "service_account", "project_id": "...", ...}
```

See `.env.example` for template.

## Performance Notes

- **Initial build time**: 10-20 minutes (downloading and installing ~900MB PyTorch library)
- **Subsequent builds**: Faster due to Docker layer caching
- **Container startup**: ~30-60 seconds for first run, ~5-10 seconds after that
- **Memory requirement**: 2GB minimum recommended (PyTorch + Gradio + ChromaDB)

## Troubleshooting

If the build still times out:
1. Increase `PIP_DEFAULT_TIMEOUT` further in Dockerfile
2. Check your network connection
3. Try building again (Docker will use cached layers)
4. Consider using a build cache: `DOCKER_BUILDKIT=1 docker build ...`

## Files Modified

- `Dockerfile` - Optimized for reliability
- `docker-compose.yml` - Removed obsolete version attribute
- `app.py` - Added missing `import os`
- `.dockerignore` - Excludes unnecessary files from build context
- `.env.example` - Template for environment variables
