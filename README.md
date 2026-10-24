# NXT AI — Own-model AI workspace

An experimental, independent AI project. No external AI inference APIs. The frontend is static and can run on GitHub Pages; the Python backend runs separately on a cloud service.

## Current capabilities

- **Chat:** a tiny byte-level autoregressive Transformer trained from random initialization with PyTorch. **Not trained yet**; answers require a checkpoint and will be low quality until meaningful training.
- **Image:** deterministic procedural SVG abstract art, **not** a trained image model.
- **Video:** procedural animated SVG, **not** AI-generated MP4.
- **UI:** responsive dark dashboard with Chat, Image Studio and Video Studio tabs.

## Train your own text model

Requires Python 3.11+ and enough local RAM. In a terminal:

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp data.example.txt data.txt
python train.py --data data.txt --steps 300 --out checkpoint.pt
```

On Windows without `cp`, copy `data.example.txt` to `data.txt` using File Explorer. Replace it with a much larger, licensed text dataset. The included sample is for demonstrating the training pipeline only. Do not commit datasets you do not have rights to use.

Start the backend from `backend/`:

```bash
uvicorn server:app --reload --port 8000
```

Open `http://localhost:8000/health` to check model status. Serve the website with `python -m http.server 5500` from the repository root. Open `http://localhost:5500` and set the backend URL to `http://localhost:8000`.

## Cloud backend

The included `backend/Dockerfile` can be deployed as a Docker service to Render, Railway, Fly.io or another cloud container provider. Choose the repository and set **Dockerfile path** to `backend/Dockerfile` and **Docker build context** to `backend`. Provider-specific options and free quotas vary. This is CPU-only and not suitable for a large model.

The trained checkpoint is not included. Put a checkpoint at `backend/checkpoint.pt` before building a private Docker image, or mount one via persistent storage and set `MODEL_PATH` to its location. **Do not commit model weights or private datasets to a public repository.** Set `ALLOWED_ORIGINS` to the exact deployed website origin, e.g. `https://USERNAME.github.io`. Never use `*` on a public backend.

**Security:** The sample backend has no authentication, per-user quotas, or request rate limiting. Keep the deployment private while testing. Add authentication, rate limits, usage limits, and abuse protection before exposing it to the public internet.

## GitHub Pages

In repository **Settings → Pages**, choose **Deploy from a branch**, branch **main**, folder **/(root)**, then Save. Your site should be available at `https://anirudhadebbarma031214-commits.github.io/NXT-AI/` after Pages finishes deploying. GitHub Pages only hosts the website, not Python inference.

In the dashboard, paste the HTTPS URL of your cloud backend into **Inference server** and click **Connect**. The URL is stored in localStorage; no API secret is used by the frontend.

## Limitations

A tiny from-scratch model trained on a short text sample will not provide ChatGPT-like answers. Training competitive text/image/video models requires large licensed datasets and significant compute. The backend's image/video routes are deliberately labeled procedural, so they are not misrepresented as neural generation.
