# 🚀 HalluGuard Deployment Guide (100% Free & Open Source)

This project is configured and ready to deploy to free cloud hosting platforms. All necessary configuration files (`Procfile`, `render.yaml`, `Dockerfile`, `vercel.json`, `wsgi.py`, and `requirements.txt`) are included in this repository.

---

## 🌟 Method 1: Deploy on Render.com (Recommended for College Presentations)

Render offers a **100% Free Web Service tier** that gives you a public HTTPS URL (e.g., `https://halluguard.onrender.com`) that can be accessed from any phone, laptop, or faculty review portal.

### Step 1: Push your Code to GitHub
1. Create a new repository on [GitHub](https://github.com/new) named `halluguard`.
2. In your terminal, run:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/halluguard.git
   git push -u origin main
   ```

### Step 2: Deploy on Render
1. Go to [render.com](https://dashboard.render.com) and sign in with GitHub (Free).
2. Click **New +** $\rightarrow$ **Web Service**.
3. Select your `halluguard` repository.
4. Render will auto-fill from `render.yaml`, or you can verify:
   - **Name:** `halluguard`
   - **Environment:** `Python 3`
   - **Region:** Any (e.g., Singapore or Frankfurt)
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn wsgi:app`
   - **Instance Type:** `Free`
5. Click **Create Web Service**.
6. Wait 1–2 minutes for the build to finish. Your app will be live at:
   👉 **`https://halluguard.onrender.com`**

---

## 🤗 Method 2: Deploy on Hugging Face Spaces (Ideal for GenAI Projects)

Hugging Face Spaces is 100% free forever for CPU-basic instances and gives you an official AI project badge and permanent URL.

### Step 1: Create a Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and log in.
2. Click **Create new Space**.
3. Fill in:
   - **Space name:** `halluguard`
   - **License:** `mit` or `openrail`
   - **Space SDK:** Choose **Docker** (Blank).
   - **Space hardware:** `Free (CPU basic - 2 vCPU, 16GB RAM)`.
4. Click **Create Space**.

### Step 2: Push your Project
Run in your project directory:
```bash
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/halluguard
git push space main
```
Hugging Face will automatically build your included `Dockerfile` and launch the app at:
👉 **`https://huggingface.co/spaces/<YOUR_HF_USERNAME>/halluguard`**

---

## ⚡ Method 3: Instant Live URL for Live Classroom Presentation (0 Seconds Setup)

If you need a live public HTTPS URL **right now** to show your faculty on their mobile or laptop without waiting for cloud builds:

1. Start your project locally:
   ```powershell
   python run.py
   ```
   *(Running on port 5000)*
2. In another terminal window, run either:
   - **Using Localtunnel (Free, no account needed):**
     ```powershell
     npx localtunnel --port 5000
     ```
   - **Or using Cloudflare Tunnel (Free, no account needed):**
     ```powershell
     cloudflared tunnel --url http://localhost:5000
     ```
3. Share the generated public link (`https://....loca.lt` or `https://....trycloudflare.com`) with your faculty!

---

## 🐳 Method 4: Run Locally in Docker

If you or your faculty want to run with Docker:
```bash
docker build -t halluguard .
docker run -p 5000:7860 -e PORT=7860 halluguard
```
Open [http://localhost:5000](http://localhost:5000).

---

## 📋 Summary of Included Deployment Files

| File | Purpose |
|---|---|
| **`Procfile`** | Configures Gunicorn web process for Render, Heroku, Railway |
| **`render.yaml`** | 1-Click Infrastructure-as-Code for Render.com |
| **`Dockerfile`** | Containerized build for Hugging Face Spaces, Koyeb, Docker |
| **`vercel.json`** | Serverless Python build config for Vercel |
| **`wsgi.py`** | Production WSGI entry point binding to `$PORT` |
| **`requirements.txt`** | Includes `gunicorn` and `waitress` for Linux/Windows production servers |
