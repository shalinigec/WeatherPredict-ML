# 🌦️ Aetheria - AI Weather Prediction & Rainfall Forecast System

A modern end-to-end Machine Learning web application designed to forecast rainfall based on atmospheric telemetry (Temperature, Humidity, Wind Speed, Cloud Cover, and Surface Pressure). 

The system leverages three supervised ML models (**Random Forest**, **Decision Tree**, and **XGBoost**) served via a high-performance **FastAPI** backend and paired with a responsive **React + Vite** frontend.

---

## 🏗️ Project Architecture

```
Weather-Project-ML/
├── backend/                  # FastAPI & ML Model Server
│   ├── models/               # Serialized models (.joblib) & metadata
│   ├── main.py               # API endpoints & inference engine
│   ├── train.py              # ML training and evaluation pipeline
│   ├── requirements.txt      # Python dependencies
│   └── weather_forecast_data.csv
│
├── frontend/                 # React (Vite / TanStack) Web Interface
│   ├── src/                  # Components, routes, and styling
│   ├── public/               # Static assets
│   ├── package.json          # Node dependencies & build scripts
│   └── .env                  # Environment configuration
│
└── .gitignore
```

---

## 🚀 Live Deployment Guide

This project is structured for easy independent deployment:
- **Backend (API + ML Models)**: Deployed as a Web Service on **[Render](https://render.com/)**
- **Frontend (UI)**: Deployed as a Single Page / Web Application on **[Vercel](https://vercel.com/)**

---

### Step 1: Deploy Backend on Render

1. **Sign in to Render**: Go to [render.com](https://render.com/) and connect your GitHub account.
2. Click **New +** > **Web Service**.
3. Select your GitHub repository.
4. Configure the service with the following settings:
   - **Name**: `weather-ml-backend` (or any name you prefer)
   - **Region**: Choose the region closest to you or your target audience (e.g., *Frankfurt* or *Oregon*)
   - **Branch**: `main`
   - **Root Directory**: `backend` *(⚠️ Critical: ensure this is set to `backend`)*
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: `Free`
5. Click **Deploy Web Service**.
6. Wait 2–3 minutes for the build to finish. Once deployed, Render will provide a public URL like:
   ```
   https://weather-ml-backend.onrender.com
   ```
7. Verify your deployment by testing the health check endpoint in your browser:
   ```
   https://weather-ml-backend.onrender.com/health
   ```
   *(You should receive: `{"status": "healthy", "models_loaded": true, ...}`)*

---

### Step 2: Deploy Frontend on Vercel

1. **Sign in to Vercel**: Go to [vercel.com](https://vercel.com/) and connect your GitHub account.
2. Click **Add New...** > **Project**.
3. Import your GitHub repository.
4. Under **Configure Project**:
   - **Project Name**: `aetheria-weather-ai` (or your preferred name)
   - **Framework Preset**: `Vite`
   - **Root Directory**: Click **Edit** and select `frontend` *(⚠️ Critical: ensure root directory points to `frontend`)*
   - **Build Command**: `npm run build` (or default)
   - **Output Directory**: `dist` (or `.output/public` if using TanStack Start server build)
5. Expand the **Environment Variables** section and add:
   - **Key**: `VITE_WEATHER_API_URL`
   - **Value**: Your Render backend URL from Step 1 (e.g. `https://weather-ml-backend.onrender.com` without a trailing slash)
6. Click **Deploy**.
7. Once deployed, Vercel will provide your production URL (e.g. `https://aetheria-weather-ai.vercel.app`). Open it to test real-time predictions!

---

## 💻 Local Development Setup

To run both services locally on your machine:

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ & npm (or Bun)

---

### 2. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux:
# python3 -m venv .venv
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
- API will be accessible at: `http://127.0.0.1:8000`
- Interactive Swagger API documentation: `http://127.0.0.1:8000/docs`

---

### 3. Frontend Setup

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Verify your .env file has the local backend URL:
# VITE_WEATHER_API_URL=http://127.0.0.1:8000

# Run local development server
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check service status & verification of loaded models |
| `POST` | `/predict` | Predict rain using default Random Forest model |
| `POST` | `/predict/model` | Predict rain using selected model (`Random Forest`, `Decision Tree`, or `XGBoost`) |

#### Sample Request Body (`POST /predict/model`):
```json
{
  "Temperature": 25.4,
  "Humidity": 78.2,
  "Wind_Speed": 14.5,
  "Cloud_Cover": 65.0,
  "Pressure": 995.2,
  "model": "XGBoost"
}
```

---

## 🧠 Machine Learning Details

- **Input Features**:
  - `Temperature` (°C)
  - `Humidity` (%)
  - `Wind_Speed` (km/h)
  - `Cloud_Cover` (%)
  - `Pressure` (hPa)
- **Target**: `Rain_Tomorrow` (0: No Rain, 1: Rain)
- **Trained Models**:
  - Random Forest Classifier (`random_forest.joblib`)
  - Decision Tree Classifier (`decision_tree.joblib`)
  - XGBoost Classifier (`xgboost.joblib`)

To retrain the models:
```bash
cd backend
python train.py
```

---

## 🛡️ Git Setup & Pushing to GitHub

If you are initializing a fresh Git repository for this project:

```bash
# In the root project directory:
git init
git add .
git commit -m "feat: initial commit with backend, frontend, and deployment config"

# Link to your GitHub repository
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
