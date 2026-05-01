# Plant Disease Classification

A full-stack AI application that classifies plant leaf diseases from images. Upload a photo of a plant leaf and get an instant diagnosis with a Grad-CAM heatmap showing which part of the leaf the model focused on.

---

## Features

- **Disease detection** — classifies 8 categories across pepper, potato, and tomato leaves
- **Grad-CAM visualization** — heatmap overlay shows the model's reasoning
- **Top-3 predictions** — ranked results with confidence scores
- **Prediction history** — browse all past analyses
- **User feedback** — mark predictions as correct or incorrect to track model accuracy
- **JWT authentication** — all API endpoints are protected

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML model | TensorFlow 2.21 · EfficientNetB0 (transfer learning) |
| Backend | FastAPI · SQLAlchemy · PostgreSQL · Alembic |
| Auth | JWT (python-jose) · bcrypt (passlib) |
| Frontend | Next.js 16 · React 19 · Tailwind CSS v4 |

---

## Project Structure

```
plant-disease-classification/
├── src/                        # Model training scripts
│   ├── train_transfer.py       # EfficientNetB0 transfer learning
│   ├── train.py                # CNN training from scratch
│   ├── predict.py              # Inference utilities
│   └── utils.py                # Data helpers
├── outputs/
│   └── checkpoints/
│       └── efficientnet_transfer.keras   # Trained model weights
├── backend/
│   ├── main.py                 # FastAPI app entry point
│   ├── config.py               # Central config (model path, JWT, DB URL)
│   ├── create_test_user.py     # One-off script to seed a user
│   ├── routes/
│   │   ├── auth.py             # POST /login
│   │   ├── predict.py          # POST /predict
│   │   ├── history.py          # GET /predictions/history, GET /predictions/{id}/image
│   │   └── feedback.py         # PATCH /predictions/{id}/feedback
│   ├── services/
│   │   ├── model_service.py    # Model loading and inference
│   │   ├── gradcam_service.py  # Grad-CAM heatmap generation
│   │   ├── auth_service.py     # Credential validation, JWT issuance
│   │   ├── user_service.py     # User CRUD
│   │   └── prediction_service.py  # Prediction record CRUD, image storage
│   ├── db/
│   │   ├── models/             # SQLAlchemy ORM models (User, PredictionRecord)
│   │   └── session.py          # DB engine and session factory
│   ├── alembic/                # Database migrations
│   └── schemas/                # Pydantic request/response schemas
└── frontend/
    ├── app/
    │   ├── login/page.tsx      # Login page
    │   ├── predict/page.tsx    # Image upload and results
    │   └── history/page.tsx    # Prediction history and feedback
    ├── components/
    │   └── Navbar.tsx          # Navigation bar
    └── lib/
        └── api.ts              # API client (fetch wrappers, token management)
```

---

## Supported Classes

| Plant | Condition |
|---|---|
| Pepper (Bell) | Bacterial Spot |
| Pepper (Bell) | Healthy |
| Potato | Early Blight |
| Potato | Late Blight |
| Potato | Healthy |
| Tomato | Early Blight |
| Tomato | Late Blight |
| Tomato | Healthy |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL

### 1. Set up the database

```bash
createdb plant_disease
```

### 2. Configure the backend

```bash
cd backend
cp .env .env.local   # or edit .env directly
```

Edit `backend/.env`:
```env
DATABASE_URL=postgresql://<user>@localhost:5432/plant_disease
JWT_SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
```

### 3. Install backend dependencies and run migrations

```bash
cd backend
pip install -r requirements.txt
alembic upgrade head
```

### 4. Create a user

```bash
python create_test_user.py
# Creates testuser / testpass123 by default
# Use --username and --password flags for custom credentials
```

### 5. Start the backend

```bash
uvicorn main:app --reload
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 6. Install frontend dependencies and start the dev server

```bash
cd frontend
npm install
npm run dev
# Frontend available at http://localhost:3000
```

---

## API Endpoints

All endpoints except `/login` require `Authorization: Bearer <token>` header.

| Method | Path | Description |
|---|---|---|
| `POST` | `/login` | Authenticate and receive a JWT |
| `POST` | `/predict` | Upload a leaf image, get diagnosis + Grad-CAM |
| `GET` | `/predictions/history` | List all predictions for the current user |
| `GET` | `/predictions/{id}/image` | Retrieve the original uploaded image |
| `PATCH` | `/predictions/{id}/feedback` | Submit correct/incorrect feedback |
| `GET` | `/health` | Health check |

---

## Model Training

Training scripts are in `src/`. The deployed model uses EfficientNetB0 with transfer learning from ImageNet weights.

```bash
# Install training dependencies
pip install -r requirements.txt

# Train the transfer learning model
python src/train_transfer.py

# Visualize Grad-CAM on a test image
python gradcam_visualize.py

# Run robustness tests
python robustness_test.py
```

The trained model is saved to `outputs/checkpoints/efficientnet_transfer.keras`.
