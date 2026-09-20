# Needle 3 Demo: Natural Language Text-to-Form Filling

Working full-stack prototype for **zero-shot unstructured text to structured form filling** using the **Needle 3 Inference Engine**, featuring the **Artify Bharat Artisan Onboarding Schema**.

![Needle 3 Demo](https://img.shields.io/badge/Model-Needle%203-orange?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Artify%20Bharat-emerald?style=flat-square)
![Backend](https://img.shields.io/badge/Backend-FastAPI%20%7C%20uv-blue?style=flat-square)
![Frontend](https://img.shields.io/badge/Frontend-React%2019%20%7C%20Tailwind-purple?style=flat-square)

---

## 🚀 Key Features

1. **Local Needle 3 Inference Pipeline**:
   - Parses complex multilingual dialect narratives (Bhojpuri, Maithili, Kannada, Gujarati, Hindi) into validated Pydantic models.
   - Extracts:
     - Artisan Name, Phone, 4-digit security PIN.
     - State, District, and specific Craft Cluster.
     - Government Identifiers: Pehchan Card ID (`PEH-...`) and TRIFED Registration ID (`TRIFED-...`).
     - Craft category, GI Tag name, GI certification status, and years of experience.
     - Auto-generated artisan craft summary.

2. **Interactive Dual-Pane UI**:
   - **Left Pane**: Unstructured voice/narrative prompt input with one-click presets for regional artisans (Varanasi Silk Weaver, Channapatna Toy Crafter, Madhubani Artist, Kutch Embroiderer).
   - **Right Pane**: Real-time form auto-population with per-field confidence metrics, interactive JSON preview, entity inspection, and final submission to the National Crafts Registry.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.12, FastAPI, Pydantic v2, `uv`, pytest.
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons.

---

## 📦 Quickstart

### 1. Start the Backend

```bash
cd backend
# Create virtual environment and install dependencies with uv
uv venv
source .venv/bin/activate
uv pip install -e .

# Run FastAPI server
uvicorn app.main:app --reload --port 8000
```

FastAPI Swagger Documentation: `http://localhost:8000/docs`

### 2. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the web app at `http://localhost:5173`.

---

## 🧪 Testing

Run backend unit and integration test suite:

```bash
cd backend
pytest
```

---

## 📂 Project Structure

```
needle-demo/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI endpoints & routes
│   │   ├── needle_engine.py   # Needle 3 inference & extraction engine
│   │   ├── sample_data.py     # Multilingual artisan presets
│   │   └── schemas.py         # Pydantic schemas for Artify Bharat
│   ├── tests/
│   │   ├── test_api.py        # API endpoint tests
│   │   └── test_needle_engine.py # Inference engine tests
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main dual-pane text-to-form UI
│   │   ├── types.ts           # TypeScript interfaces
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.ts
└── README.md
```
