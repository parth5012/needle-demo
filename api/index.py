import sys
import os

# Add backend directory to path
backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, backend_path)

from app.main import app

# Vercel ASGI Handler
handler = app
