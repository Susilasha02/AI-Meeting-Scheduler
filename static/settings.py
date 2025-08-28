# settings.py
import os

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret")  # replace in prod
REDIRECT_PATH = "/oauth/callback"
REDIRECT_URI = f"{BASE_URL}{REDIRECT_PATH}"
