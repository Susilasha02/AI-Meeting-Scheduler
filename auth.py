# auth.py
import os, pathlib
from typing import Dict
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

CLIENT_FILE = pathlib.Path("google_oauth_client.json")
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]

def build_flow(app_base_url: str) -> Flow:
    flow = Flow.from_client_secrets_file(
        str(CLIENT_FILE),
        scopes=SCOPES,
        redirect_uri=app_base_url + "/auth/callback",
    )
    flow.params.update({
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
    })
    return flow

def creds_to_dict(creds: Credentials) -> Dict:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
    }

def creds_from_dict(d: Dict) -> Credentials:
    return Credentials(**d)
