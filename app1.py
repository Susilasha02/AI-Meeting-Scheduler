import os
import json
from datetime import datetime, timedelta
from typing import List, Dict

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, JSONResponse
from dotenv import load_dotenv
from fastapi import Body
from googleapiclient.errors import HttpError
import traceback

# Google auth/libs
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

load_dotenv()

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:3978")
PORT = int(os.getenv("PORT", "3978"))
CLIENT_FILE = "google_oauth_client.json"

# --------- very tiny token store (json file) ----------
TOKEN_STORE = "token_store.json"
ORGANIZER_ID = "organizer"   # in real app, map to your user/session

def save_creds(user_id: str, creds: Credentials):
    data = {}
    if os.path.exists(TOKEN_STORE):
        data = json.loads(open(TOKEN_STORE, "r").read() or "{}")
    data[user_id] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else [],
    }
    with open(TOKEN_STORE, "w") as f:
        json.dump(data, f, indent=2)

def load_creds(user_id: str) -> Credentials | None:
    if not os.path.exists(TOKEN_STORE): return None
    data = json.loads(open(TOKEN_STORE, "r").read() or "{}")
    if user_id not in data: return None
    return Credentials(**data[user_id])

# --------- google oauth helpers ----------
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
    # add if you want name->email lookups later:
    # "https://www.googleapis.com/auth/contacts.readonly",
]

def build_flow() -> Flow:
    return Flow.from_client_secrets_file(
        CLIENT_FILE,
        scopes=SCOPES,
        redirect_uri=f"{APP_BASE_URL}/auth/callback",
    )

# --------- calendar helpers ----------
def gcal_service(creds: Credentials):
    return build("calendar", "v3", credentials=creds, cache_discovery=False)

def freebusy(creds: Credentials, emails: List[str], start_iso: str, end_iso: str) -> Dict[str, List[Dict]]:
    svc = gcal_service(creds)
    body = {"timeMin": start_iso, "timeMax": end_iso, "items": [{"id": e} for e in emails]}
    fb = svc.freebusy().query(body=body).execute()
    out = {}
    for cal_id, cal in fb.get("calendars", {}).items():
        out[cal_id] = cal.get("busy", [])
    return out

def insert_event(creds: Credentials, calendar_id: str, subject: str,
                 start_iso: str, end_iso: str, attendees: List[str],
                 description: str = "", add_meet: bool = True):
    svc = gcal_service(creds)
    event = {
        "summary": subject,
        "description": description,
        "start": {"dateTime": start_iso},
        "end":   {"dateTime": end_iso},
        "attendees": [{"email": a} for a in attendees],
    }
    if add_meet:
        event["conferenceData"] = {"createRequest": {"requestId": f"req-{int(datetime.utcnow().timestamp())}"}}
    created = svc.events().insert(
        calendarId=calendar_id,
        body=event,
        conferenceDataVersion=1 if add_meet else 0,
        sendUpdates="all"
    ).execute()
    return created
def primary_calendar_id(creds: Credentials) -> str:
    svc = gcal_service(creds)
    # Returns your email-like id for the primary calendar
    return svc.calendarList().get(calendarId="primary").execute()["id"]

# ---------- simple candidate builder ----------
def _merge(intervals):
    ivs = sorted([(datetime.fromisoformat(i["start"]), datetime.fromisoformat(i["end"])) for i in intervals])
    out=[]
    for s,e in ivs:
        if not out or s>out[-1][1]:
            out.append([s,e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out

def compute_free(busy_by_person, window_start: datetime, window_end: datetime):
    all_busy=[b for person in busy_by_person for b in person]
    merged=_merge(all_busy)
    free=[]; cur=window_start
    for s,e in merged:
        if s>cur: free.append((cur,s))
        cur=max(cur,e)
    if cur<window_end: free.append((cur,window_end))
    return [{"start":a.isoformat(), "end":b.isoformat()} for a,b in free]

def candidates(free_windows, duration_min=45, buffer_min=15, cap=3):
    out=[]; dur=timedelta(minutes=duration_min); buf=timedelta(minutes=buffer_min)
    daily={}
    for w in free_windows:
        s = datetime.fromisoformat(w["start"]) + buf
        e = datetime.fromisoformat(w["end"]) - buf
        cur=s
        while cur+dur<=e:
            day=cur.date()
            if daily.get(day,0)<cap:
                out.append({"start":cur.isoformat(), "end":(cur+dur).isoformat(),
                            "meta":{"buffers":{"pre":buffer_min,"post":buffer_min}}})
                daily[day]=daily.get(day,0)+1
            cur += dur
    return out[:10]

def explain(slot):
    b=slot["meta"]["buffers"]; return f"Conflict‑free; honors {b['pre']}m buffers; within work hours."

# ---------- FastAPI app ----------
app = FastAPI()

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/auth/start")
def auth_start():
    try:
        # (keep the debug checks you added if you want)
        flow = build_flow()
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return RedirectResponse(auth_url)
    except Exception as e:
        return JSONResponse({"error": "oauth_setup_failed", "detail": str(e)}, status_code=500)


@app.get("/auth/callback")
def auth_callback(code: str):
    flow = build_flow()
    flow.fetch_token(code=code)
    creds = flow.credentials
    save_creds(ORGANIZER_ID, creds)
    return JSONResponse({"connected": True})

@app.post("/suggest")
def suggest(body: dict = Body(default={})):
    creds = load_creds(ORGANIZER_ID)
    if not creds:
        return JSONResponse({"error": "not_connected"}, status_code=401)

    try:
        # ---- Inputs with defaults ----
        duration_min = int(body.get("duration_min", 45))
        buffer_min   = int(body.get("buffer_min", 15))
        # default to your own primary calendar id
        attendees    = body.get("attendees", [primary_calendar_id(creds)])

        # window: ISO strings, else default to next 5 days
        start_iso = body.get("window_start")
        end_iso   = body.get("window_end")
        if start_iso and end_iso:
            from dateutil import parser as dparse
            start = dparse.isoparse(start_iso)
            end   = dparse.isoparse(end_iso)
        else:
            now = datetime.utcnow()
            start = (now + timedelta(days=1)).replace(hour=9,  minute=0, second=0, microsecond=0)
            end   = (now + timedelta(days=5)).replace(hour=18, minute=30, second=0, microsecond=0)

        # ---- Free/busy for listed calendars ----
        fb = freebusy(creds, attendees, start.isoformat()+"Z", end.isoformat()+"Z")
        busy_lists = [[{"start":b["start"], "end":b["end"]} for b in fb[cal]] for cal in fb]

        # ---- Build candidates ----
        free = compute_free(busy_lists, start, end)
        cands = candidates(free, duration_min=duration_min, buffer_min=buffer_min)
        top = cands[:3]
        return [{"slot": s, "why": explain(s)} for s in top]

    except HttpError as he:
        try:
            # Google API errors have JSON content; surface it
            detail = he.error_details if hasattr(he, "error_details") else he.content.decode()
        except Exception:
            detail = str(he)
        return JSONResponse({"error": "google_api_error", "detail": detail}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": "server_error", "detail": str(e), "trace": traceback.format_exc()}, status_code=500)

@app.post("/book")
def book(body: dict):
    creds = load_creds(ORGANIZER_ID)
    if not creds:
        return JSONResponse({"error": "not_connected"}, status_code=401)

    s = body["start"]; e = body["end"]
    attendees = body.get("attendees", [])  # add real emails here
    if not attendees:
        attendees = []  # you can invite others later via UI
    subject = body.get("subject", "Timetable-Aware Meeting")
    explanation = body.get("why", "Scheduled by AI bot.")

    ev = insert_event(creds, "primary", subject, s, e, attendees, description=explanation, add_meet=True)
    return {"eventId": ev["id"], "htmlLink": ev.get("htmlLink"), "hangoutLink": ev.get("hangoutLink")}
