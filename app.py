# app.py (updated)
import os
import json
import re
import traceback
import uuid
import zoneinfo
import urllib.parse
import logging
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Body, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from dotenv import load_dotenv

# Google auth/libs
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.errors import HttpError
from google.auth.transport.requests import AuthorizedSession

from dateutil import parser as dparse
import dateparser  # pip install dateparser


load_dotenv()

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:3978").rstrip("/")
PORT = int(os.getenv("PORT", "3978"))
CLIENT_FILE = os.getenv("CLIENT_FILE", "/etc/secrets/google_oauth_client.json")

LOCAL_TZ = os.getenv("TIME_ZONE", "UTC")

# --------- stores ----------
TOKEN_STORE = "token_store.json"
CONTACTS_STORE = "contacts.json"
ORGANIZER_ID = "organizer"

# Data dirs
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOADS = DATA_DIR / "uploads"
MOMS = DATA_DIR / "moms"
UPLOADS.mkdir(parents=True, exist_ok=True)
MOMS.mkdir(parents=True, exist_ok=True)

# set up a small logger for recorder redirect debugging
logger = logging.getLogger("recorder_redirect")
if not logger.handlers:
    # avoid duplicate handlers on reload
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# ---------- tz helpers ----------
def to_local(dt: datetime) -> datetime:
    tz = zoneinfo.ZoneInfo(LOCAL_TZ)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
    return dt.astimezone(tz)

def pretty(dt: datetime) -> str:
    l = to_local(dt)
    return l.strftime("%a, %d %b %Y %I:%M %p") + f" ({LOCAL_TZ})"

def rfc3339(dt: datetime) -> str:
    """Return RFC3339 with Z if UTC. Accepts naive or tz-aware."""
    if dt.tzinfo is None:
        return dt.isoformat() + "Z"
    return dt.astimezone(zoneinfo.ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")

# ---------- token store ----------
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

# ---------- contacts store (name -> email) ----------
def load_contacts() -> Dict[str, str]:
    if not os.path.exists(CONTACTS_STORE): return {}
    try:
        return json.loads(open(CONTACTS_STORE, "r").read() or "{}")
    except Exception:
        return {}

def save_contacts(d: Dict[str, str]):
    with open(CONTACTS_STORE, "w") as f:
        json.dump(d, f, indent=2)

# --------- google oauth helpers ----------
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.readonly",
]

def build_flow() -> Flow:
    return Flow.from_client_secrets_file(
        CLIENT_FILE,
        scopes=SCOPES,
        redirect_uri=f"{APP_BASE_URL}/auth/callback",
    )

# --------- calendar helpers (Requests transport) ----------
def _authed(creds: Credentials) -> AuthorizedSession:
    return AuthorizedSession(creds)

def freebusy(creds: Credentials, emails: List[str], start_iso: str, end_iso: str) -> Dict[str, List[Dict]]:
    sess = _authed(creds)
    body = {"timeMin": start_iso, "timeMax": end_iso, "items": [{"id": e} for e in emails]}
    r = sess.post("https://www.googleapis.com/calendar/v3/freeBusy", json=body, timeout=20)
    try:
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"freeBusy failed: {r.status_code} {r.text}") from e
    data = r.json()
    out = {}
    for cal_id, cal in data.get("calendars", {}).items():
        out[cal_id] = cal.get("busy", [])
    return out

def insert_event(creds: Credentials, calendar_id: str, subject: str,
                 start_iso: str, end_iso: str, attendees: List[str],
                 description: str = "", add_meet: bool = True, time_zone: str | None = None):
    sess = _authed(creds)
    event = {
        "summary": subject,
        "description": description,
        "start": {"dateTime": start_iso},
        "end":   {"dateTime": end_iso},
        "attendees": [{"email": a} for a in attendees],
    }
    if time_zone:
        event["start"]["timeZone"] = time_zone
        event["end"]["timeZone"] = time_zone

    params = {"sendUpdates": "all"}
    if add_meet:
        event["conferenceData"] = {"createRequest": {"requestId": f"req-{int(datetime.utcnow().timestamp())}"}}
        params["conferenceDataVersion"] = "1"

    r = sess.post(
        f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events",
        params=params,
        json=event,
        timeout=20
    )
    r.raise_for_status()
    return r.json()

def patch_event_description(creds: Credentials, calendar_id: str, event_id: str, new_description: str):
    sess = _authed(creds)
    r = sess.patch(
        f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events/{event_id}",
        json={"description": new_description},
        timeout=20
    )
    r.raise_for_status()
    return r.json()

# ---------- candidate builder + explanations ----------
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
    b=slot["meta"]["buffers"]
    s=datetime.fromisoformat(slot["start"]); e=datetime.fromisoformat(slot["end"])
    return {
        "conflict_free": True,
        "buffers": {"pre_min": b["pre"], "post_min": b["post"]},
        "work_hours": "09:00–18:30 window",
        "readable": f"{pretty(s)} → {pretty(e)}",
    }

# ---------- fairness-aware scoring helpers ----------
def _adjacent_gaps_for_fb(fb: Dict[str, List[Dict]], slot_start: datetime, slot_end: datetime) -> Dict[str, Tuple[float, float]]:
    """For each calendar, compute nearest busy block before and after the slot (minutes)."""
    gaps = {}
    for cal_id, blocks in fb.items():
        before_gap = float("inf")
        after_gap = float("inf")
        for b in blocks:
            b_start = dparse.isoparse(b["start"])
            b_end   = dparse.isoparse(b["end"])
            if b_end <= slot_start:
                before_gap = min(before_gap, (slot_start - b_end).total_seconds()/60.0)
            if b_start >= slot_end:
                after_gap = min(after_gap, (b_start - slot_end).total_seconds()/60.0)
        gaps[cal_id] = (before_gap if before_gap != float("inf") else 24*60,
                        after_gap  if after_gap  != float("inf") else 24*60)
    return gaps

def _fairness_score(slot: Dict, fb: Dict[str, List[Dict]]) -> Tuple[float, Dict]:
    s = datetime.fromisoformat(slot["start"])
    e = datetime.fromisoformat(slot["end"])
    hour = s.hour + s.minute/60
    midday = 1 - abs(hour - 13) / 13  # 1 at 1pm

    gaps = _adjacent_gaps_for_fb(fb, s, e)  # minutes
    per_person_min_gap = [min(g) for g in gaps.values()] or [0]
    min_gap = min(per_person_min_gap)
    avg_gap = sum(per_person_min_gap) / max(1, len(per_person_min_gap))

    def norm_gap(x): return min(x, 120) / 120.0
    score = 0.5 * midday + 0.3 * norm_gap(min_gap) + 0.2 * norm_gap(avg_gap)
    return score, {
        "midday_closeness": round(midday, 3),
        "min_gap_minutes": round(min_gap, 1),
        "avg_gap_minutes": round(avg_gap, 1),
        "used_attendees": list(fb.keys())
    }

# ---------- very light NL parsing ----------
DUR_RE = re.compile(r"(\d+)\s*(min|mins|minutes|hour|hr|h)", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def _next_week_bounds() -> Tuple[datetime, datetime]:
    now = datetime.utcnow()
    monday = (now + timedelta(days=(7 - now.weekday()))).replace(hour=9, minute=0, second=0, microsecond=0)
    friday = (monday + timedelta(days=4)).replace(hour=18, minute=30, second=0, microsecond=0)
    return monday, friday

def parse_prompt(prompt: str, contacts: Dict[str,str]) -> Dict:
    p = prompt.strip()

    dur=45
    m = DUR_RE.search(p)
    if m:
        n=int(m.group(1)); unit=m.group(2).lower()
        dur = n*60 if unit.startswith('h') else n

    attendees = set(EMAIL_RE.findall(p))
    for w in re.findall(r"[A-Za-z]+", p):
        key = w.lower()
        if key in contacts:
            attendees.add(contacts[key])
    if not attendees:
        attendees = {"primary"}

    start=None; end=None
    p_low = p.lower()
    if "next week" in p_low:
        start, end = _next_week_bounds()
    elif "tomorrow" in p_low:
        base = datetime.utcnow() + timedelta(days=1)
        start = base.replace(hour=9, minute=0, second=0, microsecond=0)
        end   = base.replace(hour=18, minute=30, second=0, microsecond=0)
    else:
        dp = dateparser.parse(p, settings={"RETURN_AS_TIMEZONE_AWARE": False})
        if dp:
            start = dp.replace(hour=9, minute=0, second=0, microsecond=0)
            end   = dp.replace(hour=18, minute=30, second=0, microsecond=0)
    if not start or not end:
        now = datetime.utcnow()
        start = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        end   = (now + timedelta(days=5)).replace(hour=18, minute=30, second=0, microsecond=0)

    return {
        "attendees": sorted(attendees),
        "duration_min": dur,
        "buffer_min": 15,
        "window_start": rfc3339(start),
        "window_end":   rfc3339(end),
    }

# ---------- new: search window helper for 'today' mode ----------
def get_search_window_from_prompt(prompt: str, days_ahead: int = 7) -> Tuple[datetime, datetime]:
    """
    If prompt contains 'today' (case-insensitive), returns now..end_of_today in LOCAL_TZ.
    Otherwise returns now .. now + days_ahead.
    """
    tz = zoneinfo.ZoneInfo(LOCAL_TZ)
    now = datetime.now(tz)
    if "today" in (prompt or "").lower():
        start = now
        end = datetime(now.year, now.month, now.day, 23, 59, 59, tzinfo=tz)
    else:
        start = now
        end = now + timedelta(days=days_ahead)
    # convert to UTC-naive ISO for compatibility with freebusy (we use rfc3339 anyway)
    return start, end

# ---------- summarizer for MoM ----------
def summarize_transcript(transcript: str, max_sentences: int = 6):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    summary = ' '.join(sentences[:max_sentences]).strip()
    actions = []
    for s in sentences:
        s_strip = s.strip()
        if re.match(r'(?i)^(action|todo|task|follow up|follow-up|please|assign|will|should)\b', s_strip) or \
           any(k in s_strip.lower() for k in ['action:', 'todo', 'follow up', 'please', 'should', 'will']):
            actions.append(s_strip)
    actions = list(dict.fromkeys(actions))
    return {"summary": summary, "action_items": actions or []}

# ---------- FastAPI ----------
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS if needed (Teams iframe context)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static with Teams-friendly headers
from fastapi.responses import Response
@app.middleware("http")
async def add_teams_headers(request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Content-Security-Policy"] = "frame-ancestors 'self' https://teams.microsoft.com https://*.teams.microsoft.com"
    return response


app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/ui/")

@app.get("/healthz")
def health():
    return {"ok": True}

# OAuth
@app.get("/auth/start")
def auth_start(request: Request):
    try:
        next_url = request.query_params.get("next", "/ui/")
        flow = build_flow()
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=next_url,
        )
        return RedirectResponse(auth_url)
    except Exception as e:
        return JSONResponse({"error": "oauth_setup_failed", "detail": str(e)}, status_code=500)

@app.get("/auth/callback")
def auth_callback(code: str, state: str | None = None):
    flow = build_flow()
    flow.fetch_token(code=code)
    creds = flow.credentials
    save_creds(ORGANIZER_ID, creds)
    return RedirectResponse(state) if state else JSONResponse({"connected": True})

# Contacts
@app.post("/contacts/add")
def contacts_add(body: dict = Body(...)):
    name = body["name"].strip().lower()
    email = body["email"].strip()
    d = load_contacts()
    d[name] = email
    save_contacts(d)
    return {"saved": True, "count": len(d)}

@app.get("/contacts/list")
def contacts_list():
    return load_contacts()

# NL → suggest
@app.post("/nlp/suggest")
def nlp_suggest(body: dict = Body(...)):
    creds = load_creds(ORGANIZER_ID)
    if not creds: return JSONResponse({"error": "not_connected"}, status_code=401)
    prompt = body.get("prompt","").strip()
    cfg = parse_prompt(prompt, load_contacts())
    return suggest(cfg)

# Core suggest
@app.post("/suggest")
def suggest(body: dict = Body(default={})):
    """
    Body can include:
      - duration_min, buffer_min
      - attendees: list of emails (or ["primary"])
      - window_start / window_end (RFC3339)
      - OR: prompt and today_only: true/false
    """
    creds = load_creds(ORGANIZER_ID)
    if not creds:
        return JSONResponse({"error": "not_connected"}, status_code=401)

    try:
        # duration / buffer
        duration_min = int(body.get("duration_min", 45))
        buffer_min   = int(body.get("buffer_min", 15))

        attendees_raw = body.get("attendees", ["primary"])
        attendees = [a for a in attendees_raw if a == "primary" or "@" in a]
        if not attendees:
            attendees = ["primary"]

        # compute window: either explicit window OR derived from prompt + today_only
        start_iso = body.get("window_start")
        end_iso   = body.get("window_end")
        prompt_text = body.get("prompt", "")
        today_only_flag = bool(body.get("today_only", False))

        if start_iso and end_iso:
            start = dparse.isoparse(start_iso)
            end   = dparse.isoparse(end_iso)
        else:
            if prompt_text:
                if today_only_flag:
                    # clamp to today in LOCAL_TZ
                    start_dt, end_dt = get_search_window_from_prompt(prompt_text, days_ahead=0)
                else:
                    start_dt, end_dt = get_search_window_from_prompt(prompt_text, days_ahead=7)
                start = start_dt.astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
                end = end_dt.astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
            else:
                now = datetime.utcnow()
                start = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
                end   = (now + timedelta(days=5)).replace(hour=18, minute=30, second=0, microsecond=0)

        # call freebusy
        fb = freebusy(creds, attendees, rfc3339(start), rfc3339(end))

        busy_lists = [[{"start":b["start"], "end":b["end"]} for b in fb[cal]] for cal in fb]
        free = compute_free(busy_lists, start, end)
        cands = candidates(free, duration_min=duration_min, buffer_min=buffer_min)

        # fairness-aware scoring
        scored = []
        for sl in cands:
            score, comp = _fairness_score(sl, fb)
            scored.append((score, comp, sl))

        top = sorted(scored, key=lambda x: x[0], reverse=True)[:3]
        resp=[]
        for score, comp, sl in top:
            start_dt = datetime.fromisoformat(sl["start"])
            end_dt   = datetime.fromisoformat(sl["end"])
            why = explain(sl)
            why.update({"fairness": {
                "midday_closeness": comp["midday_closeness"],
                "min_gap_minutes": comp["min_gap_minutes"],
                "avg_gap_minutes": comp["avg_gap_minutes"]
            }})
            resp.append({
                "slot": sl,
                "start": sl["start"],
                "end": sl["end"],
                "human": f"{pretty(start_dt)} → {pretty(end_dt)}",
                "explanation": why,
                "score": round(score, 3)
            })
        return resp

    except HttpError as he:
        try:
            detail = he.error_details if hasattr(he, "error_details") else he.content.decode()
        except Exception:
            detail = str(he)
        return JSONResponse({"error": "google_api_error", "detail": detail}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": "server_error", "detail": str(e), "trace": traceback.format_exc()}, status_code=500)

# Book
@app.post("/book")
def book(body: dict):
    """
    Expects:
      - start (ISO), end (ISO)
      - attendees: list of emails
      - subject (optional)
      - why (optional)
    After creating event, patch description to include recorder link.
    """
    creds = load_creds(ORGANIZER_ID)
    if not creds:
        return JSONResponse({"error": "not_connected"}, status_code=401)
    try:
        s = body["start"]; e = body["end"]
        attendees = body.get("attendees", [])
        subject = body.get("subject", "Timetable-Aware Meeting")
        explanation = body.get("why", "Scheduled by AI bot.")
        tz = body.get("time_zone", LOCAL_TZ)

        def to_rfc3339(x: str) -> str:
            dt = dparse.isoparse(x)
            if dt.tzinfo is None: return dt.isoformat()+"Z"
            return dt.isoformat()

        ev = insert_event(
            creds, "primary", subject,
            to_rfc3339(s), to_rfc3339(e), attendees,
            description=explanation, add_meet=True, time_zone=tz
        )

        # After creation, patch description to include recorder link (use htmlLink if available)
        ev_id = ev.get("id")
        html_link = ev.get("htmlLink") or ""
        # determine a sensible owner identifier: prefer creator/organizer email, else client_id, else 'organizer'
        organizer_email = None
        try:
            organizer_email = (ev.get("creator") or {}).get("email") or (ev.get("organizer") or {}).get("email")
        except Exception:
            organizer_email = None
        if not organizer_email:
            # fallback to OAuth client_id if available (not ideal but usable)
            try:
                creds_local = creds
                organizer_email = getattr(creds_local, "client_id", None) or "organizer"
            except Exception:
                organizer_email = "organizer"

        # Build a direct recorder link to the static page using proper quoting (no need for redirect).
        encoded_meeting = urllib.parse.quote(html_link or ev_id or "", safe='')
        encoded_owner = urllib.parse.quote(organizer_email or "organizer", safe='')
        direct_recorder = f"{APP_BASE_URL}/ui/recorder.html?meeting={encoded_meeting}&owner={encoded_owner}"

        # Keep a legacy recorder/start link as fallback
        qp = urllib.parse.urlencode({
            "meeting": html_link or ev_id or "",
            "owner": organizer_email
        }, safe='')
        legacy_recorder = f"{APP_BASE_URL}/recorder/start?{qp}"

        # Put raw URLs in event description so copy-paste yields the correct target (not Google wrapper)
        new_description = (explanation or "") + "\n\nRecorder / Upload transcript (direct link):\n" \
                          f"{direct_recorder}\n\n(legacy compatibility link):\n{legacy_recorder}\n\n" \
                          "(Participants must click the link and consent to recording/transcription.)"
        try:
            patched = patch_event_description(creds, "primary", ev_id, new_description)
        except Exception:
            # if patch fails, we silently continue — event was created successfully
            patched = None

        return {"eventId": ev.get("id"), "htmlLink": ev.get("htmlLink"), "hangoutLink": ev.get("hangoutLink")}
    except Exception as e:
        return JSONResponse({"error":"server_error","detail":str(e)}, status_code=500)


# Robust redirect for recorder links
@app.get("/recorder/start")
def recorder_start_redirect(request: Request):
    """
    Robust handler for recorder links:
     - Logs incoming url for debugging.
     - Preserves raw querystring and redirects to absolute /ui/recorder.html?...
     - If no querystring, tries to extract a 'q=' fragment (defensive).
     - If still no params, serves static recorder.html directly.
    """
    try:
        logger.info("recorder_start incoming_url: %s", str(request.url))
    except Exception:
        pass

    qs = request.url.query  # raw querystring (percent-encoded)
    if not qs:
        raw = str(request.url)
        if "q=" in raw:
            after_q = raw.split("q=", 1)[1]
            after_q = after_q.split("&", 1)[0]
            qs = after_q

    target = f"{APP_BASE_URL}/ui/recorder.html"
    if qs:
        target = f"{target}?{qs}"

    if not qs:
        # fallback to serving the static file directly
        try:
            file_path = Path("static") / "recorder.html"
            if file_path.exists():
                return FileResponse(str(file_path), media_type="text/html")
        except Exception:
            pass

    return RedirectResponse(url=target)


# ---------- Recorder / upload endpoint (updated) ----------
@app.post("/upload-recording")
async def upload_recording(request: Request):
    """
    Accepts:
      - JSON { meeting, owner, participant_email, transcript, final (opt) }
      - multipart/form-data with 'meeting', 'owner', 'participant_email', and 'audio' file
    Behavior:
      - Stores per-meeting aggregated data in MOMS/<meeting_hash>.json (participants -> transcripts)
      - Saves uploaded audio blobs to data/uploads/
      - Returns {"status":"ok","mom_link": "..."} when transcript(s) exist
    """
    try:
        def meeting_hash(meeting_url: str) -> str:
            # create a short filesystem-safe hash/identifier for a meeting URL
            import hashlib
            return hashlib.sha256(meeting_url.encode("utf-8")).hexdigest()[:16]

        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
            meeting = payload.get("meeting")
            owner = payload.get("owner")
            participant = payload.get("participant_email") or owner or "unknown"
            transcript = payload.get("transcript", "")
            final = bool(payload.get("final", True))

            if not meeting:
                return JSONResponse({"error": "no_meeting"}, status_code=400)
            if not transcript:
                return JSONResponse({"error": "no_transcript"}, status_code=400)

            mid = meeting_hash(meeting)
            mom_file = MOMS / f"meeting_{mid}.json"
            if mom_file.exists():
                record = json.loads(mom_file.read_text(encoding="utf-8"))
            else:
                record = {
                    "meeting": meeting,
                    "owner": owner,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "participants": {},   # email -> { transcripts: [ ... ], audio_files: [] }
                }

            part = record["participants"].setdefault(participant, {"transcripts": [], "audio_files": []})
            part["transcripts"].append({
                "text": transcript,
                "uploaded_at": datetime.utcnow().isoformat() + "Z",
                "final": final
            })
            mom_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

            # optionally produce a short MoM immediately (summarizer used for whole meeting)
            # We'll generate a combined summary by concatenating participant transcripts.
            combined_text = "\n\n".join(
                " ".join([t.get("text","") for t in rec.get("transcripts",[])])
                for rec in record["participants"].values()
            )
            mom = summarize_transcript(combined_text)
            mom_id = f"meeting_{mid}"
            mom_path = MOMS / f"{mom_id}.json"
            mom_record = {
                "id": mom_id,
                "meeting": meeting,
                "owner": owner,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "participants": record["participants"],
                "mom": mom
            }
            mom_path.write_text(json.dumps(mom_record, indent=2), encoding="utf-8")
            mom_link = f"{APP_BASE_URL}/mom/{mom_id}"
            return JSONResponse({"status":"ok","mom_link": mom_link, "mom": mom_record})

        else:
            # form / multipart flow: upload audio blob + meeting + participant_email
            form = await request.form()
            meeting = form.get("meeting")
            owner = form.get("owner")
            participant = form.get("participant_email") or owner or "unknown"
            audio_file = form.get("audio")
            if not meeting:
                return JSONResponse({"error":"no_meeting"}, status_code=400)
            if not audio_file:
                return JSONResponse({"error":"no_audio"}, status_code=400)

            # save audio
            fname = f"{uuid.uuid4()}_{getattr(audio_file, 'filename', 'upload.webm')}"
            dest = UPLOADS / fname
            contents = await audio_file.read()
            dest.write_bytes(contents)

            # update meeting record (participants -> audio_files)
            mid = meeting_hash(meeting)
            mom_file = MOMS / f"meeting_{mid}.json"
            if mom_file.exists():
                record = json.loads(mom_file.read_text(encoding="utf-8"))
            else:
                record = {"meeting": meeting, "owner": owner, "created_at": datetime.utcnow().isoformat() + "Z", "participants": {}}
            part = record["participants"].setdefault(participant, {"transcripts": [], "audio_files": []})
            part["audio_files"].append({"path": str(dest), "uploaded_at": datetime.utcnow().isoformat() + "Z"})
            mom_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

            return JSONResponse({"status":"ok","saved": str(dest)})

    except Exception as e:
        return JSONResponse({"error":"server_error","detail":str(e), "trace": traceback.format_exc()}, status_code=500)


# ---------- MoM view (updated to show participants + transcripts) ----------
@app.get("/mom/{mom_id}", response_class=HTMLResponse)
def get_mom(mom_id: str):
    mom_file = MOMS / f"{mom_id}.json"
    if not mom_file.exists():
        return HTMLResponse("<h3>MoM not found</h3>", status_code=404)
    mom_record = json.loads(mom_file.read_text(encoding="utf-8"))
    html = f"<div style='font-family:Inter,Arial;padding:18px;max-width:900px;margin:20px auto;'>"
    html += f"<h2>Minutes of Meeting</h2><p><strong>Meeting:</strong> {mom_record.get('meeting')}</p>"
    html += f"<p><strong>Owner:</strong> {mom_record.get('owner')}</p>"
    html += f"<h3>Summary</h3><p>{mom_record.get('mom',{}).get('summary','(no summary)')}</p>"
    html += "<h3>Action Items</h3><ul>"
    for a in mom_record.get('mom',{}).get('action_items', []):
        html += f"<li>{a}</li>"
    html += "</ul><hr>"
    html += "<h3>Participant transcripts & files</h3>"
    for p_email, pdata in mom_record.get('participants', {}).items():
        html += f"<h4>{p_email}</h4>"
        for i, t in enumerate(pdata.get('transcripts', [])):
            text = t.get('text','')
            uploaded = t.get('uploaded_at','')
            html += f"<div style='border:1px solid #eee;padding:8px;margin-bottom:6px;'><strong>Transcript #{i+1} ({uploaded})</strong><pre style='white-space:pre-wrap;font-family:inherit'>{text}</pre></div>"
        for af in pdata.get('audio_files', []):
            path = af.get('path')
            html += f"<div>Audio file: <code>{path}</code></div>"
    html += f"<hr><p><em>Generated at {mom_record.get('created_at')}</em></p></div>"
    return HTMLResponse(html)
