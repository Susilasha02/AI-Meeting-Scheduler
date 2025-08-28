import os, json, re, traceback, zoneinfo
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

from fastapi import FastAPI, Body, Request
from fastapi.responses import JSONResponse
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

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:3978")
PORT = int(os.getenv("PORT", "3978"))
CLIENT_FILE = "google_oauth_client.json"
LOCAL_TZ = os.getenv("TIME_ZONE", "UTC")

# --------- stores ----------
TOKEN_STORE = "token_store.json"
CONTACTS_STORE = "contacts.json"
ORGANIZER_ID = "organizer"

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

# ---------- FastAPI ----------
app = FastAPI()
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
    creds = load_creds(ORGANIZER_ID)
    if not creds:
        return JSONResponse({"error": "not_connected"}, status_code=401)

    try:
        duration_min = int(body.get("duration_min", 45))
        buffer_min   = int(body.get("buffer_min", 15))

        attendees_raw = body.get("attendees", ["primary"])
        attendees = [a for a in attendees_raw if a == "primary" or "@" in a]
        if not attendees:
            attendees = ["primary"]

        start_iso = body.get("window_start")
        end_iso   = body.get("window_end")
        if start_iso and end_iso:
            start = dparse.isoparse(start_iso)
            end   = dparse.isoparse(end_iso)
        else:
            now = datetime.utcnow()
            start = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
            end   = (now + timedelta(days=5)).replace(hour=18, minute=30, second=0, microsecond=0)

        # proper RFC3339
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
        return {"eventId": ev.get("id"), "htmlLink": ev.get("htmlLink"), "hangoutLink": ev.get("hangoutLink")}
    except Exception as e:
        return JSONResponse({"error":"server_error","detail":str(e)}, status_code=500)
