import os, json, re, traceback, uuid, zoneinfo, urllib.parse, logging
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path


from fastapi import FastAPI, Body, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from dotenv import load_dotenv


from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.errors import HttpError
from google.auth.transport.requests import AuthorizedSession


from dateutil import parser as dparse
import pendulum

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

# ---------- improved_natural_date_parse helpers ----------
# Use the app-level LOCAL_TZ variable rather than hardcoding
# Regex helpers
DUR_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(hours|hour|hrs|hr|h|minutes|min|mins|m)\b", re.I)
NEXT_WEEKDAY_RE = re.compile(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
THIS_WEEKDAY_RE = re.compile(r"\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
WEEKDAY_RE = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
TIME_RE = re.compile(r"\b(\d{1,2}(:\d{2})?\s*(?:am|pm|AM|PM)?)\b")

WEEKDAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6
}

def _parse_duration_minutes(prompt: str, default_min: int = 45) -> int:
    """Parse duration from text. returns minutes."""
    m = DUR_RE.search(prompt)
    if not m:
        return default_min
    val = m.group(1).replace(",", ".")
    unit = m.group(2).lower()
    valf = float(val)
    if unit.startswith("h"):
        return int(round(valf * 60))
    # minutes
    return int(round(valf))

def _next_weekday_after(base_date: datetime, weekday_index: int, at_least_one_week_ahead: bool=False):
    """
    Return the next date for weekday_index (0=Mon..6=Sun) after base_date.
    If at_least_one_week_ahead True, always return the weekday in the next week (i.e., 'next Wednesday').
    """
    days_ahead = weekday_index - base_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    if at_least_one_week_ahead and days_ahead < 7:
        days_ahead += 7
    return base_date + timedelta(days=days_ahead)

def _apply_time_preferences(dt_date: datetime, prompt: str, local_tz: zoneinfo.ZoneInfo):
    """
    Given a date (may have 0:00 time), pick an appropriate time:
      - if explicit time present in prompt (e.g., '3pm') -> use parsed time
      - if contains 'morning' -> 09:00, 'afternoon' -> 14:00, 'evening' -> 18:00
      - otherwise default to 09:00
    Returns an aware datetime in local tz.
    """
    p = (prompt or "").lower()
    # Check explicit time tokens first
    tm = TIME_RE.search(prompt)
    if tm:
        # try to parse the time token with dateparser against today's date
        parsed = dateparser.parse(tm.group(1), settings={"PREFER_DATES_FROM": "future"})
        if parsed:
            return dt_date.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0, tzinfo=local_tz)

    if "morning" in p:
        return dt_date.replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=local_tz)
    if "afternoon" in p:
        return dt_date.replace(hour=14, minute=0, second=0, microsecond=0, tzinfo=local_tz)
    if "evening" in p or "night" in p:
        return dt_date.replace(hour=18, minute=0, second=0, microsecond=0, tzinfo=local_tz)

    # default
    return dt_date.replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=local_tz)

# ---- parse_prompt_to_window (unchanged from your last version) ----
import pendulum

def parse_prompt_to_window(prompt: str, contacts_map: dict = None, default_duration_min: int = 45):
    """
    Pendulum-backed parser. Returns dict:
      {
        "attendees": [...],
        "duration_min": int,
        "window_start": rfc3339(start_utc_naive),
        "window_end": rfc3339(end_utc_naive)
      }
    Interprets naive parsed times as LOCAL_TZ and returns RFC3339 UTC strings (matching your existing code expectations).
    """
    p = (prompt or "").strip()
    if contacts_map is None:
        contacts_map = {}

    duration_min = _parse_duration_minutes(p, default_min=default_duration_min)

    # attendees: allow @username mentions or emails in prompt
    attendees = set()
    email_re = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
    for e in email_re.findall(p):
        attendees.add(e)
    mention_re = re.compile(r"@([a-zA-Z0-9_.-]+)")
    for m in mention_re.findall(p):
        key = m.lower()
        if key in contacts_map:
            attendees.add(contacts_map[key])
        else:
            attendees.add("@" + m)

    # timezone
    try:
        local_tz_name = LOCAL_TZ or "UTC"
        tz = pendulum.timezone(local_tz_name)
    except Exception:
        tz = pendulum.timezone("UTC")

    now_local = pendulum.now(tz)

    # try explicit weekday keywords first (fast path)
    plow = p.lower()

    # handle explicit "next monday/this friday" using previous helper logic
    m_next = NEXT_WEEKDAY_RE.search(p)
    if m_next:
        weekday = m_next.group(1).lower()
        idx = WEEKDAY_MAP[weekday]
        # compute next-week date
        target = now_local.next(pendulum.MONDAY) if weekday=="monday" else None
        # pendulum has .next() for day name; simpler compute:
        days_ahead = (idx - now_local.day_of_week) % 7
        if days_ahead == 0:
            days_ahead = 7
        # enforce next-week => add 7
        days_ahead += 7
        dt_date = now_local.add(days=days_ahead).date()
        # apply time preferences
        start_local = _apply_time_preferences(datetime(dt_date.year, dt_date.month, dt_date.day), p, zoneinfo.ZoneInfo(local_tz_name))
        start_local = pendulum.instance(start_local, tz)  # ensure pendulum object
        start_local = start_local.replace(second=0, microsecond=0)
        start_utc = start_local.in_timezone("UTC").naive()
        end_utc = (start_local.add(minutes=duration_min)).in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

    m_this = THIS_WEEKDAY_RE.search(p)
    if m_this:
        weekday = m_this.group(1).lower()
        idx = WEEKDAY_MAP[weekday]
        days_ahead = (idx - now_local.day_of_week) % 7
        if days_ahead == 0:
            days_ahead = 7 if now_local.hour >= 23 else 0  # if same day late, push to next week
        dt_date = now_local.add(days=days_ahead).date()
        start_local = _apply_time_preferences(datetime(dt_date.year, dt_date.month, dt_date.day), p, zoneinfo.ZoneInfo(local_tz_name))
        start_local = pendulum.instance(start_local, tz)
        if start_local <= now_local:
            start_local = start_local.add(days=7)
        start_local = start_local.replace(second=0, microsecond=0)
        start_utc = start_local.in_timezone("UTC").naive()
        end_utc = start_local.add(minutes=duration_min).in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

    # If there's an explicit time token (e.g., "11am", "3:30pm"), pick that time on either today or requested day.
    tm = TIME_RE.search(p)
    if tm:
        # parse numeric time pieces
        hpart = tm.group(1)
        # use dateparser? Instead parse with custom logic:
        time_text = hpart
        # Use pendulum.parse with locale-aware fallback
        try:
            # pendulum's parse may be forgiving but assume time only -> combine with date
            parsed_time = pendulum.parse(time_text, tz=tz, default=now_local)
            # parsed_time could include date; normalize to today or tomorrow if contains date
            candidate = parsed_time
        except Exception:
            # fallback simple parse: "3pm" -> hour 15 etc
            m2 = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", hpart, re.I)
            if m2:
                hour = int(m2.group(1))
                minute = int(m2.group(2) or 0)
                mer = (m2.group(3) or "").lower()
                if mer == "pm" and hour < 12:
                    hour += 12
                if mer == "am" and hour == 12:
                    hour = 0
                candidate = now_local.set(hour=hour, minute=minute, second=0, microsecond=0)
            else:
                candidate = now_local

        # Determine date for candidate: if prompt mentions "tomorrow" or weekday, use that; else prefer today
        if "tomorrow" in plow:
            candidate = candidate.add(days=1)
        else:
            # if candidate time already passed today, use tomorrow
            if candidate <= now_local:
                candidate = candidate.add(days=1)

        start_local = candidate.replace(second=0, microsecond=0)
        start_utc = start_local.in_timezone("UTC").naive()
        end_utc = start_local.add(minutes=duration_min).in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

    # shortcuts: "today" / "tomorrow" / "in X days"
    if "today" in plow:
        # pick near-future (now + 15m) or 09:00 if earlier
        candidate = now_local.add(minutes=15)
        if candidate.hour < 9:
            candidate = candidate.set(hour=9, minute=0)
        start_local = candidate.replace(second=0, microsecond=0)
        start_utc = start_local.in_timezone("UTC").naive()
        end_utc = start_local.add(minutes=duration_min).in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

    if "tomorrow" in plow:
        tomorrow = now_local.add(days=1).set(hour=9, minute=0, second=0, microsecond=0)
        start_local = tomorrow
        start_utc = start_local.in_timezone("UTC").naive()
        end_utc = start_local.add(minutes=duration_min).in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

    m_in = re.search(r"in\s+(\d+)\s+day", plow)
    if m_in:
        delta = int(m_in.group(1))
        target = now_local.add(days=delta).set(hour=9, minute=0, second=0, microsecond=0)
        start_local = target
        start_utc = start_local.in_timezone("UTC").naive()
        end_utc = start_local.add(minutes=duration_min).in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

    # fallback: rolling 1-5 days window starting tomorrow 09:00
    fallback_start = now_local.add(days=1).set(hour=9, minute=0, second=0, microsecond=0)
    fallback_end = fallback_start.add(days=4).set(hour=18, minute=30)
    start_utc = fallback_start.in_timezone("UTC").naive()
    end_utc = fallback_end.in_timezone("UTC").naive()
    return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

# Wrapper parse_prompt to produce the config suggest() expects
def parse_prompt(prompt: str, contacts_map: Dict[str, str] = None) -> Dict:
    """
    Produces a dict compatible with the /suggest body:
      {
        "attendees": [...],
        "duration_min": int,
        "buffer_min": int,
        "window_start": "RFC3339",
        "window_end": "RFC3339"
      }
    """
    try:
        parsed = parse_prompt_to_window(prompt, contacts_map or {}, default_duration_min=45)
        # parsed already returns those fields; ensure buffer_min exists
        parsed.setdefault("buffer_min", 15)
        return parsed
    except Exception:
        # defensive fallback to rolling window to avoid crashes
        logger.exception("parse_prompt failed for prompt: %s", prompt)
        now = datetime.utcnow()
        return {
            "attendees": ["primary"],
            "duration_min": 45,
            "buffer_min": 15,
            "window_start": rfc3339((now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)),
            "window_end": rfc3339((now + timedelta(days=5)).replace(hour=18, minute=30, second=0, microsecond=0)),
        }

# ---------- summarizer for MoM (improved with optional spaCy) ----------
import re
from collections import defaultdict

# Try to use spaCy if available (better named-entity & dependency parsing).
# If not installed, code will gracefully fall back to the regex-based extractor.
try:
    import spacy
    try:
        # prefer loaded model if present; this may raise if not downloaded
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        # if model not present, try the simpler load which may raise ImportError
        _nlp = None
except Exception:
    _nlp = None

# Keep a small nltk fallback for sentence tokenization (as before)
try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
except Exception:
    nltk = None

# Common filler/greeting tokens to ignore when they're the first token
_GREETINGS = {"hello", "hi", "hey", "ok", "okay", "so", "alright", "right", "well", "today", "thanks", "thank"}

# Action trigger verbs / phrases (keep original list for fallback)
_ACTION_TRIGGERS = [
    "will", "shall", "is going to", "is to", "will be", "should", "must",
    "needs to", "need to", "need", "task", "assign", "please", "due", "deadline",
    "report", "deliver", "submit", "prepare", "create", "make", "send", "complete",
    "clean", "model", "process", "analyze", "analyse"
]

def _split_sentences(text: str):
    if not text or not text.strip():
        return []
    txt = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if nltk:
        try:
            sents = nltk.sent_tokenize(txt)
            return [s.strip() for s in sents if s.strip()]
        except Exception:
            pass
    # fallback naive split
    parts = re.split(r'(?<=[.!?])\s+', txt)
    return [p.strip() for p in parts if p.strip()]

def _shorten_action(s: str, maxlen: int = 180) -> str:
    s = s.strip()
    if len(s) <= maxlen:
        return s
    return s[:maxlen].rsplit(" ", 1)[0] + "…"

def _spacy_extract_actions(sent: str):
    """
    Use spaCy dependency parse to extract (subject, verb phrase / object) pairs.
    Returns list of (subject_or_None, action_text).
    """
    if not _nlp:
        return []
    doc = _nlp(sent)
    actions = []
    # find tokens that are main verbs (ROOT or VERB) and extract their subject and object/clause
    for tok in doc:
        if tok.pos_ == "VERB" or tok.dep_ == "ROOT":
            # find subject
            subj = None
            for ch in tok.children:
                if ch.dep_.startswith("nsubj"):
                    subj = ch.text
                    # prefer proper name if available in subtree
                    for ent in doc.ents:
                        if ent.start <= ch.i <= ent.end:
                            subj = ent.text
                            break
                    break
            # build a short action phrase: verb + direct object + relevant subtree
            parts = [tok.lemma_]
            # include direct objects and their compounds/objects
            dobj_parts = []
            for ch in tok.children:
                if ch.dep_.endswith("obj") or ch.dep_ in ("dobj", "pobj", "attr"):
                    dobj_parts.append(" ".join([t.text for t in ch.subtree]))
            # if no direct object, include the verb's subtree minus subject
            if dobj_parts:
                action_text = tok.text + " " + ", ".join(dobj_parts)
            else:
                # include verb + a short slice of subtree (avoid entire sentence)
                subtree_words = [t.text for t in tok.subtree if t.i >= tok.i][:10]
                action_text = " ".join(subtree_words)
            action_text = _shorten_action(action_text)
            actions.append((subj, action_text))
    return actions

def _legacy_extract_person_action(s: str):
    """Keep fallback legacy extractor (your original rules) for safety."""
    # (This reuses your original approach: triggers + name heuristics simplified)
    # Try to capture patterns like "John will X" or "please John: X"
    s_norm = s.strip()
    # look for "Name will/should/please ..." style
    m = re.search(r"\b([A-Z][a-z]{1,30})\b\s+(?:will|shall|should|is to|is going to|please|must|needs to)\b", s)
    if m:
        name = m.group(1)
        # take remainder as action
        rest = s[m.end():].strip()
        if rest:
            return name, _shorten_action(rest)
    # generic: scan for action trigger words and return the sentence
    if any(re.search(rf"\b{re.escape(tok)}\b", s_norm, re.I) for tok in _ACTION_TRIGGERS):
        return None, _shorten_action(s_norm)
    return None, None

def summarize_transcript(transcript: str, max_sentences: int = 6):
    """
    Returns {"summary": str, "action_items": [ . ] }
    Uses spaCy if available for much better action extraction; falls back to original heuristics.
    """
    if not transcript or not transcript.strip():
        return {"summary": "(no transcript provided)", "action_items": []}

    clean = re.sub(r"\s+", " ", transcript.replace("\r", " ").strip())
    sentences = _split_sentences(clean)
    if not sentences:
        sentences = [clean]

    # Build summary: prefer sentences that contain meeting-like triggers; else first N sentences
    triggers = ["discuss", "meeting", "project", "goal", "topic", "update", "decide", "plan"]
    important = [s for s in sentences if any(t in s.lower() for t in triggers)]
    if not important:
        important = sentences[:max_sentences]
    summary = " ".join(important[:max_sentences]).strip()

    # Extract actions — use spaCy when available
    person_tasks = defaultdict(list)
    general_tasks = []

    if _nlp:
        # spaCy extraction across sentences
        for s in sentences:
            acts = _spacy_extract_actions(s)
            if not acts:
                # fallback to legacy for this sentence
                name, action = _legacy_extract_person_action(s)
                if action:
                    if name:
                        person_tasks[name].append(action)
                    else:
                        general_tasks.append(action)
            else:
                for name, action in acts:
                    if action:
                        if name:
                            person_tasks[name].append(action)
                        else:
                            general_tasks.append(action)
    else:
        # legacy extraction across sentences
        for s in sentences:
            name, action = _legacy_extract_person_action(s)
            if action:
                if name:
                    person_tasks[name].append(action)
                else:
                    general_tasks.append(action)

    # Looser fallback pass if nothing found: check trigger words in short clauses
    if not person_tasks and not general_tasks:
        for s in sentences:
            if re.search(r"\b(" + "|".join([re.escape(t) for t in _ACTION_TRIGGERS]) + r")\b", s, re.I):
                general_tasks.append(_shorten_action(s))

    # Format action items
    action_items = []
    for person, tasks in person_tasks.items():
        seen = set()
        for t in tasks:
            if t not in seen:
                seen.add(t)
                action_items.append(f"{person}: {t}")
    for t in dict.fromkeys(general_tasks).keys():
        action_items.append(f"General: {t}")

    if not action_items:
        action_items = ["(no explicit action items detected)"]

    return {"summary": summary or "(no summary extracted)", "action_items": action_items}

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

# NL → suggest (safer wrapper)
@app.post("/nlp/suggest")
def nlp_suggest(body: dict = Body(...)):
    try:
        creds = load_creds(ORGANIZER_ID)
        if not creds:
            return JSONResponse({"error": "not_connected"}, status_code=401)
        prompt = body.get("prompt","").strip()
        contacts_map = load_contacts()
        cfg = parse_prompt(prompt, contacts_map)
        # pass the cfg dict into suggest() - it expects a body-like dict
        return suggest(cfg)
    except Exception as e:
        logger.error("Exception in /nlp/suggest: %s\n%s", str(e), traceback.format_exc())
        return JSONResponse({"error": "server_error", "detail": str(e), "trace": traceback.format_exc()}, status_code=500)

# ---------- New helper: generate_suggestions (evening-aware) ----------
def generate_suggestions(
    events: List[Tuple[datetime, datetime]],
    duration_min: int,
    buffer_min: int,
    window_start: datetime,
    window_end: datetime,
    tzinfo: zoneinfo.ZoneInfo,
    prompt_text: Optional[str] = None,
    max_results: int = 10
):
    """
    events: list of (start_dt, end_dt) datetimes in tzinfo timezone (or timezone-aware)
    duration_min, buffer_min: ints
    window_start, window_end: timezone-aware datetimes (in tzinfo)
    prompt_text: original user prompt (used to detect 'tonight' / pm preferences)
    Returns list of dicts: [{"start": dt_start, "end": dt_end}, ...] (datetimes in tzinfo)
    """
    # Normalize window to tzinfo
    if window_start.tzinfo is None:
        start_local = window_start.replace(tzinfo=tzinfo)
    else:
        start_local = window_start.astimezone(tzinfo)
    if window_end.tzinfo is None:
        end_local = window_end.replace(tzinfo=tzinfo)
    else:
        end_local = window_end.astimezone(tzinfo)

    # Normalize event list into sorted tuples in tzinfo
    normalized_busy = []
    for s, e in events:
        if s.tzinfo is None:
            s = s.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(tzinfo)
        else:
            s = s.astimezone(tzinfo)
        if e.tzinfo is None:
            e = e.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(tzinfo)
        else:
            e = e.astimezone(tzinfo)
        normalized_busy.append((s, e))
    normalized_busy.sort(key=lambda x: x[0])

    # Detect evening/night preference from prompt
    prefers_evening = False
    if prompt_text:
        pl = prompt_text.lower()
        if any(tok in pl for tok in ["tonight", "evening", "night", "late", "pm", "9pm", "10pm", "11pm"]):
            prefers_evening = True

    # Expand allowed daily hours if evening requested
    if prefers_evening:
        day_start_hour = 6
        day_end_hour = 23
    else:
        day_start_hour = 6
        day_end_hour = 18

    # Start scanning slots from the window start (rounded)
    slot_cursor = start_local.replace(second=0, microsecond=0)
    if slot_cursor.hour < day_start_hour:
        slot_cursor = slot_cursor.replace(hour=day_start_hour, minute=0)

    candidate_slots = []
    iterations = 0
    max_iterations = 2000

    while slot_cursor + timedelta(minutes=duration_min) <= end_local and len(candidate_slots) < max_results and iterations < max_iterations:
        iterations += 1
        slot_end = slot_cursor + timedelta(minutes=duration_min)

        # Check daily hour bounds for the slot
        if (slot_cursor.hour >= day_start_hour) and (slot_end.hour <= day_end_hour or (slot_end.hour == day_end_hour and slot_end.minute == 0)):
            # Check overlap with busy events
            overlap = False
            for bstart, bend in normalized_busy:
                if slot_end <= bstart or slot_cursor >= bend:
                    continue
                overlap = True
                break
            if not overlap:
                candidate_slots.append({"start": slot_cursor, "end": slot_end})

        slot_cursor = slot_cursor + timedelta(minutes=buffer_min)

    return candidate_slots

# ---------- Improved /suggest with robust fallbacks & debug output ----------
@app.post("/suggest")
def suggest(body: dict = Body(default={})):
    """
    Improved suggest:
      - uses parse_prompt() when prompt is present (which preserves explicit times)
      - keeps tz-awareness
      - falls back to assume free calendar if freebusy fails
      - returns debug info when helpful
    """
    creds = load_creds(ORGANIZER_ID)
    if not creds:
        return JSONResponse({"error": "not_connected"}, status_code=401)

    debug_info = {}
    try:
        # basic params
        duration_min = int(body.get("duration_min", 45))
        buffer_min = int(body.get("buffer_min", 15))
        attendees_raw = body.get("attendees", ["primary"])
        attendees = [a for a in attendees_raw if a == "primary" or "@" in a]
        if not attendees:
            attendees = ["primary"]

        prompt_text = (body.get("prompt") or "").strip()
        today_only_flag = bool(body.get("today_only", False))

        # local tz
        try:
            local_tz = zoneinfo.ZoneInfo(LOCAL_TZ)
        except Exception:
            local_tz = zoneinfo.ZoneInfo("UTC")
        debug_info["LOCAL_TZ"] = str(local_tz)

        # 1) Determine search window (prefer explicit window_start/window_end if provided)
        start_iso = body.get("window_start")
        end_iso = body.get("window_end")
        if start_iso and end_iso:
            start = dparse.isoparse(start_iso)
            end = dparse.isoparse(end_iso)
            if start.tzinfo is None:
                start = start.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
            if end.tzinfo is None:
                end = end.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
            debug_info["window_source"] = "explicit"
        else:
            # prefer parse_prompt (already handles dates/times)
            if prompt_text:
                try:
                    contacts_map = load_contacts()
                    parsed = parse_prompt(prompt_text, contacts_map)
                    debug_info["parsed_prompt"] = parsed
                    # parse RFC3339 strings into datetimes
                    start = dparse.isoparse(parsed["window_start"])
                    end = dparse.isoparse(parsed["window_end"])
                    # ensure tz-aware (parse_prompt returns naive UTC; attach UTC if missing)
                    if start.tzinfo is None:
                        start = start.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
                    if end.tzinfo is None:
                        end = end.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
                    debug_info["window_source"] = "parse_prompt"
                except Exception as e:
                    debug_info["parse_prompt_error"] = str(e)
                    # fallback to the older helper
                    s_local, e_local = get_search_window_from_prompt(prompt_text, days_ahead=7)
                    # get_search_window_from_prompt returns tz-aware LOCAL_TZ datetimes
                    start = s_local.astimezone(zoneinfo.ZoneInfo("UTC"))
                    end = e_local.astimezone(zoneinfo.ZoneInfo("UTC"))
                    debug_info["window_source"] = "get_search_window_from_prompt_fallback"
            else:
                # No prompt and no explicit window: default rolling window
                now = datetime.utcnow().replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
                start = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
                end = (now + timedelta(days=5)).replace(hour=18, minute=30, second=0, microsecond=0)
                debug_info["window_source"] = "rolling_default"

        debug_info["start_utc"] = start.isoformat()
        debug_info["end_utc"] = end.isoformat()

        # Defensive: if start >= end, widen the window
        if start >= end:
            debug_info["window_adjusted"] = True
            end = start + timedelta(days=3, hours=9)

        # Convert to tz-aware local window for suggestion algorithm
        start_local = start.astimezone(local_tz)
        end_local = end.astimezone(local_tz)
        debug_info["start_local"] = start_local.isoformat()
        debug_info["end_local"] = end_local.isoformat()

        # 2) Call freebusy (safe try/catch)
        fb = {}
        try:
            fb = freebusy(creds, attendees, rfc3339(start), rfc3339(end))
            debug_info["freebusy_ok"] = True
            debug_info["freebusy_sample"] = {k: v[:3] for k, v in fb.items()}  # only small preview
        except Exception as e:
            debug_info["freebusy_ok"] = False
            debug_info["freebusy_error"] = str(e)
            # fall back to empty freebusy => assume no busy blocks
            fb = {a: [] for a in attendees}

        # 3) Build busy_events list (tz-aware in local_tz) for generator
        busy_events = []
        try:
            for cal_id, blocks in fb.items():
                # blocks is list of {"start": iso, "end": iso}
                for b in blocks:
                    s = dparse.isoparse(b["start"])
                    e = dparse.isoparse(b["end"])
                    if s.tzinfo is None:
                        s = s.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
                    if e.tzinfo is None:
                        e = e.replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
                    # convert into local tz for generator
                    s_local = s.astimezone(local_tz)
                    e_local = e.astimezone(local_tz)
                    busy_events.append((s_local, e_local))
        except Exception as e:
            debug_info["busy_parse_error"] = str(e)
            busy_events = []

        # 4) Generate suggestions (this respects "pm"/"tonight" from prompt_text)
        candidate_slots = generate_suggestions(
            busy_events,
            duration_min,
            buffer_min,
            window_start=start_local,
            window_end=end_local,
            tzinfo=local_tz,
            prompt_text=prompt_text,
            max_results=12
        )

        debug_info["candidate_slots_count"] = len(candidate_slots)

        # 5) If no candidates returned, fallback to treating calendar as free (i.e., ignore busy_events)
        if not candidate_slots:
            debug_info["fallback_used"] = "treat_calendar_free"
            # create a free window covering start_local..end_local and use candidates() (your original strategy)
            free_windows = [{"start": start_local.isoformat(), "end": end_local.isoformat()}]
            fallback_cands = candidates(free_windows, duration_min=duration_min, buffer_min=buffer_min, cap=3)
            # If still empty, relax cap and produce sliding windows across full day range
            if not fallback_cands:
                debug_info["fallback_used"] = "relaxed_candidates"
                # create sliding windows every buffer min across the local window (naive)
                curs = start_local.replace(minute=0, second=0, microsecond=0)
                fallback_cands_tmp = []
                max_iters = 200
                it = 0
                while curs + timedelta(minutes=duration_min) <= end_local and it < max_iters:
                    fallback_cands_tmp.append({"start": curs.isoformat(), "end": (curs + timedelta(minutes=duration_min)).isoformat(),
                        "meta": {"buffers": {"pre": buffer_min, "post": buffer_min}}})
                    curs += timedelta(minutes=buffer_min)
                    it += 1
                fallback_cands = fallback_cands_tmp[:10]
            cands = fallback_cands
        else:
            # convert candidate_slots (tz-aware datetimes) into same dict form as original candidates()
            cands = []
            for s in candidate_slots:
                # Convert back to UTC-naive ISO strings (to match previous internal expectations)
                s_utc_naive = s["start"].astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
                e_utc_naive = s["end"].astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
                cands.append({
                    "start": s_utc_naive.isoformat(),
                    "end": e_utc_naive.isoformat(),
                    "meta": {"buffers": {"pre": buffer_min, "post": buffer_min}}
                })

        debug_info["candidates_count"] = len(cands)

        # fairness scoring & response (unchanged)
        scored = []
        for sl in cands:
            try:
                score, comp = _fairness_score(sl, fb)
            except Exception as e:
                # if scoring fails for a slot, make a fallback neutral score
                score, comp = 0.0, {"midday_closeness": 0, "min_gap_minutes": 0, "avg_gap_minutes": 0, "used_attendees": list(fb.keys())}
            scored.append((score, comp, sl))

        top = sorted(scored, key=lambda x: x[0], reverse=True)[:3]
        resp = []
        for score, comp, sl in top:
            start_dt = datetime.fromisoformat(sl["start"])
            end_dt = datetime.fromisoformat(sl["end"])
            why = explain(sl)
            why.update({"fairness": {
                "midday_closeness": comp.get("midday_closeness"),
                "min_gap_minutes": comp.get("min_gap_minutes"),
                "avg_gap_minutes": comp.get("avg_gap_minutes")
            }})
            resp.append({
                "slot": sl,
                "start": sl["start"],
                "end": sl["end"],
                "human": f"{pretty(start_dt)} → {pretty(end_dt)}",
                "explanation": why,
                "score": round(score, 3)
            })

        # If no slots to return, include debug_info to help diagnose
        if not resp:
            return JSONResponse({"slots": [], "debug": debug_info}, status_code=200)

        # Otherwise return slots + small debug_info
        debug_info_brief = {k: debug_info.get(k) for k in ("window_source","start_local","end_local","candidate_slots_count","candidates_count")}
        return {"slots": resp, "debug": debug_info_brief}

    except HttpError as he:
        try:
            detail = he.error_details if hasattr(he, "error_details") else he.content.decode()
        except Exception:
            detail = str(he)
        return JSONResponse({"error": "google_api_error", "detail": detail}, status_code=500)
    except Exception as e:
        # Return full debug if something unexpected happens
        debug_info["exception"] = str(e)
        debug_info["trace"] = traceback.format_exc()
        return JSONResponse({"error": "server_error", "detail": str(e), "debug": debug_info}, status_code=500)


# ---------- Small debug endpoint to inspect parser/window outputs ----------
@app.get("/debug/parse")
def debug_parse(prompt: str):
    """
    Call /debug/parse?prompt=... to see:
      - parse_prompt(...) output (RFC3339 window)
      - get_search_window_from_prompt(...) (older fallback)
      - now (LOCAL_TZ)
    Useful to verify prompt -> window mapping.
    """
    try:
        contacts_map = load_contacts()
        parsed = parse_prompt(prompt, contacts_map)
    except Exception as e:
        parsed = {"error": str(e)}

    try:
        gw_start, gw_end = get_search_window_from_prompt(prompt, days_ahead=7)
        gw = {"start": gw_start.isoformat(), "end": gw_end.isoformat()}
    except Exception as e:
        gw = {"error": str(e)}

    try:
        tz = zoneinfo.ZoneInfo(LOCAL_TZ)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")
    now = datetime.now(tz).isoformat()

    return {"prompt": prompt, "parse_prompt": parsed, "get_search_window_from_prompt": gw, "now_local": now, "LOCAL_TZ": str(tz)}


# Helper get_search_window_from_prompt used by /suggest when prompt_text exists
def get_search_window_from_prompt(prompt: str, days_ahead: int = 7) -> Tuple[datetime, datetime]:
    """
    Pendulum-backed: returns (start_local_dt, end_local_dt) both tz-aware in LOCAL_TZ (pendulum DateTime).
    Converted where necessary to native datetime objects before returning to the rest of your code.
    """
    try:
        tzname = LOCAL_TZ or "UTC"
        tz = pendulum.timezone(tzname)
    except Exception:
        tz = pendulum.timezone("UTC")

    now = pendulum.now(tz)
    p = (prompt or "").lower().strip()

    # simple relative words
    if "today" in p:
        start = now.set(hour=9, minute=0, second=0, microsecond=0)
        end = now.set(hour=18, minute=30, second=0, microsecond=0)
        # if too late, roll to tomorrow
        if now.hour >= 18 and now.minute > 30:
            tomorrow = now.add(days=1)
            start = tomorrow.set(hour=9, minute=0, second=0, microsecond=0)
            end = tomorrow.set(hour=18, minute=30, second=0, microsecond=0)
        return start, end

    if "tomorrow" in p:
        t = now.add(days=1)
        start = t.set(hour=9, minute=0, second=0, microsecond=0)
        end = t.set(hour=18, minute=30, second=0, microsecond=0)
        return start, end

    m_in = re.search(r"in\s+(\d+)\s+day", p)
    if m_in:
        delta = int(m_in.group(1))
        t = now.add(days=delta)
        start = t.set(hour=9, minute=0, second=0, microsecond=0)
        end = t.set(hour=18, minute=30, second=0, microsecond=0)
        return start, end

    if "next week" in p:
        # start next week's Monday-ish
        days_to_next_week = (7 - now.day_of_week) or 7
        start = now.add(days=days_to_next_week).set(hour=9, minute=0, second=0, microsecond=0)
        end = start.add(days=5).set(hour=18, minute=30)
        return start, end

    # weekdays
    for word, idx in WEEKDAY_MAP.items():
        if f"next {word}" in p:
            days_ahead = (idx - now.day_of_week) % 7
            if days_ahead == 0:
                days_ahead = 7
            target = now.add(days=days_ahead + 7)
            start = target.set(hour=9, minute=0, second=0, microsecond=0)
            end = target.set(hour=18, minute=30, second=0, microsecond=0)
            return start, end
        if f"this {word}" in p:
            days_ahead = (idx - now.day_of_week) % 7
            target = now.add(days=days_ahead)
            start = target.set(hour=9, minute=0, second=0, microsecond=0)
            end = target.set(hour=18, minute=30, second=0, microsecond=0)
            if start <= now:
                # push to next week's same weekday
                target = target.add(days=7)
                start = target.set(hour=9, minute=0, second=0, microsecond=0)
                end = target.set(hour=18, minute=30, second=0, microsecond=0)
            return start, end
        if re.search(rf"\b{word}\b", p):
            days_ahead_calc = (idx - now.day_of_week) % 7
            if days_ahead_calc == 0:
                days_ahead_calc = 7
            target = now.add(days=days_ahead_calc)
            start = target.set(hour=9, minute=0, second=0, microsecond=0)
            end = target.set(hour=18, minute=30, second=0, microsecond=0)
            return start, end

    # explicit date/time parsing attempt: look for yyyy-mm-dd or day-month patterns or "Nov 5" style via pendulum's parse
    try:
        # pendulum.parse will attempt to parse natural strings; default to local tz
        parsed = None
        # try parse full prompt; if it returns a time in future it's acceptable
        try:
            parsed = pendulum.parse(p, tz=tz)
        except Exception:
            parsed = None
        if parsed:
            # if parsed looks date-only, produce work hours
            if (parsed.hour == 0 and parsed.minute == 0) and not TIME_RE.search(p):
                start = parsed.set(hour=9, minute=0, second=0, microsecond=0)
                end = parsed.set(hour=18, minute=30, second=0, microsecond=0)
            else:
                start = parsed.set(second=0, microsecond=0)
                tentative_end = start.add(hours=4)
                workday_end = start.set(hour=18, minute=30, second=0, microsecond=0)
                end = tentative_end if tentative_end <= workday_end else workday_end
            # if parsed date/time is in the past, move ahead to next sensible day
            if start <= now:
                start = now.add(minutes=15)
                end = start.add(hours=4)
            return start, end
    except Exception:
        pass

    # default fallback: tomorrow 09:00 to days_ahead window
    start = now.add(days=1).set(hour=9, minute=0, second=0, microsecond=0)
    end = start.add(days=days_ahead).set(hour=18, minute=30)
    return start, end


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

# ---------- Recorder / upload endpoint ----------
@app.post("/upload-recording")
async def upload_recording(request: Request):
    """
    Accepts:
      - JSON { meeting, owner, participant_email, transcript, final (opt) }
      - multipart/form-data with 'meeting', 'owner', 'participant_email', 'audio' file, optional 'transcript'
    Behavior:
      - Stores per-meeting aggregated data in MOMS/<meeting_hash>.json (participants -> transcripts)
      - Saves uploaded audio blobs to data/uploads/
      - Returns {"status":"ok","mom_link": "...", "mom": {...}} when transcript(s) exist
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
                " ".join([t.get("text","") for t in rec.get("transcripts",[])]).strip()
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
            # form / multipart flow: upload audio blob + meeting + participant_email (+ optional transcript)
            form = await request.form()
            meeting = form.get("meeting")
            owner = form.get("owner")
            participant = form.get("participant_email") or owner or "unknown"
            audio_file = form.get("audio")
            transcript = form.get("transcript", "") or ""
            final = True
            ffinal = form.get("final")
            if ffinal is not None:
                final = str(ffinal).lower() not in ("0", "false", "no")

            if not meeting:
                return JSONResponse({"error":"no_meeting"}, status_code=400)
            if not audio_file and not transcript:
                # allow transcript-only in some flows, but require at least audio OR transcript
                return JSONResponse({"error":"no_audio_or_transcript"}, status_code=400)

            # save audio if present
            saved_audio_path = None
            if audio_file:
                fname = f"{uuid.uuid4()}_{getattr(audio_file, 'filename', 'upload.webm')}"
                dest = UPLOADS / fname
                contents = await audio_file.read()
                dest.write_bytes(contents)
                saved_audio_path = str(dest)

            # update meeting record (participants -> audio_files / transcripts)
            mid = meeting_hash(meeting)
            mom_file = MOMS / f"meeting_{mid}.json"
            if mom_file.exists():
                record = json.loads(mom_file.read_text(encoding="utf-8"))
            else:
                record = {"meeting": meeting, "owner": owner, "created_at": datetime.utcnow().isoformat() + "Z", "participants": {}}

            part = record["participants"].setdefault(participant, {"transcripts": [], "audio_files": []})
            if saved_audio_path:
                part["audio_files"].append({"path": saved_audio_path, "uploaded_at": datetime.utcnow().isoformat() + "Z"})
            if transcript:
                part["transcripts"].append({
                    "text": transcript,
                    "uploaded_at": datetime.utcnow().isoformat() + "Z",
                    "final": final
                })

            # persist the record
            mom_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

            # If there are transcripts in the meeting record, generate the MoM immediately
            combined_text = "\n\n".join(
                " ".join([t.get("text","") for t in rec.get("transcripts",[])]).strip()
                for rec in record["participants"].values()
            ).strip()

            if combined_text:
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
                resp = {"status":"ok", "saved": saved_audio_path or "", "mom_link": mom_link, "mom": mom_record}
                return JSONResponse(resp)

            # otherwise return saved audio info
            return JSONResponse({"status":"ok","saved": saved_audio_path or ""})

    except Exception as e:
        return JSONResponse({"error":"server_error","detail":str(e), "trace": traceback.format_exc()}, status_code=500)


# ---------- MoM view (unchanged) ----------
@app.get("/mom/{mom_id}", response_class=HTMLResponse)
def get_mom(mom_id: str):
    mom_file = MOMS / f"{mom_id}.json"
    if not mom_file.exists():
        return HTMLResponse("<h3>MoM not found</h3>", status_code=404)
    mom_record = json.loads(mom_file.read_text(encoding="utf-8"))

    # Build a small, pleasant HTML layout
    html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width,initial-scale=1"/>
      <title>MoM - {title}</title>
      <style>
        body{{font-family:Inter,system-ui,Arial;background:#f6f8fb;margin:0;padding:24px;color:#111}}
        .card{{max-width:980px;margin:24px auto;background:#fff;border-radius:12px;padding:22px;box-shadow:0 10px 30px rgba(20,20,40,0.06)}}
        h1{{margin:0 0 6px;font-size:20px}}
        .meta{{color:#6b7280;font-size:13px;margin-bottom:12px}}
        .section{{margin-top:18px}}
        .actions{{display:flex;gap:8px;flex-wrap:wrap}}
        .chip{{background:#eef2ff;color:#0f172a;padding:8px 10px;border-radius:8px;font-weight:600;font-size:13px}}
        .summary{{background:#fbfdff;padding:14px;border-radius:8px;border:1px solid #eef2ff}}
        .participant{{margin-top:12px;padding:12px;border-left:3px solid #eef2ff;background:#fff;border-radius:6px}}
        pre{{white-space:pre-wrap;font-family:inherit;background:#fafbff;padding:12px;border-radius:8px;border:1px solid #f1f5f9}}
        footer{{color:#7b7f86;margin-top:16px;font-size:13px}}
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Minutes of Meeting</h1>
        <div class="meta">Owner: {owner} • Created: {created}</div>

        <div class="section">
          <div style="display:flex;align-items:center;justify-content:space-between">
            <div>
              <div style="font-weight:700;margin-bottom:6px">Summary</div>
              <div class="summary">{summary}</div>
            </div>
            <div class="actions">
              <a href="{meeting}" target="_blank" class="chip">Open original meeting</a>
            </div>
          </div>
        </div>

        <div class="section">
          <div style="font-weight:700;margin-bottom:8px">Action Items</div>
          {actions_html}
        </div>

        <div class="section">
          <div style="font-weight:700;margin-bottom:8px">Participant transcripts & files</div>
          {participants_html}
        </div>

        <footer>Generated at {created}</footer>
      </div>
    </body>
    </html>
    """.strip()

    # prepare actions list HTML
    actions = mom_record.get('mom',{}).get('action_items', [])
    if actions:
        actions_html = "<ul style='line-height:1.6'>" + "".join(f"<li>{a}</li>" for a in actions) + "</ul>"
    else:
        actions_html = "<div style='color:#6b7280'>No action items detected.</div>"

    # participants
    parts_html = ""
    for p_email, pdata in mom_record.get('participants', {}).items():
        parts_html += f"<div class='participant'><div style='font-weight:700'>{p_email}</div>"
        for i, t in enumerate(pdata.get('transcripts', [])):
            text = t.get('text','(empty)')
            uploaded = t.get('uploaded_at','')
            parts_html += f"<div style='margin-top:8px'><div style='font-weight:600;font-size:13px'>Transcript #{i+1} <span style='color:#6b7280;font-weight:400;font-size:12px'>{uploaded}</span></div><pre>{text}</pre></div>"
        for af in pdata.get('audio_files', []):
            path = af.get('path')
            parts_html += f"<div style='margin-top:8px;color:#6b7280'>Audio file: <code>{path}</code></div>"
        parts_html += "</div>"

    html = html.format(
        title = mom_record.get('meeting','Meeting'),
        owner = mom_record.get('owner','-'),
        created = mom_record.get('created_at','-'),
        summary = mom_record.get('mom',{}).get('summary','(no summary)'),
        meeting = mom_record.get('meeting',''),
        actions_html = actions_html,
        participants_html = parts_html
    )
    return HTMLResponse(html)
