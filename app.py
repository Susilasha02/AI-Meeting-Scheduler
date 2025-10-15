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
from dateparser.search import search_dates

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

def parse_prompt_to_window(prompt: str, contacts_map: dict = None, default_duration_min: int = 45):
    """
    Main parser. Returns a dict with keys:
      {
        "attendees": [...],
        "duration_min": int,
        "window_start": rfc3339(start_utc_naive),
        "window_end": rfc3339(end_utc_naive)
      }
    """
    p = (prompt or "").strip()
    if contacts_map is None:
        contacts_map = {}

    duration_min = _parse_duration_minutes(p, default_min=default_duration_min)

    # attendees: allow @username mentions (Teams style) or emails in prompt
    attendees = set()
    # email regex (simple)
    email_re = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
    for e in email_re.findall(p):
        attendees.add(e)
    # @mentions like @raj or @raj@tenant
    mention_re = re.compile(r"@([a-zA-Z0-9_.-]+)")
    for m in mention_re.findall(p):
        key = m.lower()
        if key in contacts_map:
            attendees.add(contacts_map[key])
        else:
            # add raw mention form so you can resolve it later in Teams flow
            attendees.add("@" + m)

    # timezone tools
    try:
        local_tz = zoneinfo.ZoneInfo(LOCAL_TZ)
    except Exception:
        local_tz = zoneinfo.ZoneInfo("UTC")

    now_local = datetime.now(local_tz)
    # 1) try to find explicit date/time using dateparser.search.search_dates (future preference)
    try:
        sd = search_dates(p, settings={
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": datetime.utcnow(),
            "RETURN_AS_TIMEZONE_AWARE": False
        })
    except Exception:
        sd = None

    # If search_dates found something, prefer first meaningful match
    if sd:
        # choose the first match that looks date-like
        chosen_dt = None
        for matched_text, dt in sd:
            cleaned = (matched_text or "").strip()
            if len(cleaned) <= 1:
                continue
            chosen_dt = dt
            break
        if chosen_dt:
            # Normalize chosen_dt into local tz safely:
            try:
                if chosen_dt.tzinfo is None:
                    chosen_dt = chosen_dt.replace(tzinfo=None)  # keep naive for now
                    # If it's date-only midnight, we'll apply time preferences
                else:
                    chosen_dt = chosen_dt.astimezone(local_tz)
            except Exception:
                # defensive: if something odd, treat as naive
                chosen_dt = chosen_dt.replace(tzinfo=None)

            # if chosen_dt is date-only (hour==0/min==0) apply time prefs
            if getattr(chosen_dt, "hour", 0) == 0 and getattr(chosen_dt, "minute", 0) == 0 and "00:" not in str(chosen_dt):
                date_only = datetime(year=chosen_dt.year, month=chosen_dt.month, day=chosen_dt.day)
                start_local = _apply_time_preferences(date_only, p, local_tz)
            else:
                # chosen_dt might be naive datetime; interpret as local
                if chosen_dt.tzinfo is None:
                    start_local = chosen_dt.replace(tzinfo=local_tz)
                else:
                    start_local = chosen_dt.astimezone(local_tz)

            start_utc = start_local.astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
            end_utc = start_utc + timedelta(minutes=duration_min)
            return {
                "attendees": sorted(attendees) if attendees else ["primary"],
                "duration_min": duration_min,
                "window_start": rfc3339(start_utc),
                "window_end": rfc3339(end_utc)
            }

    # 2) explicit phrases next/this weekday handling:
    m_next = NEXT_WEEKDAY_RE.search(p)
    if m_next:
        weekday = m_next.group(1).lower()
        target = WEEKDAY_MAP[weekday]
        # enforce next-week (at_least_one_week_ahead=True)
        nextday = _next_weekday_after(now_local, target, at_least_one_week_ahead=True)
        start_local = _apply_time_preferences(nextday, p, local_tz)
        start_utc = start_local.astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = start_utc + timedelta(minutes=duration_min)
        return {
            "attendees": sorted(attendees) if attendees else ["primary"],
            "duration_min": duration_min,
            "window_start": rfc3339(start_utc),
            "window_end": rfc3339(end_utc)
        }

    m_this = THIS_WEEKDAY_RE.search(p)
    if m_this:
        weekday = m_this.group(1).lower()
        target = WEEKDAY_MAP[weekday]
        nextday = _next_weekday_after(now_local, target, at_least_one_week_ahead=False)
        start_local = _apply_time_preferences(nextday, p, local_tz)
        start_utc = start_local.astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = start_utc + timedelta(minutes=duration_min)
        return {
            "attendees": sorted(attendees) if attendees else ["primary"],
            "duration_min": duration_min,
            "window_start": rfc3339(start_utc),
            "window_end": rfc3339(end_utc)
        }

    # 3) shortcuts: today / tomorrow / in X days
    plow = p.lower()
    if "today" in plow:
        # choose next available near now
        start_local = now_local + timedelta(minutes=15)  # short buffer
        start_local = start_local.replace(second=0, microsecond=0)
        start_utc = start_local.astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = start_utc + timedelta(minutes=duration_min)
        return {
            "attendees": sorted(attendees) if attendees else ["primary"],
            "duration_min": duration_min,
            "window_start": rfc3339(start_utc),
            "window_end": rfc3339(end_utc)
        }
    if "tomorrow" in plow:
        tomorrow = (now_local + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        start_local = _apply_time_preferences(tomorrow, p, local_tz)
        start_utc = start_local.astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
        return {
            "attendees": sorted(attendees) if attendees else ["primary"],
            "duration_min": duration_min,
            "window_start": rfc3339(start_utc),
            "window_end": rfc3339(start_utc + timedelta(minutes=duration_min))
        }

    # 4) final fallback: rolling window next 1-5 days (existing behavior)
    fallback_start = (datetime.utcnow() + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    fallback_end = (datetime.utcnow() + timedelta(days=5)).replace(hour=18, minute=30, second=0, microsecond=0)
    return {
        "attendees": sorted(attendees) if attendees else ["primary"],
        "duration_min": duration_min,
        "window_start": rfc3339(fallback_start),
        "window_end": rfc3339(fallback_end)
    }

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

# ---------- summarizer for MoM ----------

import re
import nltk
from collections import defaultdict

# Ensure punkt exists (safe / quiet)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

def _split_sentences(text: str):
    if not text or not text.strip():
        return []
    txt = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    try:
        sents = nltk.sent_tokenize(txt)
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        # simple safe fallback
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", txt)
        return [p.strip() for p in parts if p.strip()]

def _extract_person_action(sentence: str):
    """
    Try to extract (person, short_action) from a sentence.
    Returns (person (or None), action_text).
    """
    s = sentence.strip()
    # normalize punctuation spaces
    s_norm = re.sub(r'\s+', ' ', s)

    # Patterns that explicitly name a person followed by a task:
    # e.g. "Asha your task is to clean the data"
    m = re.search(r"\b([A-Z][a-z]{1,24})\b[,:\s]{1,20}(?:your task is|your task|you will|you'll|you are|please|please\s+)?\s*(.*)", s_norm, re.I)
    if m and m.group(2).strip():
        person = m.group(1)
        action = m.group(2).strip()
        # trim trailing clauses that are too long: keep first clause
        action = re.split(r"(?:\band\b|\bthen\b|[;:])", action, maxsplit=1)[0].strip()
        return person, action

    # Patterns "John will do X", "Mark will be doing X", etc.
    m2 = re.search(r"\b([A-Z][a-z]{1,24})\b.*?\b(will|shall|is going to|is to|will be|should|must|needs to|need to)\b\s*(.*)", s_norm, re.I)
    if m2 and m2.group(3).strip():
        person = m2.group(1)
        action = m2.group(3).strip()
        action = re.split(r"(?:\band\b|\bthen\b|[;:])", action, maxsplit=1)[0].strip()
        return person, action

    # Imperatives that may include a direct object: "Please prepare the report by Friday"
    m3 = re.search(r"^(?:please\s+)?\b(assign|prepare|send|submit|create|make|share|prepare|compile|deliver|complete|finish|report|present|analyse|analyze|clean|model|process)\b\s+(.*)", s_norm, re.I)
    if m3:
        person = None
        action = (m3.group(1) + " " + m3.group(2)).strip()
        action = re.split(r"(?:\band\b|\bthen\b|[;:])", action, maxsplit=1)[0].strip()
        return person, action

    # Patterns like "Your task is to ...", no name
    m4 = re.search(r"(?:your task is to|your task is|you will need to|you need to|you are to)\s+(.*)", s_norm, re.I)
    if m4:
        action = m4.group(1).strip()
        action = re.split(r"(?:\band\b|\bthen\b|[;:])", action, maxsplit=1)[0].strip()
        return None, action

    # Last-resort: look for "X will" anywhere without a good tail
    m5 = re.search(r"\b([A-Z][a-z]{1,24})\b.*\b(will|should|must|needs to|need to)\b", s_norm, re.I)
    if m5:
        person = m5.group(1)
        # return the whole sentence trimmed if no small phrase found
        return person, s_norm

    # no person/action match; return None person and whole sentence as a candidate if it contains action keywords
    if re.search(r"\b(will|should|must|need to|need|task|assign|please|due|deadline|report|deliver|submit|prepare|complete|clean|model)\b", s_norm, re.I):
        return None, s_norm

    return None, None

def summarize_transcript(transcript: str, max_sentences: int = 6):
    """
    Returns: {"summary": str, "action_items": [str,...]}
    Improved action extraction: shorter, person-aware items.
    """
    if not transcript or not transcript.strip():
        return {"summary": "(no transcript provided)", "action_items": []}

    clean = re.sub(r"\s+", " ", transcript.replace("\r", " ").strip())
    sentences = _split_sentences(clean)
    if not sentences:
        sentences = [clean]

    # Build a short summary (prefer sentences containing meeting/project triggers)
    triggers = ["discuss", "meeting", "project", "goal", "topic", "update", "overview", "decide", "plan"]
    important = [s for s in sentences if any(t in s.lower() for t in triggers)]
    if not important:
        important = sentences[:max_sentences]
    summary = " ".join(important[:max_sentences]).strip()

    # Extract actions more carefully
    person_tasks = defaultdict(list)
    general_actions = []
    for s in sentences:
        person, action = _extract_person_action(s)
        if action:
            # sanitize action string
            action_clean = re.sub(r'\s+', ' ', action).strip().rstrip('.,;:')
            if person:
                person_tasks[person].append(action_clean)
            else:
                general_actions.append(action_clean)

    # If we found nothing using the patterns, fallback to scanning for lines with action keywords
    if not any(person_tasks.values()) and not general_actions:
        for s in sentences:
            if re.search(r"\b(will|should|must|need to|task|assign|please|due|deadline|report|deliver|submit|prepare|complete|clean|model)\b", s, re.I):
                # prefer short first clause
                short = re.split(r"(?:[;:]|\band\b|\bthen\b)", s, maxsplit=1)[0].strip()
                general_actions.append(short)

    # Format action items
    action_items = []
    for person, tasks in person_tasks.items():
        # dedupe
        seen = set()
        for t in tasks:
            if t not in seen:
                seen.add(t)
                action_items.append(f"{person}: {t}")
    # append general actions
    for ga in dict.fromkeys(general_actions).keys():
        action_items.append(f"General: {ga}")

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

# Helper get_search_window_from_prompt used by /suggest when prompt_text exists
def get_search_window_from_prompt(prompt: str, days_ahead: int = 7) -> Tuple[datetime, datetime]:
    """
    Determine a tz-aware local window for searching free slots based on natural prompt.
    Returns (start_local_dt, end_local_dt) both tz-aware in LOCAL_TZ.
    Handles 'today', 'tomorrow', weekdays, 'next week', and explicit dates.
    """
    try:
        tz = zoneinfo.ZoneInfo(LOCAL_TZ)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")
    now = datetime.now(tz)
    p = (prompt or "").lower().strip()

    # --- Handle simple relative words ---
    if "today" in p:
        start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        end = now.replace(hour=18, minute=30, second=0, microsecond=0)
        return start, end

    if "tomorrow" in p:
        tomorrow = now + timedelta(days=1)
        start = tomorrow.replace(hour=9, minute=0, second=0, microsecond=0)
        end = tomorrow.replace(hour=18, minute=30, second=0, microsecond=0)
        return start, end

    # --- Handle "in X days" ---
    m_in = re.search(r"in\s+(\d+)\s+day", p)
    if m_in:
        delta = int(m_in.group(1))
        target = now + timedelta(days=delta)
        start = target.replace(hour=9, minute=0, second=0, microsecond=0)
        end = target.replace(hour=18, minute=30, second=0, microsecond=0)
        return start, end

    # --- Handle "next week" ---
    if "next week" in p:
        start = (now + timedelta(days=(7 - now.weekday()))).replace(hour=9, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=5, hours=9, minutes=30)
        return start, end

    # --- Handle weekdays ("next Wednesday", "this Monday", plain "Friday") ---
    for word, idx in WEEKDAY_MAP.items():
        if f"next {word}" in p:
            target = now + timedelta((idx - now.weekday()) % 7 + 7)
            start = target.replace(hour=9, minute=0, second=0, microsecond=0)
            end = target.replace(hour=18, minute=30, second=0, microsecond=0)
            return start, end
        if f"this {word}" in p:
            target = now + timedelta((idx - now.weekday()) % 7)
            start = target.replace(hour=9, minute=0, second=0, microsecond=0)
            end = target.replace(hour=18, minute=30, second=0, microsecond=0)
            return start, end
        # plain weekday mention (e.g., "Wednesday") → nearest future weekday
        if re.search(rf"\b{word}\b", p):
            days_ahead_calc = (idx - now.weekday()) % 7
            if days_ahead_calc == 0:
                days_ahead_calc = 7
            target = now + timedelta(days=days_ahead_calc)
            start = target.replace(hour=9, minute=0, second=0, microsecond=0)
            end = target.replace(hour=18, minute=30, second=0, microsecond=0)
            return start, end

    # --- Handle explicit date with dateparser ---
    try:
        sd = search_dates(prompt or "", settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.utcnow()})
        if sd:
            dt = sd[0][1]
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            start = dt.replace(hour=9, minute=0, second=0, microsecond=0)
            end = start.replace(hour=18, minute=30)
            return start, end
    except Exception:
        pass

    # --- Default fallback ---
    start = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=days_ahead)
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


# ---------- Recorder / upload endpoint (updated) ----------
# ---------- Recorder / upload endpoint (updated) ----------
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



# ---------- MoM view (updated to show participants + transcripts) ----------
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

