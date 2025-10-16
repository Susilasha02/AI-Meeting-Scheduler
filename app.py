# app.py (updated to use pendulum for date parsing/timezones)
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
        # try to parse the time token with pendulum against today's date
        try:
            parsed = pendulum.parse(tm.group(1))
            return dt_date.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0, tzinfo=local_tz)
        except Exception:
            # defensive fallback simple parse
            m2 = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", tm.group(1), re.I)
            if m2:
                hour = int(m2.group(1))
                minute = int(m2.group(2) or 0)
                mer = (m2.group(3) or "").lower()
                if mer == "pm" and hour < 12:
                    hour += 12
                if mer == "am" and hour == 12:
                    hour = 0
                return dt_date.replace(hour=hour, minute=minute, second=0, microsecond=0, tzinfo=local_tz)

    if "morning" in p:
        return dt_date.replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=local_tz)
    if "afternoon" in p:
        return dt_date.replace(hour=14, minute=0, second=0, microsecond=0, tzinfo=local_tz)
    if "evening" in p or "night" in p:
        return dt_date.replace(hour=18, minute=0, second=0, microsecond=0, tzinfo=local_tz)

    # default
    return dt_date.replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=local_tz)

# ---------- Pendulum-backed parse helpers ----------
def parse_prompt_to_window(prompt: str, contacts_map: dict = None, default_duration_min: int = 45):
    """
    Pendulum-backed parser. Returns dict:
      {
        "attendees": [...],
        "duration_min": int,
        "window_start": rfc3339(start_utc_naive),
        "window_end": rfc3339(end_utc_naive)
      }
    Interprets times in LOCAL_TZ and returns RFC3339 UTC strings (same shape as original).
    """
    p = (prompt or "").strip()
    if contacts_map is None:
        contacts_map = {}

    duration_min = _parse_duration_minutes(p, default_min=default_duration_min)

    # attendees: allow @username mentions (Teams style) or emails in prompt
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

    # timezone tools (pendulum)
    try:
        tz = pendulum.timezone(LOCAL_TZ)
    except Exception:
        tz = pendulum.timezone("UTC")

    now_local = pendulum.now(tz)

    plow = p.lower()

    # 1) explicit weekday/time via regex shortcuts (fast path)
    # time token present?
    m_time = TIME_RE.search(p)

    # handle "next <weekday>"
    m_next = NEXT_WEEKDAY_RE.search(p)
    if m_next:
        weekday = m_next.group(1).lower()
        idx = WEEKDAY_MAP[weekday]
        days_ahead = (idx - now_local.day_of_week) % 7
        if days_ahead == 0:
            days_ahead = 7
        days_ahead += 7  # next-week
        date = now_local.add(days=days_ahead)
        # apply time preferences
        start_local_dt = _apply_time_preferences(datetime(date.year, date.month, date.day), p, zoneinfo.ZoneInfo(LOCAL_TZ))
        start_local = pendulum.instance(start_local_dt, tz)
        if start_local <= now_local:
            start_local = start_local.add(days=7)
        start_utc = start_local.in_timezone("UTC").naive()
        end_utc = (start_local.add(minutes=duration_min)).in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

    # handle "this <weekday>"
    m_this = THIS_WEEKDAY_RE.search(p)
    if m_this:
        weekday = m_this.group(1).lower()
        idx = WEEKDAY_MAP[weekday]
        days_ahead = (idx - now_local.day_of_week) % 7
        date = now_local.add(days=days_ahead)
        start_local_dt = _apply_time_preferences(datetime(date.year, date.month, date.day), p, zoneinfo.ZoneInfo(LOCAL_TZ))
        start_local = pendulum.instance(start_local_dt, tz)
        if start_local <= now_local:
            # push to next week's same weekday
            start_local = start_local.add(days=7)
        start_utc = start_local.in_timezone("UTC").naive()
        end_utc = (start_local.add(minutes=duration_min)).in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

    # 2) explicit time token (e.g., "11am", "3:30pm")
    if m_time:
        time_text = m_time.group(1)
        # try robust parsing via pendulum; fallback to manual parse
        try:
            parsed_time = pendulum.parse(time_text, strict=False)
            hour = parsed_time.hour
            minute = parsed_time.minute
        except Exception:
            mm = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_text, re.I)
            if mm:
                hour = int(mm.group(1))
                minute = int(mm.group(2) or 0)
                mer = (mm.group(3) or "").lower()
                if mer == "pm" and hour < 12:
                    hour += 12
                if mer == "am" and hour == 12:
                    hour = 0
            else:
                hour, minute = 9, 0

        # choose date: if 'tomorrow' mentioned use tomorrow, else today (and if passed, use next day)
        if "tomorrow" in plow:
            candidate = now_local.add(days=1).set(hour=hour, minute=minute, second=0, microsecond=0)
        else:
            candidate = now_local.set(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate <= now_local:
                candidate = candidate.add(days=1)

        start_local = candidate
        start_utc = start_local.in_timezone("UTC").naive()
        end_utc = (start_local.add(minutes=duration_min)).in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

    # 3) shortcuts: today / tomorrow / in X days
    if "today" in plow:
        # choose next available near now
        start_local = now_local.add(minutes=15)
        if start_local.hour >= 23:
            start_local = now_local.add(days=1).set(hour=9, minute=0, second=0, microsecond=0)
        start_utc = start_local.in_timezone("UTC").naive()
        end_utc = (start_local.add(minutes=duration_min)).in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339(end_utc)}

    if "tomorrow" in plow:
        tomorrow = now_local.add(days=1).set(hour=9, minute=0, second=0, microsecond=0)
        start_local = tomorrow
        if start_local <= now_local:
            start_local = start_local.add(days=1)
        start_utc = start_local.in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339((start_local.add(minutes=duration_min)).in_timezone('UTC').naive())}

    m_in = re.search(r"in\s+(\d+)\s+day", plow)
    if m_in:
        delta = int(m_in.group(1))
        target = now_local.add(days=delta).set(hour=9, minute=0, second=0, microsecond=0)
        start_utc = target.in_timezone("UTC").naive()
        return {"attendees": sorted(attendees) if attendees else ["primary"], "duration_min": duration_min, "window_start": rfc3339(start_utc), "window_end": rfc3339((target.add(minutes=duration_min)).in_timezone('UTC').naive())}

    # 4) final fallback: rolling window next 1-5 days
    fallback_start = now_local.add(days=1).set(hour=9, minute=0, second=0, microsecond=0)
    fallback_end = now_local.add(days=5).set(hour=18, minute=30, second=0, microsecond=0)
    fallback_start_utc = fallback_start.in_timezone("UTC").naive()
    fallback_end_utc = fallback_end.in_timezone("UTC").naive()
    return {
        "attendees": sorted(attendees) if attendees else ["primary"],
        "duration_min": duration_min,
        "window_start": rfc3339(fallback_start_utc),
        "window_end": rfc3339(fallback_end_utc)
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
    s_norm = s.strip()
    m = re.search(r"\b([A-Z][a-z]{1,30})\b\s+(?:will|shall|should|is to|is going to|please|must|needs to)\b", s)
    if m:
        name = m.group(1)
        rest = s[m.end():].strip()
        if rest:
            return name, _shorten_action(rest)
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
                # start_dt/end_dt are pendulum DateTimes (tz-aware); convert to UTC-aware datetimes
                try:
                    start = start_dt.astimezone(zoneinfo.ZoneInfo("UTC"))
                    end = end_dt.astimezone(zoneinfo.ZoneInfo("UTC"))
                except Exception:
                    # fallback convert via .in_timezone if underlying object is pendulum
                    try:
                        start = dparse.isoparse(start_dt.to_iso8601_string())
                        end = dparse.isoparse(end_dt.to_iso8601_string())
                    except Exception:
                        now = datetime.utcnow()
                        start = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
                        end   = (now + timedelta(days=5)).replace(hour=18, minute=30, second=0, microsecond=0)
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

# get_search_window_from_prompt uses pendulum and returns tz-aware start/end
def get_search_window_from_prompt(prompt: str, days_ahead: int = 7) -> Tuple[datetime, datetime]:
    """
    Determine a tz-aware local window for searching free slots based on natural prompt.
    Returns (start_local_dt, end_local_dt) both tz-aware in LOCAL_TZ.
    Handles 'today', 'tomorrow', weekdays, 'next week', and explicit dates.
    """
    try:
        tz = pendulum.timezone(LOCAL_TZ)
    except Exception:
        tz = pendulum.timezone("UTC")
    now = pendulum.now(tz)
    p = (prompt or "").lower().strip()

    # --- Handle simple relative words ---
    if "today" in p:
        start = now.set(hour=9, minute=0, second=0, microsecond=0)
        end = now.set(hour=18, minute=30, second=0, microsecond=0)
        if now.hour >= 18 and now.minute > 30:
            tomorrow = now.add(days=1)
            start = tomorrow.set(hour=9, minute=0, second=0, microsecond=0)
            end = tomorrow.set(hour=18, minute=30, second=0, microsecond=0)
        return start, end

    if "tomorrow" in p:
        tomorrow = now.add(days=1)
        start = tomorrow.set(hour=9, minute=0, second=0, microsecond=0)
        end = tomorrow.set(hour=18, minute=30, second=0, microsecond=0)
        return start, end

    # --- Handle "in X days" ---
    m_in = re.search(r"in\s+(\d+)\s+day", p)
    if m_in:
        delta = int(m_in.group(1))
        target = now.add(days=delta)
        start = target.set(hour=9, minute=0, second=0, microsecond=0)
        end = target.set(hour=18, minute=30, second=0, microsecond=0)
        return start, end

    # --- Handle "next week" ---
    if "next week" in p:
        start = (now.add(days=(7 - now.day_of_week) or 7)).set(hour=9, minute=0, second=0, microsecond=0)
        end = start.add(days=5).set(hour=18, minute=30)
        return start, end

    # --- Handle weekdays ("next Wednesday", "this Monday", plain "Friday") ---
    for word, idx in WEEKDAY_MAP.items():
        if f"next {word}" in p:
            target = now.add(days=((idx - now.day_of_week) % 7) + 7)
            start = target.set(hour=9, minute=0, second=0, microsecond=0)
            end = target.set(hour=18, minute=30, second=0, microsecond=0)
            return start, end
        if f"this {word}" in p:
            target = now.add(days=(idx - now.day_of_week) % 7)
            start = target.set(hour=9, minute=0, second=0, microsecond=0)
            end = target.set(hour=18, minute=30, second=0, microsecond=0)
            if start <= now:
                target = target.add(days=7)
                start = target.set(hour=9, minute=0, second=0, microsecond=0)
                end = target.set(hour=18, minute=30, second=0, microsecond=0)
            return start, end
        # plain weekday mention (e.g., "Wednesday") → nearest future weekday
        if re.search(rf"\b{word}\b", p):
            days_ahead_calc = (idx - now.day_of_week) % 7
            if days_ahead_calc == 0:
                days_ahead_calc = 7
            target = now.add(days=days_ahead_calc)
            start = target.set(hour=9, minute=0, second=0, microsecond=0)
            end = target.set(hour=18, minute=30, second=0, microsecond=0)
            return start, end

    # --- Explicit date/time parsing attempt with pendulum ---
    try:
        parsed = None
        try:
            parsed = pendulum.parse(p, tz=tz)
        except Exception:
            parsed = None
        if parsed:
            if (parsed.hour == 0 and parsed.minute == 0) and not TIME_RE.search(p):
                start = parsed.set(hour=9, minute=0, second=0, microsecond=0)
                end = parsed.set(hour=18, minute=30, second=0, microsecond=0)
            else:
                start = parsed.set(second=0, microsecond=0)
                tentative_end = start.add(hours=4)
                workday_end = start.set(hour=18, minute=30, second=0, microsecond=0)
                end = tentative_end if tentative_end <= workday_end else workday_end
            if start <= now:
                start = now.add(minutes=15)
                end = start.add(hours=4)
            return start, end
    except Exception:
        pass

    # --- Default fallback ---
    start = (now.add(days=1)).set(hour=9, minute=0, second=0, microsecond=0)
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

        ev_id = ev.get("id")
        html_link = ev.get("htmlLink") or ""
        organizer_email = None
        try:
            organizer_email = (ev.get("creator") or {}).get("email") or (ev.get("organizer") or {}).get("email")
        except Exception:
            organizer_email = None
        if not organizer_email:
            try:
                creds_local = creds
                organizer_email = getattr(creds_local, "client_id", None) or "organizer"
            except Exception:
                organizer_email = "organizer"

        encoded_meeting = urllib.parse.quote(html_link or ev_id or "", safe='')
        encoded_owner = urllib.parse.quote(organizer_email or "organizer", safe='')
        direct_recorder = f"{APP_BASE_URL}/ui/recorder.html?meeting={encoded_meeting}&owner={encoded_owner}"

        qp = urllib.parse.urlencode({
            "meeting": html_link or ev_id or "",
            "owner": organizer_email
        }, safe='')
        legacy_recorder = f"{APP_BASE_URL}/recorder/start?{qp}"

        new_description = (explanation or "") + "\n\nRecorder / Upload transcript (direct link):\n" \
                          f"{direct_recorder}\n\n(legacy compatibility link):\n{legacy_recorder}\n\n" \
                          "(Participants must click the link and consent to recording/transcription.)"
        try:
            patched = patch_event_description(creds, "primary", ev_id, new_description)
        except Exception:
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
                    "participants": {},
                }

            part = record["participants"].setdefault(participant, {"transcripts": [], "audio_files": []})
            part["transcripts"].append({
                "text": transcript,
                "uploaded_at": datetime.utcnow().isoformat() + "Z",
                "final": final
            })
            mom_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

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
                return JSONResponse({"error":"no_audio_or_transcript"}, status_code=400)

            saved_audio_path = None
            if audio_file:
                fname = f"{uuid.uuid4()}_{getattr(audio_file, 'filename', 'upload.webm')}"
                dest = UPLOADS / fname
                contents = await audio_file.read()
                dest.write_bytes(contents)
                saved_audio_path = str(dest)

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

            mom_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

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

    actions = mom_record.get('mom',{}).get('action_items', [])
    if actions:
        actions_html = "<ul style='line-height:1.6'>" + "".join(f"<li>{a}</li>" for a in actions) + "</ul>"
    else:
        actions_html = "<div style='color:#6b7280'>No action items detected.</div>"

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
