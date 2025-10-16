# app.py (patched - weekday handling fixes)
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
import os
print(">>> STARTUP app.py loaded from:", __file__, "CWD:", os.getcwd())


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
# ---------- pendulum-aware helpers ----------
def to_rfc3339_with_tz(dt):
    """
    Accepts a pendulum.DateTime or datetime (aware or naive).
    Returns an ISO8601/RFC3339 string including timezone offset.
    """
    # Ensure we use a pendulum instance so .to_iso8601_string() is available
    try:
        pd = pendulum.instance(dt)
    except Exception:
        # fallback: treat naive as UTC
        pd = pendulum.instance(dt).in_timezone("UTC")
    return pd.to_iso8601_string()

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
    """
    Merge list of intervals where each interval is a dict {"start": ISO, "end": ISO}
    Use pendulum.parse to robustly handle offsets / tz-aware strings.
    Returns list of (pendulum_datetime_start, pendulum_datetime_end).
    """
    ivs = []
    for i in intervals:
        # defensive: ensure string keys exist
        try:
            s = pendulum.parse(i["start"])
            e = pendulum.parse(i["end"])
            ivs.append((s, e))
        except Exception:
            continue
    ivs.sort(key=lambda x: x[0])
    out = []
    for s, e in ivs:
        if not out or s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out

def compute_free(busy_by_person, window_start: pendulum.DateTime, window_end: pendulum.DateTime):
    """
    busy_by_person: list of lists of {"start": ISO, "end": ISO}
    window_start/window_end: pendulum tz-aware datetimes
    Returns: list of {"start": ISO_with_offset, "end": ISO_with_offset} using same tz as window_start
    """
    # flatten busy blocks
    all_busy = [b for person in busy_by_person for b in person]
    merged = _merge(all_busy)  # list of [pendulum_start, pendulum_end]
    free = []
    cur = window_start
    for s, e in merged:
        # normalize both s and e to window_start timezone for comparisons
        s_local = s.in_timezone(window_start.timezone)
        e_local = e.in_timezone(window_start.timezone)
        if s_local > cur:
            free.append((cur, s_local))
        cur = max(cur, e_local)
    if cur < window_end:
        free.append((cur, window_end))
    # return ISO strings (preserve tz offset)
    return [{"start": a.to_iso8601_string(), "end": b.to_iso8601_string()} for a, b in free]

def candidates(free_windows, duration_min=45, buffer_min=15, cap=6, step_min=15):
    """
    Generate candidate meeting slots from free windows.

    - step_min: how far the sliding window advances each iteration (smaller -> more candidates)
    - cap: max slots per day (increase to get more options across the day)
    - buffer_min: pre/post buffer enforced around meetings

    Returns up to 50 slots by default (trimmed at end).
    """
    out = []
    dur = timedelta(minutes=duration_min)
    buf = timedelta(minutes=buffer_min)
    daily = {}

    # iterate each free window and produce sliding slots with small steps
    for w in free_windows:
        s = pendulum.parse(w["start"])
        e = pendulum.parse(w["end"])
        # enforce buffers
        s = s.add(minutes=buffer_min)
        e = e.subtract(minutes=buffer_min)
        if s >= e:
            continue

        cur = s
        step = timedelta(minutes=step_min)
        while cur + dur <= e:
            day = cur.date()
            day_count = daily.get(day, 0)
            if day_count < cap:
                out.append({
                    "start": cur.to_iso8601_string(),
                    "end": (cur + dur).to_iso8601_string(),
                    "meta": {"buffers": {"pre": buffer_min, "post": buffer_min}}
                })
                daily[day] = day_count + 1
            cur = cur + step

    # cap global results
    return out[:50]


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
TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm))\b", re.I)


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

    
def parse_time_token(time_text: str):
    """
    Parse '2pm', '14:30', '2:15 pm' safely.
    Returns (hour, minute) or (None, None).
    This version ignores plain numbers like '30' that may appear in '30 min'.
    """
    if not time_text:
        return None, None

    # Ignore if token is followed by 'min' or 'mins' etc.
    if re.search(r"\b\d+\s*(?:min|mins|minutes)\b", time_text, re.I):
        return None, None

    m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", time_text, re.I)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        mer = (m.group(3) or "").lower()
        if mer == "pm" and hour < 12:
            hour += 12
        if mer == "am" and hour == 12:
            hour = 0
        if 0 <= hour < 24 and 0 <= minute < 60:
            return hour, minute
    return None, None



def _apply_time_preferences_pd(date_pd: pendulum.DateTime, prompt: str):
    """
    Given a pendulum date (tz-aware), set hour/minute according to prompt tokens.
    Returns a pendulum.DateTime in the same timezone.
    """
    p = (prompt or "").lower()
    # explicit time token
    tm = TIME_RE.search(prompt)
    if tm:
        hour, minute = parse_time_token(tm.group(1))
        if hour is None:
            hour, minute = 9, 0
        return date_pd.set(hour=hour, minute=minute, second=0, microsecond=0)

    if "morning" in p:
        return date_pd.set(hour=9, minute=0, second=0)
    if "afternoon" in p:
        return date_pd.set(hour=14, minute=0, second=0)
    if "evening" in p or "night" in p:
        return date_pd.set(hour=18, minute=0, second=0)

    # default
    return date_pd.set(hour=9, minute=0, second=0)


def parse_prompt_to_window(prompt: str, contacts_map: dict = None, default_duration_min: int = 45):
    """
    Pendulum-aware parser returning offset-preserving ISO windows.
    Now returns additional keys:
      - explicit_time: True/False (True if a concrete time token was present)
      - requested_start: ISO-with-offset (the exact requested start when explicit_time=True)
    """
    p_raw = (prompt or "").strip()
    p = p_raw or ""
    if contacts_map is None:
        contacts_map = {}

    # duration
    duration_min = _parse_duration_minutes(p, default_min=default_duration_min)

    # attendees (unchanged)
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

    # tz
    try:
        tz_pd = pendulum.timezone(LOCAL_TZ)
    except Exception:
        tz_pd = pendulum.timezone("UTC")
    now_local = pendulum.now(tz_pd)

    # normalized prompt
    plow = p.lower()
    norm_map = {
        r"\btomm?or?ow\b": "tomorrow",
        r"\btmrw\b": "tomorrow",
        r"\bto ?dayy?\b": "today",
        r"\bnextweek\b": "next week",
        r"\bnextwk\b": "next week",
        r"\bmonda?y\b": "monday",
        r"\btuesd?ay\b": "tuesday",
        r"\bwednesd?ay\b": "wednesday",
        r"\bthursd?ay\b": "thursday",
        r"\bfrida?y\b": "friday",
        r"\bsaturd?ay\b": "saturday",
        r"\bsunda?y\b": "sunday",
    }
    for pat, repl in norm_map.items():
        plow = re.sub(pat, repl, plow, flags=re.I)
    plow = re.sub(r"\s+", " ", plow).strip()

    # cleaned_for_weekday removes durations and stray numbers to avoid accidental parse consumption
    cleaned_for_weekday = DUR_RE.sub("", plow)
    cleaned_for_weekday = re.sub(r"\b\d+\s*(min|mins|minutes|h|hr|hrs)?\b", "", cleaned_for_weekday)
    cleaned_for_weekday = re.sub(r"\bfor\b", "", cleaned_for_weekday)
    cleaned_for_weekday = re.sub(r"\s+", " ", cleaned_for_weekday).strip()

    # debug: what do we see?
    time_token_match = TIME_RE.search(p) or TIME_RE.search(plow) or TIME_RE.search(cleaned_for_weekday)
    logger.info("parse_prompt_to_window DEBUG: raw=%r", p_raw)
    logger.info("parse_prompt_to_window DEBUG: normalized(plow)=%r", plow)
    logger.info("parse_prompt_to_window DEBUG: cleaned_for_weekday=%r", cleaned_for_weekday)
    logger.info("parse_prompt_to_window DEBUG: TIME_RE match (orig/plow/cleaned)=%r", bool(time_token_match))

    def _resp_from_start(start_local: pendulum.DateTime, branch_name: str = "unknown", explicit_time: bool = False):
        start_local = start_local.set(second=0, microsecond=0)
        if start_local <= now_local:
            # small nudge into the future to avoid past starts
            start_local = now_local.add(minutes=15).set(second=0, microsecond=0)
        end_local = start_local.add(minutes=duration_min)
        logger.info("parse_prompt_to_window CHOSEN: branch=%s start=%s end=%s explicit_time=%s",
                    branch_name, start_local.to_iso8601_string(), end_local.to_iso8601_string(), explicit_time)
        resp = {
            "attendees": sorted(attendees) if attendees else ["primary"],
            "duration_min": duration_min,
            "window_start": start_local.to_iso8601_string(),
            "window_end": end_local.to_iso8601_string()
        }
        if explicit_time:
            resp["explicit_time"] = True
            resp["requested_start"] = start_local.to_iso8601_string()
        else:
            resp["explicit_time"] = False
            resp["requested_start"] = None
        return resp

    # 1) tomorrow
    if "tomorrow" in plow:
        base = now_local.add(days=1)
        tm = TIME_RE.search(p) or TIME_RE.search(plow)
        if tm:
            hour, minute = parse_time_token(tm.group(1))
            hour = 9 if hour is None else hour
            minute = 0 if minute is None else minute
            base = base.set(hour=hour, minute=minute, second=0, microsecond=0)
            return _resp_from_start(base, "tomorrow", explicit_time=True)
        else:
            base = _apply_time_preferences_pd(base, p)
            return _resp_from_start(base, "tomorrow", explicit_time=False)

    # 2) weekday detection on cleaned text
    wd = re.search(r"\b(?:(next|this)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", cleaned_for_weekday, flags=re.I)
    if wd:
        prefix = (wd.group(1) or "").lower().strip()
        weekday_name = wd.group(2).lower()
        idx = WEEKDAY_MAP[weekday_name]  # 0..6

        # Use unambiguous Python weekday: Monday=0 ... Sunday=6
        today_idx = now_local.date().weekday()

        base_days = (idx - today_idx) % 7

        if prefix == "":
            days_ahead = base_days if base_days != 0 else 7
        elif prefix == "this":
            days_ahead = base_days
            if days_ahead == 0:
                tm2 = TIME_RE.search(p) or TIME_RE.search(plow)
                if tm2:
                    h, m = parse_time_token(tm2.group(1))
                    h = 9 if h is None else h
                    m = 0 if m is None else m
                    candidate = now_local.set(hour=h, minute=m, second=0, microsecond=0)
                else:
                    candidate = _apply_time_preferences_pd(now_local.start_of("day"), p)
                if candidate <= now_local:
                    days_ahead = 7
        else:  # prefix == "next"
            days_ahead = base_days + 7 if base_days < 7 else base_days
            if days_ahead == 0:
                days_ahead = 7

        if days_ahead < 0:
            days_ahead = 0

        target = now_local.add(days=days_ahead)

        tm = TIME_RE.search(p) or TIME_RE.search(plow)
        if tm:
            hour, minute = parse_time_token(tm.group(1))
            hour = 9 if hour is None else hour
            minute = 0 if minute is None else minute
            target = target.set(hour=hour, minute=minute, second=0, microsecond=0)
            return _resp_from_start(target, f"weekday({weekday_name},prefix={prefix})", explicit_time=True)
        else:
            target = _apply_time_preferences_pd(target, p)
            if target <= now_local:
                target = target.add(days=7)
            return _resp_from_start(target, f"weekday({weekday_name},prefix={prefix})", explicit_time=False)

    # 3) explicit pendulum parse
    try:
        parsed = None
        try:
            parsed = pendulum.parse(p, tz=tz_pd, strict=False)
        except Exception:
            parsed = None
        if parsed:
            parsed_local = parsed.in_timezone(tz_pd)
            has_time = bool(TIME_RE.search(p) or TIME_RE.search(plow))
            if not has_time:
                parsed_local = _apply_time_preferences_pd(parsed_local.start_of("day"), p)
                return _resp_from_start(parsed_local, "pendulum_parse", explicit_time=False)
            else:
                if parsed_local <= now_local:
                    parsed_local = parsed_local.add(days=1)
                return _resp_from_start(parsed_local, "pendulum_parse", explicit_time=True)
    except Exception:
        pass

    # 4) today
    if "today" in plow:
        base = now_local
        tm = TIME_RE.search(p) or TIME_RE.search(plow)
        if tm:
            hour, minute = parse_time_token(tm.group(1))
            hour = 9 if hour is None else hour
            minute = 0 if minute is None else minute
            base = base.set(hour=hour, minute=minute, second=0, microsecond=0)
            return _resp_from_start(base, "today", explicit_time=True)
        else:
            base = _apply_time_preferences_pd(base, p)
            return _resp_from_start(base, "today", explicit_time=False)

    # 5) in X days
    m_in = re.search(r"in\s+(\d+)\s+day", plow)
    if m_in:
        delta = int(m_in.group(1))
        target = now_local.add(days=delta)
        target = _apply_time_preferences_pd(target.set(hour=9, minute=0), p)
        return _resp_from_start(target, f"in_{delta}_days", explicit_time=False)

    # 6) next week
    if "next week" in plow:
        # compute days until next Monday using unambiguous date().weekday()
        today_idx = now_local.date().weekday()
        days_until_monday = (0 - today_idx) % 7 or 7
        start = now_local.add(days=days_until_monday + 7).set(hour=9, minute=0, second=0)
        return _resp_from_start(start, "next_week", explicit_time=False)

    # fallback
    fallback_start = now_local.add(days=1).set(hour=9, minute=0, second=0)
    fallback_end = fallback_start.add(days=4).set(hour=18, minute=30, second=0)
    logger.info("parse_prompt_to_window CHOSEN: branch=fallback start=%s end=%s prompt=%s",
                fallback_start.to_iso8601_string(), fallback_end.to_iso8601_string(), p_raw)
    return {
        "attendees": sorted(attendees) if attendees else ["primary"],
        "duration_min": duration_min,
        "window_start": fallback_start.to_iso8601_string(),
        "window_end": fallback_end.to_iso8601_string(),
        "explicit_time": False,
        "requested_start": None
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
    Use spaCy dependency parse to extract (subject, action) pairs.
    Better attempts to recover subjects when casing/punctuation is poor.
    """
    if not _nlp:
        return []
    # If text is all-lower, give spaCy a capitalized variant for better entity recognition,
    # but keep original for subtree extraction.
    doc_input = sent
    needs_title = sent.strip() and sent.strip() == sent.strip().lower()
    if needs_title:
        doc_for_ents = _nlp(sent.title())
    else:
        doc_for_ents = _nlp(sent)
    doc = _nlp(sent)

    actions = []
    for tok in doc:
        if tok.pos_ == "VERB" or tok.dep_ == "ROOT":
            subj = None
            # find subject token in dependency tree
            for ch in tok.children:
                if ch.dep_.startswith("nsubj"):
                    subj = ch.text
                    # try to see if spaCy produced an entity covering this subject in doc_for_ents
                    for ent in doc_for_ents.ents:
                        # ent start/end are indices inside doc_for_ents. Map by text compare fallback.
                        if ent.text.lower().find(ch.text.lower()) != -1:
                            subj = ent.text
                            break
                    break
            # assemble action phrase
            dobj_parts = []
            for ch in tok.children:
                if ch.dep_.endswith("obj") or ch.dep_ in ("dobj", "pobj", "attr"):
                    dobj_parts.append(" ".join([t.text for t in ch.subtree]))
            if dobj_parts:
                action_text = tok.lemma_ + " " + ", ".join(dobj_parts)
            else:
                subtree_words = [t.text for t in tok.subtree if t.i >= tok.i][:12]
                action_text = " ".join(subtree_words)
            action_text = _shorten_action(action_text)
            actions.append((subj, action_text))
    return actions

def _legacy_extract_person_action(s: str):
    """
    Heuristic fallback:
    - Accept lowercase names like 'john will ...' (case-insensitive)
    - Patterns:
        'john will ...'
        'please john: ...'
        'john, please ...'
    - If no person found but action triggers exist, return general action.
    """
    s_norm = s.strip()
    if not s_norm:
        return None, None

    # 1) 'Please xyz: do X' or 'please xyz, do X'
    m = re.search(r"(?:please\s+)?\b([A-Za-z][a-z]{0,30})\b[:\,]?\s+(.*)", s_norm, re.I)
    if m:
        candidate_name = m.group(1).strip()
        rest = m.group(2).strip()
        # require that rest contains an action trigger
        if re.search(r"\b(" + "|".join([re.escape(tok) for tok in _ACTION_TRIGGERS]) + r")\b", rest, re.I):
            return candidate_name.title(), _shorten_action(rest)

    # 2) 'xyz will ...' or 'xyz will ...' pattern
    m2 = re.search(r"\b([A-Za-z][a-z]{0,30})\b\s+(?:will|shall|should|is to|is going to|needs to|need to|please|must|should)\b\s*(.*)", s_norm, re.I)
    if m2:
        name = m2.group(1).strip()
        rest = m2.group(2).strip()
        if rest:
            return name.title(), _shorten_action(rest)

    # 3) If not name-specific, look for any action-trigger in sentence -> general action
    if re.search(r"\b(" + "|".join([re.escape(tok) for tok in _ACTION_TRIGGERS]) + r")\b", s_norm, re.I):
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

# ------ DEBUG endpoint (temporary) ------
import inspect, sys, json

@app.get("/_debug_parser")
def _debug_parser():
    """
    Debug: returns where parse_prompt_to_window is loaded from,
    what it returns for a canonical prompt, and some runtime info.
    Remove this endpoint after debugging.
    """
    try:
        # Import fresh reference (defensive)
        from app import parse_prompt_to_window  # noqa: E402
        src = inspect.getsourcefile(parse_prompt_to_window)
        try:
            out = parse_prompt_to_window("30 min meet tomorrow 3pm")
        except Exception as e:
            out = {"error_running_parser": str(e)}
        return {
            "source_file": src,
            "parser_result": out,
            "cwd": os.getcwd(),
            "sys_path": sys.path[:8],        # show first entries only (safe)
            "python": sys.version,
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
# ---------------------------------------


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
    """
    Debugging-wrapper around the normal suggest flow that defensively reloads parser
    and enforces strict behavior when an explicit time was present in the user's prompt.
    """
    try:
        import importlib, sys
        if "app" in sys.modules:
            importlib.reload(sys.modules["app"])

        from app import parse_prompt_to_window  # noqa: E402
        prompt = (body.get("prompt","") or "").strip()
        contacts_map = load_contacts()

        cfg = parse_prompt_to_window(prompt, contacts_map)
        logger.info("nlp_suggest: parser returned: %s", cfg)

        # If user asked for an explicit time (e.g. "tomorrow 3pm") we tighten the window
        # so empty calendars still propose slots at/near the requested time.
        if cfg.get("explicit_time") and cfg.get("requested_start"):
            # requested_start is ISO with offset in user's local tz; parse into pendulum
            try:
                req_pd = pendulum.parse(cfg["requested_start"])
                # keep the provided start as the lower bound
                window_start_pd = req_pd
                # set a small upper bound: requested_start + 3 hours (adjustable)
                window_end_pd = req_pd.add(hours=3)
                cfg["window_start"] = window_start_pd.to_iso8601_string()
                cfg["window_end"] = window_end_pd.to_iso8601_string()
                logger.info("nlp_suggest: tightened window around requested_time: %s -> %s",
                            cfg["window_start"], cfg["window_end"])
            except Exception:
                # if parse fails, keep parser window as-is (defensive)
                logger.exception("nlp_suggest: failed to parse requested_start; using original window")

        cfg.setdefault("buffer_min", 15)
        # call core suggest (body-shaped config)
        return suggest(cfg)

    except Exception as e:
        logger.exception("nlp_suggest debug exception")
        return JSONResponse({"error":"server_error","detail":str(e)}, status_code=500)


## ------------------ Updated /suggest endpoint (drop-in replacement) ------------------
@app.post("/suggest")
def suggest(body: dict = Body(default={})):
    """
    Body can include:
      - duration_min, buffer_min
      - attendees: list of emails (or ["primary"])
      - window_start / window_end (RFC3339)
      - OR: prompt and today_only: true/false
      - optionally: explicit_time (bool) and requested_start (ISO string) to request strict behavior
    Behavior:
      - If explicit_time + requested_start present, prefer slots near requested_start (STRICT window).
      - If prompt provided and no explicit window, parse it with parse_prompt_to_window() to derive window/flags.
    """
    creds = load_creds(ORGANIZER_ID)
    if not creds:
        return JSONResponse({"error": "not_connected"}, status_code=401)

    try:
        # ---------------- basic inputs ----------------
        duration_min = int(body.get("duration_min", 45))
        buffer_min = int(body.get("buffer_min", 15))

        attendees_raw = body.get("attendees", ["primary"])
        attendees = [a for a in attendees_raw if a == "primary" or "@" in a]
        if not attendees:
            attendees = ["primary"]

        prompt_text = (body.get("prompt", "") or "").strip()
        today_only_flag = bool(body.get("today_only", False))

        # ---------------- derive window: prefer explicit window_start/window_end ----------------
        start_iso = body.get("window_start")
        end_iso = body.get("window_end")

        # If caller didn't supply explicit window, but supplied prompt, use the parser to derive window + flags
        explicit_flag = bool(body.get("explicit_time", False))
        requested_start_iso = body.get("requested_start")  # may be None

        if (not start_iso or not end_iso) and prompt_text:
            # call parser to derive window and possibly explicit_time/requested_start
            try:
                parsed_cfg = parse_prompt_to_window(prompt_text, load_contacts(), default_duration_min=duration_min)
                # parser returns window_start/window_end in local tz ISO and explicit_time/requested_start flags
                start_iso = start_iso or parsed_cfg.get("window_start")
                end_iso = end_iso or parsed_cfg.get("window_end")
                # only override explicit flags if not present in body
                if "explicit_time" not in body:
                    explicit_flag = bool(parsed_cfg.get("explicit_time", False))
                if "requested_start" not in body:
                    requested_start_iso = parsed_cfg.get("requested_start")
                logger.info("suggest: parse_prompt_to_window produced explicit_time=%s requested_start=%s window=%s->%s",
                            explicit_flag, requested_start_iso, start_iso, end_iso)
            except Exception:
                logger.exception("suggest: parse_prompt_to_window failed; falling back to defaults")

        # If still no explicit start/end, compute default rolling window (local tz)
        if start_iso and end_iso:
            start_pd = pendulum.parse(start_iso)
            end_pd = pendulum.parse(end_iso)
            try:
                start = start_pd.in_timezone(LOCAL_TZ)
                end = end_pd.in_timezone(LOCAL_TZ)
            except Exception:
                start = start_pd
                end = end_pd
        else:
            tz_local = pendulum.timezone(LOCAL_TZ)
            now_local = pendulum.now(tz_local)
            start = now_local.add(days=1).set(hour=9, minute=0, second=0)
            end = now_local.add(days=5).set(hour=18, minute=30, second=0)
            start_iso = start.to_iso8601_string()
            end_iso = end.to_iso8601_string()

        # prepare UTC strings for Google freebusy
        start_iso_for_api = pendulum.instance(start).in_timezone("UTC").to_iso8601_string()
        end_iso_for_api = pendulum.instance(end).in_timezone("UTC").to_iso8601_string()

        logger.info("suggest: attendees=%s start(local)=%s end(local)=%s prompt=%s explicit_time=%s requested_start=%s",
                    attendees, start.to_iso8601_string(), end.to_iso8601_string(), prompt_text, explicit_flag, requested_start_iso)
        logger.info("FINAL search window (local): %s → %s", start.to_iso8601_string(), end.to_iso8601_string())

        # --- SAFETY PATCH: auto-expand short windows so scheduler can always generate slots ---
        window_minutes = (end - start).total_minutes() if hasattr(end, 'total_minutes') else (end - start).total_seconds() / 60.0
        if window_minutes < 180:
            # if window shorter than 3 hours
            old_end = end
            end = start.add(hours=4)
            logger.info("Auto-expanded short window from %s to %s (was %.1f min)", old_end.to_iso8601_string(), end.to_iso8601_string(), window_minutes)
# --- END PATCH ---


        # ---------------- freebusy call ----------------
        fb = freebusy(creds, attendees, start_iso_for_api, end_iso_for_api)
        logger.info("RAW freebusy: %s", json.dumps(fb))

        # prepare busy lists (raw strings)
        busy_lists = [[{"start": b["start"], "end": b["end"]} for b in fb.get(cal, [])] for cal in fb]

        # convert start/end to pendulum UTC datetimes
        start_pd_utc = pendulum.parse(start_iso_for_api)   # tz-aware UTC
        end_pd_utc = pendulum.parse(end_iso_for_api)

        # compute free windows
        free = compute_free(busy_lists, start_pd_utc, end_pd_utc)

        # generate candidate slots
        raw_candidates = candidates(free, duration_min=duration_min, buffer_min=buffer_min)

        # ---------------- strict filtering when explicit requested time present ----------------
                # ---------------- strict filtering when explicit requested time present ----------------
        cands = []
        now_utc = pendulum.now("UTC")
        effective_start = start_pd_utc if start_pd_utc > now_utc else now_utc

        if explicit_flag and requested_start_iso:
            # progressive relaxation strategy
            try:
                req_pd_local = pendulum.parse(requested_start_iso)
                req_pd_utc = req_pd_local.in_timezone("UTC")

                # 1) try tight strict window (requested_start -> +strict_hours)
                strict_hours = float(body.get("strict_window_hours", 2.0))
                strict_start_utc = req_pd_utc
                strict_end_utc = req_pd_utc.add(hours=strict_hours)
                logger.info("suggest: strict filtering active. requested_start(utc)=%s strict_end(utc)=%s", strict_start_utc.to_iso8601_string(), strict_end_utc.to_iso8601_string())

                cands = [c for c in raw_candidates if (pendulum.parse(c["start"]) >= strict_start_utc and pendulum.parse(c["start"]) <= strict_end_utc)]

                # 2) Relax 1 — widen the strict window up to a max (configurable)
                if not cands:
                    max_relax_hours = float(body.get("max_relax_hours", 12.0))
                    widen_hours = min(max_relax_hours, max(4.0, strict_hours * 3))
                    wide_start = req_pd_utc.subtract(hours=widen_hours/2)
                    wide_end = req_pd_utc.add(hours=widen_hours/2)
                    logger.info("suggest: widening strict window to %s - %s (hours=%s)", wide_start.to_iso8601_string(), wide_end.to_iso8601_string(), widen_hours)
                    cands = [c for c in raw_candidates if (pendulum.parse(c["start"]) >= wide_start and pendulum.parse(c["start"]) <= wide_end)]

                # 3) Relax 2 — any candidate at or after requested_start
                if not cands:
                    logger.info("suggest: strict window returned no candidates; relaxing to candidates >= requested_start")
                    cands = [c for c in raw_candidates if pendulum.parse(c["start"]) >= req_pd_utc]

                # 4) Relax 3 — expand to the whole working day of the requested date (local 09:00-18:30)
                if not cands:
                    try:
                        local_req = req_pd_local.in_timezone(LOCAL_TZ)
                        day_start = local_req.set(hour=9, minute=0, second=0, microsecond=0)
                        day_end = local_req.set(hour=18, minute=30, second=0, microsecond=0)
                        day_start_utc = day_start.in_timezone("UTC")
                        day_end_utc = day_end.in_timezone("UTC")
                        logger.info("suggest: relaxing to full working day window %s -> %s", day_start_utc.to_iso8601_string(), day_end_utc.to_iso8601_string())
                        cands = [c for c in raw_candidates if (pendulum.parse(c["start"]) >= day_start_utc and pendulum.parse(c["start"]) <= day_end_utc)]
                    except Exception:
                        logger.exception("suggest: failed to expand to working day window")

                # 5) Relax 4 — if still none and freebusy shows calendars completely free, generate fallback synthetic slots across the day
                if not cands:
                    all_free = all(len(v) == 0 for v in fb.values())
                    if all_free:
                        # create synthetic candidates across the requested date (local) to show the user options
                        synthetic = []
                        try:
                            local_mid = req_pd_local.in_timezone(LOCAL_TZ).set(hour=9, minute=0, second=0, microsecond=0)
                            end_local_day = local_mid.set(hour=18, minute=0)
                            cur = local_mid
                            step = timedelta(minutes=30)
                            dur = timedelta(minutes=duration_min)
                            while cur + dur <= end_local_day:
                                synthetic.append({
                                    "start": cur.to_iso8601_string(),
                                    "end": (cur + dur).to_iso8601_string(),
                                    "meta": {"buffers": {"pre": buffer_min, "post": buffer_min}}
                                })
                                cur = cur + step
                        except Exception:
                            synthetic = []
                        if synthetic:
                            logger.info("suggest: returning %d synthetic candidates because calendars are empty", len(synthetic))
                            cands = synthetic

            except Exception:
                logger.exception("suggest: failed strict requested_start parsing; falling back to standard behavior")
                cands = [c for c in raw_candidates if pendulum.parse(c["start"]) >= start_pd_utc]
        else:
            # normal behaviour: proposals start at or after the requested start (start_pd_utc)
            cands = [c for c in raw_candidates if pendulum.parse(c["start"]) >= start_pd_utc]

        # If no candidates after filtering, relax progressively to effective_start and then to raw candidates
        if not cands:
            logger.info("suggest: no candidates after requested filtering; relaxing to effective_start >= now or window start")
            cands = [c for c in raw_candidates if pendulum.parse(c["start"]) >= effective_start]

        if not cands:
            logger.info("suggest: still none after relax; returning raw candidates (last resort)")
            cands = raw_candidates


        # ---------------- scoring & response ----------------
        scored = []
        for sl in cands:
            score, comp = _fairness_score(sl, fb)
            scored.append((score, comp, sl))

        top = sorted(scored, key=lambda x: x[0], reverse=True)[:3]
        resp = []
        for score, comp, sl in top:
            start_dt = pendulum.parse(sl["start"])
            end_dt = pendulum.parse(sl["end"])
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
        logger.exception("Exception in suggest: %s", traceback.format_exc())
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
        today_idx = now.date().weekday()
        start = (now.add(days=(7 - today_idx) or 7)).set(hour=9, minute=0, second=0, microsecond=0)
        end = start.add(days=5).set(hour=18, minute=30)
        return start, end

    # --- Handle weekdays ("next Wednesday", "this Monday", plain "Friday") ---
    for word, idx in WEEKDAY_MAP.items():
        if f"next {word}" in p:
            today_idx = now.date().weekday()
            target = now.add(days=((idx - today_idx) % 7) + 7)
            start = target.set(hour=9, minute=0, second=0, microsecond=0)
            end = target.set(hour=18, minute=30, second=0, microsecond=0)
            return start, end
        if f"this {word}" in p:
            today_idx = now.date().weekday()
            target = now.add(days=(idx - today_idx) % 7)
            start = target.set(hour=9, minute=0, second=0, microsecond=0)
            end = target.set(hour=18, minute=30, second=0, microsecond=0)
            if start <= now:
                target = target.add(days=7)
                start = target.set(hour=9, minute=0, second=0, microsecond=0)
                end = target.set(hour=18, minute=30, second=0, microsecond=0)
            return start, end
        # plain weekday mention (e.g., "Wednesday") → nearest future weekday
        if re.search(rf"\b{word}\b", p):
            today_idx = now.date().weekday()
            days_ahead_calc = (idx - today_idx) % 7
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
