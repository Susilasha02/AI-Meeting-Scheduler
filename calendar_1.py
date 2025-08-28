# calender.py  (typo kept to match your filename)
from datetime import datetime
from typing import List, Dict
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

def _svc(creds: Credentials):
    return build("calendar", "v3", credentials=creds, cache_discovery=False)

def freebusy(creds: Credentials, emails: List[str], start_iso: str, end_iso: str) -> Dict[str, List[Dict]]:
    svc = _svc(creds)
    body = {"timeMin": start_iso, "timeMax": end_iso, "items": [{"id": e} for e in emails]}
    fb = svc.freebusy().query(body=body).execute()
    out={}
    for cal_id, cal in fb.get("calendars", {}).items():
        out[cal_id] = cal.get("busy", [])
    return out

def insert_event(creds: Credentials, calendar_id: str, subject: str,
                 start_iso: str, end_iso: str, attendees: List[str],
                 description: str = "", add_meet: bool = True):
    svc = _svc(creds)
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
