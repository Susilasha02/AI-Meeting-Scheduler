# builder.py
from datetime import datetime, timedelta

def _merge(intervals):
    ivs = sorted([(datetime.fromisoformat(i["start"]), datetime.fromisoformat(i["end"])) for i in intervals])
    out=[]
    for s,e in ivs:
        if not out or s>out[-1][1]: out.append([s,e])
        else: out[-1][1]=max(out[-1][1], e)
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
