# store.py
import json, os
STORE = "token_store.json"

def save_user_creds(user_id: str, creds_dict: dict):
    data = {}
    if os.path.exists(STORE):
        with open(STORE, "r") as f:
            try: data = json.load(f)
            except: data = {}
    data[user_id] = creds_dict
    with open(STORE, "w") as f:
        json.dump(data, f, indent=2)

def load_user_creds(user_id: str):
    if not os.path.exists(STORE): return None
    with open(STORE, "r") as f:
        try: data = json.load(f)
        except: return None
    return data.get(user_id)
