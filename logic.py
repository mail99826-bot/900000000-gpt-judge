from aqt import mw

DEFAULT_CFG = {
    "openai_api_key": "",
    "model": "gpt-4o-mini",
    "lang": "en",
    "strictness": "medium",
    "timeout_sec": 12,
    "auto_answer": False,
    "fields": {
        "sentence_field": "Front",
        "etalon_field": "Back"
    }
}

def get_cfg():
    cfg = mw.addonManager.getConfig(__name__) or {}
    merged = DEFAULT_CFG.copy()
    merged.update(cfg)
    merged["fields"] = {**DEFAULT_CFG["fields"], **cfg.get("fields", {})}
    return merged

def map_to_ease(e):
    try:
        e = int(e)
    except Exception:
        return 0
    return e if e in (1,2,3,4) else 0
