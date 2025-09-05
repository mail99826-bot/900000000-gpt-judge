from aqt import mw, gui_hooks
from aqt.utils import tooltip
import json, hashlib, time

from .logic import get_cfg, map_to_ease
from .gpt_client import judge_text

LAST_CALL = {}

def on_js_message(handled, message, context):
    if not isinstance(message, str) or not message.startswith("judge:"):
        return handled

    try:
        payload = json.loads(message[len("judge:"):])
    except Exception:
        tooltip("Ошибка: некорректный payload")
        return (True, None)

    card = getattr(mw.reviewer, "card", None)
    if not card:
        return (True, None)

    cfg = get_cfg()
    note = card.note()
    gold = (note.get(cfg["fields"]["etalon_field"], "") or "").strip()
    user_text = (payload.get("text") or "").strip()
    if not user_text:
        return (True, None)

    key = f"{card.id}:{hashlib.sha1(user_text.encode('utf8')).hexdigest()}"
    now = time.time()
    if key in LAST_CALL and now - LAST_CALL[key] < 6:
        return (True, None)
    LAST_CALL[key] = now

    try:
        verdict = judge_text(
            user_text=user_text,
            gold_text=gold,
            lang=cfg.get("lang","en"),
            strictness=cfg.get("strictness","medium"),
            model=cfg.get("model","gpt-4o-mini"),
            timeout=cfg.get("timeout_sec", 12),
            api_key=cfg.get("openai_api_key","")
        )
    except Exception as e:
        tooltip(f"GPT недоступен: {e}")
        return (True, None)

    ease = map_to_ease(verdict.get("ease"))
    comment = (verdict.get("comment") or "").strip()
    if comment:
        label = ["Again","Hard","Good","Easy"][ease-1] if ease in (1,2,3,4) else "—"
        tooltip(f"GPT: {comment} → {label}")

    if cfg.get("auto_answer", True) and ease in (1,2,3,4):
        mw.reviewer._answerCard(ease)

    return (True, None)

def on_profile_loaded():
    gui_hooks.webview_did_receive_js_message.append(on_js_message)

gui_hooks.profile_did_open.append(on_profile_loaded)
