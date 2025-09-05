import hashlib
import json
import time

from aqt import gui_hooks, mw
from aqt.utils import tooltip

from .gpt_client import judge_text
from .logic import get_cfg, map_to_ease

LAST_CALL = {}


def _post_verdict_in_webview(context, msg):
    # показать на карточке (если подключён колбэк), и тултипом
    try:
        context.eval(
            f"if(window._ankiAddonCallback) _ankiAddonCallback({json.dumps(msg)});"
        )
    except Exception:
        pass
    tooltip(msg)


def on_js_message(handled, message, context):
    if not isinstance(message, str) or not message.startswith("judge:"):
        return handled

    try:
        payload = json.loads(message[len("judge:") :])
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

    # анти-дубль Enter
    key = f"{card.id}:{hashlib.sha1(user_text.encode('utf8')).hexdigest()}"
    now = time.time()
    if key in LAST_CALL and now - LAST_CALL[key] < 6:
        return (True, None)
    LAST_CALL[key] = now

    # --- фоновая задача ---
    def work():
        return judge_text(
            user_text=user_text,
            gold_text=gold,
            lang=cfg.get("lang", "en"),
            strictness=cfg.get("strictness", "medium"),
            model=cfg.get("model", "gpt-4o-mini"),
            timeout=cfg.get("timeout_sec", 12),
            api_key=cfg.get("openai_api_key", ""),
        )

    def on_done(fut):
        try:
            verdict = fut.result()
        except Exception as e:
            _post_verdict_in_webview(context, f"GPT недоступен: {e}")
            return

        ease = map_to_ease(verdict.get("ease"))
        comment = (verdict.get("comment") or "").strip()
        label = (
            ["Again", "Hard", "Good", "Easy"][ease - 1] if ease in (1, 2, 3, 4) else "—"
        )
        msg = f"GPT: {comment} → {label}" if comment else f"GPT → {label}"
        _post_verdict_in_webview(context, msg)

        if cfg.get("auto_answer", True) and ease in (1, 2, 3, 4):
            mw.reviewer._answerCard(ease)

    mw.taskman.run_in_background(work, on_done)
    return (True, None)


def on_profile_loaded():
    gui_hooks.webview_did_receive_js_message.append(on_js_message)


gui_hooks.profile_did_open.append(on_profile_loaded)
