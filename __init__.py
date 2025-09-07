import hashlib
import json
import re
import time
import unicodedata

from aqt import gui_hooks, mw
from aqt.utils import tooltip

from .gpt_client import GPTError, judge_text
from .logic import get_cfg, map_to_ease

LAST_CALL = {}
LAST_REQUEST_TS = 0.0
CACHE = {}

MIN_INTERVAL_SEC = 2.0
ANTI_DUP_WINDOW_SEC = 6.0

_ws_re = re.compile(r"\s+")
_p_space = re.compile(r"\s+([,.:;!?])")


def _norm(s):
    s = (s or "").strip()
    s = unicodedata.normalize("NFC", s)
    s = _ws_re.sub(" ", s)
    s = _p_space.sub(r"\1", s)
    return s


def _get_field(note, name):
    try:
        return (note[name] or "").strip()
    except KeyError:
        return ""


def _cache_key(user, gold):
    h = hashlib.sha256()
    h.update(user.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(gold.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def on_js_message(handled, message, context):
    if not isinstance(message, str) or not message.startswith("judge:"):
        return handled

    try:
        payload = json.loads(message[len("judge:") :])
    except Exception:
        tooltip("Ошибка: некорректный payload")
        return (True, None)

    reviewer = mw.reviewer
    card = getattr(reviewer, "card", None)
    if not card:
        return (True, None)

    cfg = get_cfg()
    api_key = (cfg.get("openai_api_key") or "").strip()
    if not api_key:
        tooltip("⚠ Укажи openai_api_key в настройках аддона")
        return (True, None)

    note = card.note()
    gold = _get_field(note, cfg["fields"]["etalon_field"])
    user_text = (payload.get("text") or "").strip()
    if not user_text or not gold:
        return (True, None)

    user_text = _norm(user_text)
    gold = _norm(gold)

    if _norm(user_text) == _norm(gold):
        ease = 4
        tooltip("Exact match → Easy")
        if cfg.get("auto_answer", True):
            try:
                mw.reviewer._answerCard(ease)
            except Exception:
                pass
        return (True, None)

    global LAST_REQUEST_TS
    now = time.time()
    if now - LAST_REQUEST_TS < MIN_INTERVAL_SEC:
        tooltip("Слишком часто (анти-флуд)")
        return (True, None)
    LAST_REQUEST_TS = now

    short_key = f"{card.id}:{hashlib.sha1(user_text.encode('utf8')).hexdigest()}"
    if short_key in LAST_CALL and now - LAST_CALL[short_key] < ANTI_DUP_WINDOW_SEC:
        return (True, None)
    LAST_CALL[short_key] = now

    cache_key = (
        f"{card.id}:{hashlib.sha1((gold + '|' + user_text).encode('utf8')).hexdigest()}"
    )
    cache_ttl = int(cfg.get("cache_ttl_sec", 600))
    if cache_ttl > 0 and cache_key in CACHE:
        ts, v = CACHE[cache_key]
        if now - ts < cache_ttl:
            ease = map_to_ease(v.get("ease"))
            comment = (v.get("comment") or "").strip()
            label = (
                ["Again", "Hard", "Good", "Easy"][ease - 1]
                if ease in (1, 2, 3, 4)
                else "—"
            )
            tooltip(
                f"GPT(кеш): {comment} → {label}" if comment else f"GPT(кеш) → {label}"
            )
            if cfg.get("auto_answer", True) and ease in (1, 2, 3, 4):
                try:
                    mw.reviewer._answerCard(ease)
                except Exception:
                    pass
            return (True, None)

    requested_card_id = card.id

    def work():
        return judge_text(
            user_text=user_text,
            gold_text=gold,
            model=cfg.get("model", "gpt-4o-mini"),
            temperature=cfg.get("temperature", 0.3),
            max_tokens=cfg.get("max_tokens", 64),
            timeout=cfg.get("timeout_sec", 12),
            api_key=api_key,
            base_url=cfg.get("base_url", "https://api.openai.com/v1/"),
            retries=cfg.get("retries", 1),
            backoff_ms=cfg.get("backoff_ms", [400]),
        )

    def on_done(fut):
        cur = getattr(mw.reviewer, "card", None)
        if not cur or cur.id != requested_card_id:
            return
        try:
            verdict = fut.result()
        except Exception as e:
            tooltip(f"GPT error: {e}")
            return

        if cache_ttl > 0:
            CACHE[cache_key] = (time.time(), verdict)

        ease = map_to_ease(verdict.get("ease"))
        comment = (verdict.get("comment") or "").strip()
        label = (
            ["Again", "Hard", "Good", "Easy"][ease - 1] if ease in (1, 2, 3, 4) else "—"
        )
        tooltip(f"GPT: {comment} → {label}" if comment else f"GPT → {label}")

        if cfg.get("auto_answer", True) and ease in (1, 2, 3, 4):
            try:
                mw.reviewer._answerCard(ease)
            except Exception:
                pass

    mw.taskman.run_in_background(work, on_done)
    return (True, None)


def on_profile_loaded():
    cfg = get_cfg()
    if not (cfg.get("openai_api_key") or "").strip():
        tooltip("⚠ Укажи openai_api_key в настройках аддона")
    gui_hooks.webview_did_receive_js_message.append(on_js_message)


gui_hooks.profile_did_open.append(on_profile_loaded)
