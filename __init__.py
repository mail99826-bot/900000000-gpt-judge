import hashlib
import json
import re
import time

from aqt import gui_hooks, mw
from aqt.utils import tooltip

from .gpt_client import judge_text
from .logic import (
    get_cfg,
    is_deck_allowed,
    is_note_type_allowed,
    map_to_ease,
)

# локальные состояния
LAST_CALL = {}
LAST_REQUEST_TS = 0.0
CACHE = {}

# параметры локальной защиты
MIN_INTERVAL_SEC = 2.0
ANTI_DUP_WINDOW_SEC = 6.0

_ws_re = re.compile(r"\s+")


def _norm(s: str) -> str:
    # (2) жёсткий тримминг: пробелы, регистр
    s = (s or "").strip()
    s = _ws_re.sub(" ", s)
    return s


def _get_field(note, name: str) -> str:
    try:
        return (note[name] or "").strip()
    except KeyError:
        return ""


def _post_to_card(context, msg: str):
    try:
        context.eval(
            f"if(window._ankiAddonCallback) _ankiAddonCallback({json.dumps(msg)});"
        )
    except Exception:
        pass
    tooltip(msg)


def _in_review_state() -> bool:
    try:
        return (
            getattr(mw, "state", None) == "review"
            and getattr(mw, "reviewer", None) is not None
        )
    except Exception:
        return False


def on_js_message(handled, message, context):
    if not isinstance(message, str) or not message.startswith("judge:"):
        return handled
    if not _in_review_state():
        return (True, None)

    # payload
    try:
        payload = json.loads(message[len("judge:") :])
    except Exception:
        _post_to_card(context, "Ошибка: некорректный payload")
        return (True, None)

    reviewer = mw.reviewer
    card = getattr(reviewer, "card", None)
    if not card:
        return (True, None)

    cfg = get_cfg()

    # ключ
    api_key = (cfg.get("openai_api_key") or "").strip()
    if not api_key:
        _post_to_card(context, "⚠ Укажи openai_api_key в настройках аддона")
        return (True, None)

    # фильтры
    note = card.note()
    if not is_deck_allowed(card, cfg) or not is_note_type_allowed(note, cfg):
        return (True, None)

    # поля
    gold = _get_field(note, cfg["fields"]["etalon_field"])
    user_text = (payload.get("text") or "").strip()
    if not user_text:
        return (True, None)
    if not gold:
        _post_to_card(
            context, f"Поле эталона '{cfg['fields']['etalon_field']}' пусто/нет"
        )
        return (True, None)

    # (2) trimming + лимиты длины
    user_text = _norm(user_text)
    gold = _norm(gold)

    max_in = int(cfg.get("max_input_len", 800))
    if len(user_text) > max_in:
        user_text = user_text[:max_in]
        _post_to_card(context, f"Ввод длиннее {max_in} символов — обрезано")

    max_gold = int(cfg.get("max_gold_len", 800))
    if len(gold) > max_gold:
        gold = gold[:max_gold]
        _post_to_card(context, f"Эталон длиннее {max_gold} символов — обрезано")

    # (3) fast-path: точное совпадение после нормализации → Easy; пустяк → Again
    if user_text.lower() == gold.lower():
        ease = 4
        _post_to_card(context, "GPT(fast): exact match → Easy")
        if cfg.get("auto_answer", True):
            try:
                mw.reviewer._answerCard(ease)
            except Exception as e:
                tooltip(f"Не удалось ответить: {e}")
        return (True, None)

    if len(user_text.split()) <= 1 and len(gold.split()) >= 3:
        ease = 1
        _post_to_card(context, "GPT(fast): too short vs reference → Again")
        if cfg.get("auto_answer", True):
            try:
                mw.reviewer._answerCard(ease)
            except Exception as e:
                tooltip(f"Не удалось ответить: {e}")
        return (True, None)

    # анти-флуд: частота
    global LAST_REQUEST_TS
    now = time.time()
    if now - LAST_REQUEST_TS < MIN_INTERVAL_SEC:
        _post_to_card(context, "Слишком часто (анти-флуд). Подожди секунду.")
        return (True, None)
    LAST_REQUEST_TS = now

    # анти-дубль: тот же текст за короткое окно
    short_key = f"{card.id}:{hashlib.sha1(user_text.encode('utf8')).hexdigest()}"
    if short_key in LAST_CALL and now - LAST_CALL[short_key] < ANTI_DUP_WINDOW_SEC:
        return (True, None)
    LAST_CALL[short_key] = now

    # (4) кеш: карта + gold + user_text
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
            _post_to_card(
                context,
                f"GPT(кеш): {comment} → {label}" if comment else f"GPT(кеш) → {label}",
            )
            if cfg.get("auto_answer", True) and ease in (1, 2, 3, 4):
                try:
                    mw.reviewer._answerCard(ease)
                except Exception as e:
                    tooltip(f"Не удалось ответить: {e}")
            return (True, None)

    requested_card_id = card.id

    def work():
        return judge_text(
            user_text=user_text,
            gold_text=gold,
            lang=cfg.get("lang", "en"),
            strictness=cfg.get("strictness", "medium"),
            model=cfg.get("model", "gpt-4o-mini"),
            timeout=cfg.get("timeout_sec", 12),  # (7) таймаут контролируется конфигом
            api_key=api_key,
            base_url=cfg.get("base_url", "https://api.openai.com"),
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
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                _post_to_card(context, "⚠ Rate limit (429). Попробуй позже.")
            elif "auth" in msg or "401" in msg or "403" in msg:
                _post_to_card(context, "⚠ Ошибка авторизации (проверь API-ключ).")
            elif "timeout" in msg or "timed out" in msg:
                _post_to_card(
                    context, f"⌛ GPT не ответил за {cfg.get('timeout_sec', 12)} c."
                )
            elif "json" in msg:
                _post_to_card(context, "⚠ Не удалось разобрать ответ GPT (JSON).")
            else:
                _post_to_card(context, f"GPT недоступен: {str(e)}")
            return

        # сохранить в кеш
        if cache_ttl > 0:
            CACHE[cache_key] = (time.time(), verdict)

        ease = map_to_ease(verdict.get("ease"))
        if ease not in (1, 2, 3, 4):
            _post_to_card(context, "⚠ Не удалось определить оценку. Кнопка не нажата.")
            return

        comment = (verdict.get("comment") or "").strip()
        label = ["Again", "Hard", "Good", "Easy"][ease - 1]
        _post_to_card(
            context, f"GPT: {comment} → {label}" if comment else f"GPT → {label}"
        )

        if cfg.get("auto_answer", True):
            try:
                mw.reviewer._answerCard(ease)
            except Exception as e:
                tooltip(f"Не удалось ответить: {e}")

    mw.taskman.run_in_background(work, on_done)
    return (True, None)


def on_profile_loaded():
    cfg = get_cfg()
    if not (cfg.get("openai_api_key") or "").strip():
        tooltip("⚠ Укажи openai_api_key в настройках аддона")
    gui_hooks.webview_did_receive_js_message.append(on_js_message)


gui_hooks.profile_did_open.append(on_profile_loaded)
