import hashlib
import json
import time

from aqt import gui_hooks, mw
from aqt.utils import tooltip

from .gpt_client import judge_text
from .logic import get_cfg, is_deck_allowed, is_note_type_allowed, map_to_ease

# ===== Настройки локальных защит =====
MAX_INPUT_LEN = 800  # (2) ограничение длины ввода
MIN_INTERVAL_SEC = 2.0  # (3) анти-флуд: не чаще одного запроса раз в N сек
ANTI_DUP_WINDOW_SEC = 6.0  # анти-дубль по тому же тексту

LAST_CALL = {}  # key=(card.id+hash(user_text)) -> ts
LAST_REQUEST_TS = 0.0  # (3) анти-флуд по времени


def _get_field(note, name: str) -> str:
    try:
        return (note[name] or "").strip()
    except KeyError:
        return ""


def _post_to_card(context, msg: str):
    # показать на карточке (если webview жив) и в тултипе
    try:
        context.eval(
            f"if(window._ankiAddonCallback) _ankiAddonCallback({json.dumps(msg)});"
        )
    except Exception:
        pass
    tooltip(msg)


def _in_review_state() -> bool:
    # (6) не работать в Preview/Browse — только в режиме ревью
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

    # (6) блокируем вне режима ревью
    if not _in_review_state():
        return (True, None)

    # распарсим payload
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

    # (5) проверка наличия ключа заранее
    api_key = (cfg.get("openai_api_key") or "").strip()
    if not api_key:
        _post_to_card(context, "⚠ Укажи openai_api_key в настройках аддона")
        return (True, None)

    # фильтры: по желанию ограничиваем колоды/типы заметок
    note = card.note()
    if not is_deck_allowed(card, cfg) or not is_note_type_allowed(note, cfg):
        return (True, None)

    gold = _get_field(note, cfg["fields"]["etalon_field"])
    user_text = (payload.get("text") or "").strip()

    # защита: пустой ввод / пустой эталон
    if not user_text:
        return (True, None)
    if not gold:
        _post_to_card(
            context, f"Поле эталона '{cfg['fields']['etalon_field']}' пусто/нет"
        )
        return (True, None)

    # (2) ограничение длины ввода
    if len(user_text) > MAX_INPUT_LEN:
        user_text = user_text[:MAX_INPUT_LEN]
        _post_to_card(context, f"Ввод длиннее {MAX_INPUT_LEN} символов — обрезано")

    # (3) анти-флуд по времени
    global LAST_REQUEST_TS
    now = time.time()
    if now - LAST_REQUEST_TS < MIN_INTERVAL_SEC:
        _post_to_card(context, "Слишком часто (анти-флуд). Подожди секунду.")
        return (True, None)
    LAST_REQUEST_TS = now

    # анти-дубль по тому же тексту
    key = f"{card.id}:{hashlib.sha1(user_text.encode('utf8')).hexdigest()}"
    if key in LAST_CALL and now - LAST_CALL[key] < ANTI_DUP_WINDOW_SEC:
        return (True, None)
    LAST_CALL[key] = now

    # (1) запомним карту, для которой уехал запрос
    requested_card_id = card.id

    # Фоновая задача: вызов GPT
    def work():
        return judge_text(
            user_text=user_text,
            gold_text=gold,
            lang=cfg.get("lang", "en"),
            strictness=cfg.get("strictness", "medium"),
            model=cfg.get("model", "gpt-4o-mini"),
            timeout=cfg.get("timeout_sec", 12),
            api_key=api_key,
            base_url=cfg.get("base_url", "https://api.openai.com"),
            retries=cfg.get("retries", 2),
            backoff_ms=cfg.get("backoff_ms", [300, 800]),
        )

    def on_done(fut):
        # (1) проверка актуальности карты — не трогаем следующую
        cur = getattr(mw.reviewer, "card", None)
        if not cur or cur.id != requested_card_id:
            return

        try:
            verdict = fut.result()
        except Exception as e:
            # (7)+(9)+(4) понятные сообщения об ошибках/таймаутах/джейсоне
            msg = str(e)
            low = msg.lower()
            if "rate limit" in low or "429" in low:
                _post_to_card(
                    context,
                    "⚠ Rate limit (429): слишком много запросов. Попробуй позже.",
                )
            elif "auth" in low or "401" in low or "403" in low:
                _post_to_card(context, "⚠ Ошибка авторизации (проверь API-ключ).")
            elif "timeout" in low or "timed out" in low:
                _post_to_card(
                    context,
                    f"⌛ GPT не ответил за {cfg.get('timeout_sec',12)} c. Оценка не поставлена.",
                )
            elif "json" in low:
                _post_to_card(context, "⚠ Не удалось разобрать ответ GPT (JSON).")
            else:
                _post_to_card(context, f"GPT недоступен: {msg}")
            return

        # нормальная ветка
        ease = map_to_ease(verdict.get("ease"))
        comment = (verdict.get("comment") or "").strip()
        label = (
            ["Again", "Hard", "Good", "Easy"][ease - 1] if ease in (1, 2, 3, 4) else "—"
        )
        msg = f"GPT: {comment} → {label}" if comment else f"GPT → {label}"
        _post_to_card(context, msg)

        if cfg.get("auto_answer", True) and ease in (1, 2, 3, 4):
            try:
                mw.reviewer._answerCard(ease)
            except Exception as e:
                tooltip(f"Не удалось ответить: {e}")

    mw.taskman.run_in_background(work, on_done)
    return (True, None)


def on_profile_loaded():
    # (5) при загрузке профиля — предупредим, если ключ пустой
    cfg = get_cfg()
    if not (cfg.get("openai_api_key") or "").strip():
        tooltip("⚠ Укажи openai_api_key в настройках аддона")
    gui_hooks.webview_did_receive_js_message.append(on_js_message)


gui_hooks.profile_did_open.append(on_profile_loaded)
