import hashlib
import json
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

# ===== Внутренние структуры =====
LAST_CALL = {}  # анти-дубль по (card.id + hash(user_text)) на коротком окне
LAST_REQUEST_TS = 0.0  # анти-флуд по времени
CACHE = {}  # (1) кеш ответов: key -> (ts, verdict_dict)

# ===== Порог для анти-дубля и анти-флуда (локальные, не в конфиге) =====
MIN_INTERVAL_SEC = 2.0  # анти-флуд: не чаще одного запроса раз в N сек
ANTI_DUP_WINDOW_SEC = 6.0  # анти-дубль: повторный Enter с тем же текстом в течение окна


def _get_field(note, name: str) -> str:
    """Безопасное получение значения поля заметки"""
    try:
        return (note[name] or "").strip()
    except KeyError:
        return ""


def _post_to_card(context, msg: str):
    """Показываем совет в тултипе и (если webview жив) прямо на карточке"""
    try:
        context.eval(
            f"if(window._ankiAddonCallback) _ankiAddonCallback({json.dumps(msg)});"
        )
    except Exception:
        pass
    tooltip(msg)


def _in_review_state() -> bool:
    """Работаем только в режиме Review (не в Preview/Browse)"""
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

    # только в режиме ревью
    if not _in_review_state():
        return (True, None)

    # распарсим payload из JS
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

    # проверка ключа заранее (чтобы не уходить в сеть впустую)
    api_key = (cfg.get("openai_api_key") or "").strip()
    if not api_key:
        _post_to_card(context, "⚠ Укажи openai_api_key в настройках аддона")
        return (True, None)

    # фильтры по колодам/типам (если заданы)
    note = card.note()
    if not is_deck_allowed(card, cfg) or not is_note_type_allowed(note, cfg):
        return (True, None)

    # извлекаем поля
    gold = _get_field(note, cfg["fields"]["etalon_field"])
    user_text = (payload.get("text") or "").strip()

    # пустые значения не отправляем
    if not user_text:
        return (True, None)
    if not gold:
        _post_to_card(
            context, f"Поле эталона '{cfg['fields']['etalon_field']}' пусто/нет"
        )
        return (True, None)

    # (3) лимиты длины: ВВОД
    max_in = int(cfg.get("max_input_len", 800))
    if len(user_text) > max_in:
        user_text = user_text[:max_in]
        _post_to_card(context, f"Ввод длиннее {max_in} символов — обрезано")

    # (3) лимиты длины: ЭТАЛОН
    max_gold = int(cfg.get("max_gold_len", 800))
    if len(gold) > max_gold:
        gold = gold[:max_gold]
        _post_to_card(context, f"Эталон длиннее {max_gold} символов — обрезано")

    # анти-флуд по времени
    global LAST_REQUEST_TS
    now = time.time()
    if now - LAST_REQUEST_TS < MIN_INTERVAL_SEC:
        _post_to_card(context, "Слишком часто (анти-флуд). Подожди секунду.")
        return (True, None)
    LAST_REQUEST_TS = now

    # анти-дубль по тому же тексту в коротком окне
    short_key = f"{card.id}:{hashlib.sha1(user_text.encode('utf8')).hexdigest()}"
    if short_key in LAST_CALL and now - LAST_CALL[short_key] < ANTI_DUP_WINDOW_SEC:
        return (True, None)
    LAST_CALL[short_key] = now

    # (1) кеш: ключ на карту + комбинацию эталон/ответ пользователя
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
            msg = f"GPT(кеш): {comment} → {label}" if comment else f"GPT(кеш) → {label}"
            _post_to_card(context, msg)
            if cfg.get("auto_answer", True) and ease in (1, 2, 3, 4):
                try:
                    mw.reviewer._answerCard(ease)
                except Exception as e:
                    tooltip(f"Не удалось ответить: {e}")
            return (True, None)

    # запоминаем карту, для которой уедет запрос (чтобы не нажать кнопку на другой)
    requested_card_id = card.id

    # фоновая задача: вызов GPT
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
        # та ли ещё карта на экране?
        cur = getattr(mw.reviewer, "card", None)
        if not cur or cur.id != requested_card_id:
            return

        try:
            verdict = fut.result()
        except Exception as e:
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
                    f"⌛ GPT не ответил за {cfg.get('timeout_sec', 12)} c. Оценка не поставлена.",
                )
            elif "json" in low:
                _post_to_card(context, "⚠ Не удалось разобрать ответ GPT (JSON).")
            else:
                _post_to_card(context, f"GPT недоступен: {msg}")
            return

        # сохраним в кеш
        if cache_ttl > 0:
            CACHE[cache_key] = (time.time(), verdict)

        # нормальная ветка
        ease = map_to_ease(verdict.get("ease"))
        if ease not in (1, 2, 3, 4):
            _post_to_card(context, "⚠ Не удалось определить оценку. Кнопка не нажата.")
            return

        comment = (verdict.get("comment") or "").strip()
        label = ["Again", "Hard", "Good", "Easy"][ease - 1]
        msg = f"GPT: {comment} → {label}" if comment else f"GPT → {label}"
        _post_to_card(context, msg)

        if cfg.get("auto_answer", True):
            try:
                mw.reviewer._answerCard(ease)
            except Exception as e:
                tooltip(f"Не удалось ответить: {e}")

    mw.taskman.run_in_background(work, on_done)
    return (True, None)


def on_profile_loaded():
    # раннее предупреждение, если ключ пустой
    cfg = get_cfg()
    if not (cfg.get("openai_api_key") or "").strip():
        tooltip("⚠ Укажи openai_api_key в настройках аддона")
    gui_hooks.webview_did_receive_js_message.append(on_js_message)


gui_hooks.profile_did_open.append(on_profile_loaded)
