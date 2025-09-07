# -*- coding: utf-8 -*-
import hashlib
import json
import re
import secrets
import time
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, Optional

from aqt import gui_hooks, mw
from aqt.utils import tooltip

from .gpt_client import judge_text  # GPTError не использовался — убрал импорт
from .logic import get_cfg, map_to_ease

# ---------------- LRU-кеш ----------------
CACHE: "OrderedDict[str, tuple]" = OrderedDict()
CACHE_MAX = 500
_SALT = secrets.token_bytes(16)

LAST_REQUEST_TS = 0.0
HOOKS_ATTACHED = False  # защита от двойного навешивания хуков

_ws_re = re.compile(r"\s+")
_p_space = re.compile(r"\s+([,.:;!?])")


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = _ws_re.sub(" ", s)
    s = _p_space.sub(r"\1", s)
    return s


def _get_field(note, name: str) -> str:
    try:
        return (note[name] or "").strip()
    except Exception:
        return ""


def _cache_key(user: str, gold: str, model: str = "") -> str:
    # соль, чтобы по ключу нельзя было догадаться о содержимом
    h = hashlib.sha256()
    h.update(_SALT)
    h.update(model.encode("utf-8", "ignore"))
    h.update(b"\0")
    h.update(user.encode("utf-8", "ignore"))
    h.update(b"\0")
    h.update(gold.encode("utf-8", "ignore"))
    return h.hexdigest()


def _cache_get(k: str, now: float, ttl: int):
    v = CACHE.get(k)
    if not v:
        return None
    ts, payload = v
    if now - ts >= ttl:
        try:
            del CACHE[k]
        except KeyError:
            pass
        return None
    # LRU: помечаем как недавно использованный
    CACHE.move_to_end(k)
    return payload


def _cache_put(k: str, payload, now: float) -> None:
    CACHE[k] = (now, payload)
    CACHE.move_to_end(k)
    while len(CACHE) > CACHE_MAX:
        # удаляем самый старый
        CACHE.popitem(last=False)


def _tooltip_from_verdict(verdict: dict) -> str:
    """
    "<Category> — <Comment> → <Button>"
    или "<Comment> → <Button>"
    или просто "→ <Button>"
    """
    category = (verdict.get("category") or "").strip()
    button = (verdict.get("button") or "").strip()
    comment = (verdict.get("comment") or "").strip()

    right = f"→ {button}" if button else ""
    if category and comment:
        return f"{category} — {comment} {right}".strip()
    if comment:
        return f"{comment} {right}".strip()
    if button:
        return right.strip()
    return "✓"


def _ease_from_verdict(verdict: dict) -> int:
    # сначала пробуем числовой ease из модели
    e = map_to_ease(verdict.get("ease"))
    if e in (1, 2, 3, 4):
        return e
    # затем маппим из button
    btn = (verdict.get("button") or "").strip().lower()
    return {"again": 1, "hard": 2, "good": 3, "easy": 4}.get(btn, 0)


def _try_answer(ease: int) -> None:
    """Безопасно нажимаем кнопку повтора (API у Anki менялся)."""
    try:
        if hasattr(mw.reviewer, "_answerCard"):
            mw.reviewer._answerCard(ease)
    except Exception:
        pass


# ---------------- Отправка в UI карточки ----------------
def _push_ui_advice(
    ease: Optional[int] = None, comment: str = "", confidence: Optional[float] = None
) -> None:
    """
    Посылает совет в шаблон: window._ankiAddonCallback({...})
    (отрисовка панели «GPT» на карточке).
    """
    try:
        payload: Dict[str, Any] = {}
        if ease in (1, 2, 3, 4):
            payload["ease"] = int(ease)
        if comment:
            payload["comment"] = str(comment)
        if isinstance(confidence, (int, float)):
            c = max(0.0, min(1.0, float(confidence)))
            payload["confidence"] = c

        if not payload:
            return

        js = f"window._ankiAddonCallback({json.dumps(payload, ensure_ascii=False)});"
        if getattr(mw, "reviewer", None) and getattr(mw.reviewer, "web", None):
            mw.reviewer.web.eval(js)
    except Exception:
        # молча: UI-подсказка — приятный бонус, не критично
        pass


# ---------------- ЛОГ: отправка событий в панель на карточке ----------------
def _log_to_card(event: dict) -> None:
    """
    Отправить объект события в правую панель лога карточки:
    window._ankiLogAppend(ev)
    """
    try:
        js = f"window._ankiLogAppend({json.dumps(event, ensure_ascii=False)});"
        if getattr(mw, "reviewer", None) and getattr(mw.reviewer, "web", None):
            mw.reviewer.web.eval(js)
    except Exception:
        pass


# ---------------- Основной обработчик ----------------
def on_js_message(handled, message, context):
    # ожидаем строку вида "judge:{...json...}"
    if not isinstance(message, str) or not message.startswith("judge:"):
        return handled

    try:
        payload = json.loads(message[len("judge:") :])
    except Exception:
        tooltip("Ошибка: некорректный payload")
        _push_ui_advice(comment="Ошибка: некорректный payload")
        return (True, None)

    reviewer = mw.reviewer
    card = getattr(reviewer, "card", None)
    if not card:
        return (True, None)

    cfg = get_cfg()
    api_key = (cfg.get("openai_api_key") or "").strip()
    if not api_key:
        tooltip("⚠ Укажи openai_api_key в настройках аддона")
        _push_ui_advice(comment="Укажи openai_api_key в настройках аддона")
        return (True, None)

    note = card.note()
    gold = _get_field(note, cfg["fields"]["etalon_field"])
    user_text = (payload.get("text") or "").strip()
    if not user_text or not gold:
        if not user_text:
            _push_ui_advice(comment="Пустой ответ")
        return (True, None)

    user_norm = _norm(user_text)
    gold_norm = _norm(gold)

    # быстрый путь: точное совпадение
    if user_norm == gold_norm and gold_norm:
        ease = 4
        tooltip("Exact match → Easy")
        _push_ui_advice(ease=ease, comment="Точное совпадение")
        if cfg.get("auto_answer", True):
            _try_answer(ease)
        return (True, None)

    # анти-флуд (порог 0.5s между запросами к модели)
    global LAST_REQUEST_TS
    now = time.time()
    if now - LAST_REQUEST_TS < 0.5:
        return (True, None)
    LAST_REQUEST_TS = now

    # кэш
    cache_ttl = int(cfg.get("cache_ttl_sec", 600))
    model = (cfg.get("model") or "gpt-4o-mini").strip()
    ckey = _cache_key(user_norm, gold_norm, model)
    if cache_ttl > 0:
        cached = _cache_get(ckey, now, cache_ttl)
        if cached:
            ease = _ease_from_verdict(cached)
            tooltip(f"GPT(кеш): {_tooltip_from_verdict(cached)}")
            _push_ui_advice(
                ease=ease,
                comment=(cached.get("comment") or cached.get("category") or "").strip(),
                confidence=cached.get("confidence"),
            )
            if cfg.get("auto_answer", True) and ease in (1, 2, 3, 4):
                _try_answer(ease)
            return (True, None)

    requested_card_id = card.id

    def work():
        return judge_text(
            user_text=user_text[: cfg.get("max_input_len", 800)],
            gold_text=gold[: cfg.get("max_gold_len", 800)],
            api_key=api_key,
        )

    # лог: запрос
    _log_to_card(
        {
            "kind": "request",
            "model": model,
            "text_len": len(user_text),
            "ts": int(time.time() * 1000),
        }
    )

    def on_done(fut):
        # та же карта на экране?
        cur = getattr(mw.reviewer, "card", None)
        if not cur or cur.id != requested_card_id:
            return
        try:
            verdict = (
                fut.result()
            )  # {category, button, comment, ease, confidence?, usage?, cost_usd?, balance_usd?}
        except Exception as e:
            tooltip(f"GPT error: {e}")
            _push_ui_advice(comment=f"GPT error: {e}")
            _log_to_card(
                {
                    "kind": "error",
                    "model": model,
                    "message": str(e),
                    "ts": int(time.time() * 1000),
                }
            )
            return

        if cache_ttl > 0:
            _cache_put(ckey, verdict, time.time())

        ease = _ease_from_verdict(verdict)
        tooltip(f"GPT: {_tooltip_from_verdict(verdict)}")

        # UI карточки
        _push_ui_advice(
            ease=ease,
            comment=(verdict.get("comment") or verdict.get("category") or "").strip(),
            confidence=verdict.get("confidence"),
        )

        # лог: ответ/usage/стоимость
        usage = verdict.get("usage") or verdict.get("_usage") or {}
        _log_to_card(
            {
                "kind": "response",
                "model": model,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cost_usd": verdict.get("cost_usd"),
                "balance_usd": verdict.get("balance_usd"),
                "ts": int(time.time() * 1000),
            }
        )

        # авто-ответ
        if cfg.get("auto_answer", True) and ease in (1, 2, 3, 4):
            _try_answer(ease)

    try:
        mw.taskman.run_in_background(work, on_done)
    except Exception as e:
        tooltip(f"Ошибка запуска фоновой задачи: {e}")

    return (True, None)


def on_profile_loaded():
    global HOOKS_ATTACHED
    cfg = get_cfg()
    if not (cfg.get("openai_api_key") or "").strip():
        tooltip("⚠ Укажи openai_api_key в настройках аддона")

    # защита от повторного добавления хука (на случай повторных инициализаций)
    if not HOOKS_ATTACHED:
        gui_hooks.webview_did_receive_js_message.append(on_js_message)
        HOOKS_ATTACHED = True


gui_hooks.profile_did_open.append(on_profile_loaded)
