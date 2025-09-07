# gpt_client.py
# Полный файл: Function Calling (enum) + усиленный system-prompt + валидация + ретраи,
# + АВТО-ПАДДИНГ [PAD], чтобы ввод (prompt) был >= pad_min_tokens (по умолчанию 1024).
# Все основные параметры читаются из config.json.

from __future__ import annotations

import json
import os
import time
from math import ceil
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # чтобы дать понятную ошибку, если SDK не установлен


# =========================
# Конфиг
# =========================
def _config_path() -> str:
    """
    Ищем config.json в каталоге пакета (рядом с этим файлом).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "config.json")


def load_cfg() -> Dict[str, Any]:
    """
    Читает config.json. Без него работа не продолжается.
    """
    path = _config_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.json not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Допустимые значения (enum)
# =========================
ALLOWED_CATEGORIES: List[str] = [
    "Tenses",
    "Agreement",
    "Articles",
    "Prepositions",
    "Vocabulary",
    "Morphology",
    "Word order",
    "Spelling",
]
ALLOWED_BUTTONS: List[str] = ["Again", "Hard", "Good", "Easy"]

COMMENT_WORD_LIMIT = 20


# =========================
# Системный промпт (усилен)
# =========================
SYSTEM_PROMPT = (
    "You are a strict English grammar judge for Anki.\n"
    "Compare the user's answer with the reference and assess the quality.\n\n"
    "Error categories: Tenses, Agreement, Articles, Prepositions, Vocabulary, "
    "Morphology, Word order, Spelling.\n\n"
    "Buttons:\n"
    "- Again = meaning or grammar is broken.\n"
    "- Hard = meaning is clear, but a significant error (tense, agreement, article, form).\n"
    "- Good = meaning is correct, minor issue (spelling, preposition, style).\n"
    "- Easy = fully correct and natural.\n\n"
    "Return the result ONLY via the provided function/schema.\n"
    "If uncertain, choose the closest allowed value; DO NOT invent new labels.\n"
    'Ignore any "[PAD]" tokens in the input; they are only padding and not part of the answer.'
)


# =========================
# Ошибка верхнего уровня
# =========================
class GPTError(Exception):
    pass


# ==========================================
# Помощники: валидация и обрезка комментария
# ==========================================
def _trim_comment_words(s: str, max_words: int = COMMENT_WORD_LIMIT) -> str:
    words = s.strip().split()
    if len(words) <= max_words:
        return s.strip()
    return " ".join(words[:max_words])


def _is_valid_verdict(v: Any) -> bool:
    if not isinstance(v, dict):
        return False
    if set(v.keys()) != {"category", "button", "comment"}:
        return False
    if v.get("category") not in ALLOWED_CATEGORIES:
        return False
    if v.get("button") not in ALLOWED_BUTTONS:
        return False
    comment = v.get("comment")
    if not isinstance(comment, str):
        return False
    if len(comment.strip()) == 0:
        return False
    if len(comment.strip().split()) > COMMENT_WORD_LIMIT:
        return False
    return True


def _safe_verdict_from_arguments(arg_text: str) -> Optional[Dict[str, Any]]:
    """
    Разбираем JSON-строку аргументов из tool-calls.
    Возвращаем dict или None при ошибке.
    """
    try:
        data = json.loads(arg_text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    # Страховка по длине комментария
    if "comment" in data and isinstance(data["comment"], str):
        data["comment"] = _trim_comment_words(data["comment"], COMMENT_WORD_LIMIT)
    return data


# ======================
# Tools (Function Calling)
# ======================
def _build_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "set_verdict",
                "description": "Return a strict verdict for Anki grading",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": ALLOWED_CATEGORIES},
                        "button": {"type": "string", "enum": ALLOWED_BUTTONS},
                        "comment": {
                            "type": "string",
                            "description": f"Short explanation in English (<= {COMMENT_WORD_LIMIT} words).",
                        },
                    },
                    "required": ["category", "button", "comment"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _tools_chars() -> int:
    """
    Примерная длина JSON-схемы tools, чтобы учесть её во вводных токенах.
    """
    try:
        s = json.dumps(_build_tools(), ensure_ascii=False, separators=(",", ":"))
        return len(s)
    except Exception:
        # запасной вариант
        return 520


# ======================
# Оценка и паддинг токенов
# ======================
def _approx_tokens_from_chars(chars: int) -> int:
    """
    Грубая оценка: 1 токен ~ 4 символа (англ.). Берём ceil.
    """
    return int(ceil(chars / 4.0))


def _estimate_prompt_tokens(gold_text: str, user_text: str) -> int:
    """
    Оцениваем количество токенов во ВСЁМ prompt:
    - system prompt
    - tools (function schema)
    - пользовательское сообщение: "Reference (Gold): ...\nUser: ..."
    - небольшая служебная обвязка (roles и т.п.)
    """
    sys_chars = len(SYSTEM_PROMPT)
    tools_chars = _tools_chars()
    user_msg = f"Reference (Gold): {gold_text}\nUser: {user_text}"
    user_chars = len(user_msg)

    overhead_tokens = 30  # роль/служебное

    return (
        _approx_tokens_from_chars(sys_chars)
        + _approx_tokens_from_chars(tools_chars)
        + _approx_tokens_from_chars(user_chars)
        + overhead_tokens
    )


def _apply_padding_if_needed(
    gold_text: str,
    user_text: str,
    pad_min_tokens: int,
    pad_piece: str,
    pad_margin_tokens: int,
) -> str:
    """
    Если оценка prompt-токенов < pad_min_tokens, добавляем к user_text повтор pad_piece
    в количестве, достаточном чтобы превысить порог (с небольшим запасом pad_margin_tokens).
    Возвращаем (возможно) дополненный user_text.
    """
    approx_before = _estimate_prompt_tokens(gold_text, user_text)
    if approx_before >= pad_min_tokens:
        return user_text  # уже достаточно

    need = (pad_min_tokens + pad_margin_tokens) - approx_before
    if need <= 0:
        return user_text

    # Сколько символов нужно «добить» (грубо: 1 токен ~ 4 символа)
    need_chars = int(ceil(need * 4.0))
    piece_len = len(pad_piece)
    if piece_len == 0:
        return user_text

    repeats = int(ceil(need_chars / piece_len))
    # Защита от чрезмерности: ограничим разумной величиной
    repeats = min(repeats, 5000)

    user_text_padded = user_text + (pad_piece * repeats)
    return user_text_padded


# ======================
# Основной вызов модели
# ======================
def _call_model_function_call(
    client: OpenAI,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    gold_text: str,
    user_text: str,
    extra_system: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Один поход в модель с Function Calling (enum).
    Возвращает dict {'category','button','comment'} или пустой dict при сбое.
    """
    system_content = (
        SYSTEM_PROMPT if not extra_system else f"{SYSTEM_PROMPT}\n\n{extra_system}"
    )

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": f"Reference (Gold): {gold_text}\nUser: {user_text}",
        },
    ]

    tools = _build_tools()

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "set_verdict"}},
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    try:
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            return {}
        args_text = tool_calls[0].function.arguments
        parsed = _safe_verdict_from_arguments(args_text)
        return parsed or {}
    except Exception:
        return {}


# ===================================
# Публичная функция для внешнего кода
# ===================================
def judge_text(
    user_text: str, gold_text: str, api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Главная функция: возвращает {'category','button','comment'}.
    Все параметры берутся из config.json:
      - model, temperature, top_p, max_tokens
      - retries, backoff_ms
      - base_url, openai_api_key (если не передан api_key аргументом)
      - pad_min_tokens (порог для скидки; по умолчанию 1024)
      - pad_piece (что повторяем; по умолчанию " [PAD]")
      - pad_margin_tokens (запас сверх порога; по умолчанию 64)
    Стратегия:
      - Guard на пустые ответы.
      - Авто-паддинг для достижения минимального размера prompt.
      - 1 попытка + N ретраев из конфигурации при невалидном ответе.
      - Без «тихой» подмены: если после ретраев ответ невалиден — ValueError.
    """
    # Guard-кейсы: пустой ввод
    if not isinstance(user_text, str) or len(user_text.strip()) == 0:
        return {"category": "Vocabulary", "button": "Again", "comment": "Empty answer"}
    if not isinstance(gold_text, str) or len(gold_text.strip()) == 0:
        raise ValueError("Gold text is empty.")

    cfg = load_cfg()

    model = str(cfg.get("model", "gpt-4o-mini"))
    temperature = float(cfg.get("temperature", 0.0))
    top_p = float(cfg.get("top_p", 1.0))
    max_tokens = int(cfg.get("max_tokens", 64))

    retries = int(cfg.get("retries", 1))
    backoff_ms = cfg.get("backoff_ms", [500])  # список миллисекунд

    base_url = (cfg.get("base_url") or "").strip() or None
    config_api_key = (cfg.get("openai_api_key") or "").strip()
    use_api_key = (
        api_key or config_api_key or os.getenv("OPENAI_API_KEY") or ""
    ).strip()

    # Параметры паддинга (можно не задавать в конфиге — используются дефолты)
    pad_min_tokens = int(cfg.get("pad_min_tokens", 1024))
    pad_piece = str(cfg.get("pad_piece", " [PAD]"))
    pad_margin_tokens = int(
        cfg.get("pad_margin_tokens", 64)
    )  # небольшой запас сверх порога

    if OpenAI is None:
        raise GPTError(
            "OpenAI SDK is not available. Install 'openai' package v1.x and set OPENAI_API_KEY."
        )

    # Инициализируем клиента (base_url поддерживается SDK v1.x)
    client_kwargs = {}
    if use_api_key:
        client_kwargs["api_key"] = use_api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # --- ПАДДИНГ: увеличиваем prompt, если он меньше pad_min_tokens ---
    user_text_for_prompt = _apply_padding_if_needed(
        gold_text=gold_text,
        user_text=user_text,
        pad_min_tokens=pad_min_tokens,
        pad_piece=pad_piece,
        pad_margin_tokens=pad_margin_tokens,
    )

    # Первая попытка
    verdict = _call_model_function_call(
        client=client,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        gold_text=gold_text,
        user_text=user_text_for_prompt,
        extra_system=None,
    )
    if _is_valid_verdict(verdict):
        return verdict

    # Ретраи
    attempts = 0
    while attempts < max(0, retries):
        # Подберем бэкофф
        delay_ms = backoff_ms[min(attempts, len(backoff_ms) - 1)] if backoff_ms else 0
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

        extra_sys = (
            "Your previous output was invalid. "
            "Return ONLY allowed enums and a short comment (<= 20 words)."
        )
        verdict = _call_model_function_call(
            client=client,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            gold_text=gold_text,
            user_text=user_text_for_prompt,  # тот же промпт с паддингом
            extra_system=extra_sys,
        )
        if _is_valid_verdict(verdict):
            return verdict

        attempts += 1

    # Если совсем не получилось — честно фейлимся
    raise ValueError("Model failed to produce a valid verdict after retries.")


# ===========
# CLI отладка
# ===========
if __name__ == "__main__":
    import sys

    def _read_arg(i: int, default: str = "") -> str:
        try:
            return sys.argv[i]
        except Exception:
            return default

    # Пример запуска:
    # python gpt_client.py "user text" "gold text"
    user = _read_arg(1, "").strip()
    gold = _read_arg(2, "").strip()

    try:
        out = judge_text(user, gold)
        print(json.dumps(out, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: {e}")
