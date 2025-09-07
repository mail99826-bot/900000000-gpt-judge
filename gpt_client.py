# gpt_client.py
# Полный файл: Function Calling (enum) + усиленный system-prompt + валидация + ретраи,
# и ВСЕ параметры читаются из config.json (model, temperature, top_p, max_tokens, retries, backoff, base_url, api_key).

from __future__ import annotations

import json
import os
import time
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
    "If uncertain, choose the closest allowed value; DO NOT invent new labels."
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

    tools = [
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
    Стратегия:
      - Guard на пустые ответы.
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
    backoff_ms = cfg.get(
        "backoff_ms", [500]
    )  # список миллисекунд; если короче чем retries — будем повторять последний интервал

    base_url = (cfg.get("base_url") or "").strip() or None
    config_api_key = (cfg.get("openai_api_key") or "").strip()
    use_api_key = (
        api_key or config_api_key or os.getenv("OPENAI_API_KEY") or ""
    ).strip()

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

    # Первая попытка
    verdict = _call_model_function_call(
        client=client,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        gold_text=gold_text,
        user_text=user_text,
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
            user_text=user_text,
            extra_system=extra_sys,
        )
        if _is_valid_verdict(verdict):
            return verdict

        attempts += 1

    # Если совсем не получилось — честно фейлимся
    raise ValueError("Model failed to produce a valid verdict after retries.")
