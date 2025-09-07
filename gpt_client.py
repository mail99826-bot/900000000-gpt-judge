import json
import random
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict

DEFAULT_UA = "Anki-GPT-Judge/0.4"
_ctx = ssl.create_default_context()
_opener = urllib.request.build_opener()
_opener.addheaders = [
    ("User-Agent", DEFAULT_UA),
    ("Connection", "keep-alive"),
    ("Accept-Encoding", "gzip"),
    ("Content-Type", "application/json"),
]


# ----------------------------- HTTP -----------------------------


def _post_json(
    url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int
) -> Dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with _opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


# ---------------------------- Parsing ---------------------------


def _sanitize_json_text(s: str) -> str:
    s = (s or "").strip()
    # Иногда модель кладёт ответ в ```json ... ```
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1)
    return s.strip()


def _norm_button(btn: str) -> str:
    btn = (btn or "").strip().lower()
    if btn in ("again", "1"):
        return "Again"
    if btn in ("hard", "2"):
        return "Hard"
    if btn in ("good", "3"):
        return "Good"
    if btn in ("easy", "4"):
        return "Easy"
    return ""


def _button_to_ease(btn: str) -> int:
    m = {"Again": 1, "Hard": 2, "Good": 3, "Easy": 4}
    return m.get(btn, 0)


_ALLOWED_CATEGORIES = {
    "Tenses",
    "Agreement",
    "Articles",
    "Prepositions",
    "Vocabulary",
    "Morphology",
    "Word order",
    "Spelling",
}


def _coerce_verdict(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Приводим ответ модели к строгой схеме:
      category ∈ _ALLOWED_CATEGORIES
      button ∈ {"Again","Hard","Good","Easy"}
      comment ≤ 20 слов, без управляющих символов
      ease ∈ {1..4} (производное от button)
    """
    # category
    cat = (obj.get("category") or "").strip()
    # допускаем разные кейсы, нормализуем по регистру
    for allowed in _ALLOWED_CATEGORIES:
        if cat.lower() == allowed.lower():
            cat = allowed
            break
    else:
        # если модель не дала валидную категорию — ставим пусто
        cat = ""

    # button → ease
    btn = _norm_button(obj.get("button"))
    ease = _button_to_ease(btn)

    # comment
    comment = (obj.get("comment") or "").strip()
    if comment:
        words = comment.split()
        if len(words) > 20:
            comment = " ".join(words[:20])
        # убрать управляющие символы
        comment = "".join(ch for ch in comment if (ch >= " " or ch == "\n"))

    return {
        "category": cat,
        "button": btn,
        "comment": comment,
        "ease": ease,  # для совместимости с остальным кодом
    }


# ------------------------------ Errors ------------------------------


class GPTError(RuntimeError):
    pass


def _sleep_with_backoff(i: int, backoff_ms, retry_after_sec=None) -> None:
    if retry_after_sec:
        time.sleep(float(retry_after_sec))
        return
    ms = backoff_ms[min(i, len(backoff_ms) - 1)] if backoff_ms else 400
    base = ms / 1000.0
    # джиттер ±30% для рассинхронизации при наплыве
    jitter = base * (0.7 + 0.6 * random.random())
    time.sleep(jitter)


# ------------------------------ Main ------------------------------


def judge_text(
    user_text: str,
    gold_text: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    api_key: str,
    base_url: str = "https://api.openai.com/v1/",
    retries: int = 1,
    backoff_ms=(300, 800),
) -> Dict[str, Any]:
    """
    Сравнивает ответ пользователя с эталоном.
    Возвращает дикт: {"category": str, "button": str, "comment": str, "ease": int}
    """
    if not api_key:
        raise GPTError("OpenAI API key is empty")

    # Твой англоязычный промпт, ужатый и структурированный под JSON-ответ
    sys_prompt = (
        "You are a strict English grammar judge for Anki.\n"
        "Compare the user's answer with the reference and assess the quality.\n\n"
        "Error categories: Tenses, Agreement, Articles, Prepositions, Vocabulary, Morphology, Word order, Spelling.\n\n"
        "Buttons:\n"
        "- Again = meaning or grammar is broken.\n"
        "- Hard = meaning is clear, but a significant error (tense, agreement, article, form).\n"
        "- Good = meaning is correct, minor issue (spelling, preposition, style).\n"
        "- Easy = fully correct and natural.\n\n"
        "Return ONLY JSON with exactly these fields:\n"
        "{\n"
        '  "category": one of ["Tenses","Agreement","Articles","Prepositions","Vocabulary","Morphology","Word order","Spelling"],\n'
        '  "button": one of ["Again","Hard","Good","Easy"],\n'
        '  "comment": short explanation in English (max 20 words)\n'
        "}\n"
    )

    user_prompt = f"Reference (Gold): {gold_text}\nUser: {user_text}"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    attempt = 0
    last_err = None

    while attempt <= max(0, int(retries)):
        try:
            resp = _post_json(url, body, headers, timeout)
            msg = resp["choices"][0]["message"]["content"]
            obj = json.loads(_sanitize_json_text(msg))
            return _coerce_verdict(obj)

        except urllib.error.HTTPError as he:
            status = he.code
            retry_after = None
            try:
                ra = he.headers.get("Retry-After")
                if ra:
                    retry_after = int(ra)
            except Exception:
                retry_after = None

            if status in (401, 403):
                # авторизация/права — не ретраим
                raise GPTError("Auth error (check API key)") from he
            if status in (429, 500, 502, 503, 504):
                last_err = GPTError(f"HTTP {status}")
            else:
                body_preview = ""
                try:
                    body_preview = he.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                raise GPTError(f"HTTP {status}: {body_preview}") from he

        except Exception as e:
            last_err = e

        if attempt < retries:
            _sleep_with_backoff(attempt, backoff_ms, retry_after)
            attempt += 1
            continue
        break

    raise last_err or GPTError("Unknown error")
