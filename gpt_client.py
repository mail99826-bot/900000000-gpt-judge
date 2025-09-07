import json
import random
import ssl
import time
import urllib.error
import urllib.request

DEFAULT_UA = "Anki-GPT-Judge/0.3"
_ctx = ssl.create_default_context()
_opener = urllib.request.build_opener()
_opener.addheaders = [
    ("User-Agent", DEFAULT_UA),
    ("Connection", "keep-alive"),
    ("Accept-Encoding", "gzip"),
    ("Content-Type", "application/json"),
]


def _post_json(url, body, headers, timeout):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with _opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _sanitize_json_text(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        # иногда модель возвращает fenced code block
        s = s.strip("`")
        s = s.replace("json\n", "", 1)
    return s.strip()


def _coerce_verdict(obj: dict) -> dict:
    # минимальная и строгая схема ответа
    ease = obj.get("ease")
    try:
        ease = int(ease)
    except Exception:
        ease = 0
    if ease not in (1, 2, 3, 4):
        ease = 0

    comment = (obj.get("comment") or "").strip()
    if comment:
        words = comment.split()
        if len(words) > 12:
            comment = " ".join(words[:12])
        # убрать управляющие символы
        comment = "".join(ch for ch in comment if (ch >= " " or ch == "\n"))
    return {"ease": ease, "comment": comment}


class GPTError(RuntimeError):
    pass


def _sleep_with_backoff(i, backoff_ms, retry_after_sec=None):
    if retry_after_sec:
        time.sleep(float(retry_after_sec))
        return
    ms = backoff_ms[min(i, len(backoff_ms) - 1)] if backoff_ms else 400
    base = ms / 1000.0
    # джиттер ±30% для рассинхронизации
    jitter = base * (0.7 + 0.6 * random.random())
    time.sleep(jitter)


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
):
    if not api_key:
        raise GPTError("OpenAI API key is empty")

    # краткий и жёсткий системный промпт
    sys_prompt = (
        "Compare user answer to reference. "
        'Return ONLY JSON {"ease":1..4,"comment":string<=12w}.'
    )
    user_prompt = f"Gold:{gold_text}\nUser:{user_text}"

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
            # читаем Retry-After, если прислал сервер (секунды)
            retry_after = None
            try:
                ra = he.headers.get("Retry-After")
                if ra:
                    retry_after = int(ra)
            except Exception:
                retry_after = None

            if status in (401, 403):
                # не пытаться ретраить — это ошибка авторизации / прав
                raise GPTError("Auth error (check API key)") from he
            if status in (429, 500, 502, 503, 504):
                last_err = GPTError(f"HTTP {status}")
            else:
                # непрозрачные ошибки без повторов
                body_preview = ""
                try:
                    body_preview = he.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                raise GPTError(f"HTTP {status}: {body_preview}") from he

        except Exception as e:
            last_err = e

        # backoff + джиттер
        if attempt < retries:
            _sleep_with_backoff(attempt, backoff_ms, retry_after)
            attempt += 1
            continue
        break

    raise last_err or GPTError("Unknown error")
