import json
import ssl
import time
import urllib.error
import urllib.request

DEFAULT_UA = "Anki-GPT-Judge/0.2"
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
        return json.loads(r.read().decode("utf-8"))


def _sanitize_json_text(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1)
    return s.strip()


class GPTError(RuntimeError):
    pass


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
    backoff_ms=(400,),
):
    if not api_key:
        raise GPTError("Не задан openai_api_key в конфиге")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    sys_prompt = (
        "Grade RU→EN vs reference. "
        'Return ONLY JSON {"ease":1..4, "comment":string<=12w}.'
    )
    user_prompt = f"Gold:{gold_text}\nUser:{user_text}"

    body = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    attempt = 0
    last_err = None
    delays = list(backoff_ms) if isinstance(backoff_ms, (list, tuple)) else [400]

    while attempt <= retries:
        try:
            resp = _post_json(url, body, headers, timeout)
            content = resp["choices"][0]["message"]["content"]
            txt = _sanitize_json_text(content)
            data = json.loads(txt)

            e = data.get("ease", 0)
            try:
                e = int(e)
            except Exception:
                e = 0
            data["ease"] = e if e in (1, 2, 3, 4) else 0

            c = (data.get("comment") or "").strip()
            if c:
                words = c.split()
                if len(words) > 12:
                    c = " ".join(words[:12])
                data["comment"] = c

            return data

        except urllib.error.HTTPError as he:
            status = he.code
            try:
                err_body = he.read().decode("utf-8")
            except Exception:
                err_body = ""
            if status in (401, 403):
                raise GPTError("Auth error (проверь API ключ)") from he
            if status == 429:
                last_err = GPTError("Rate limit (429): слишком много запросов")
            elif 500 <= status < 600:
                last_err = GPTError(f"Server error {status}")
            else:
                raise GPTError(f"HTTP {status}: {err_body[:200]}") from he

        except Exception as e:
            last_err = e

        if attempt < retries:
            time.sleep(delays[min(attempt, len(delays) - 1)] / 1000.0)
        attempt += 1

    raise last_err or GPTError("Неизвестная ошибка при обращении к GPT")
