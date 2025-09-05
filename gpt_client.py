import json, ssl, urllib.request

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

def _post_json(url, body, headers, timeout):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        return json.loads(r.read().decode("utf-8"))

def judge_text(user_text, gold_text, lang, strictness, model, timeout, api_key):
    if not api_key:
        raise RuntimeError("Не задан openai_api_key в конфиге")

    sys_prompt = (
        "You are a strict translation grader. Compare user's translation with the reference semantically. "
        "Return ONLY JSON with keys: ease (1..4), comment. "
        "Mapping: 4 Easy, 3 Good, 2 Hard, 1 Again."
    )
    user_prompt = f"Gold: {gold_text}\nUser: {user_text}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
    }

    resp = _post_json(OPENAI_URL, body, headers, timeout)
    content = resp["choices"][0]["message"]["content"]
    return json.loads(content)
