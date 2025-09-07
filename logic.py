from aqt import mw

DEFAULT_CFG = {
    "openai_api_key": "",
    "model": "gpt-4o-mini",
    "timeout_sec": 12,
    "auto_answer": True,
    # Поля заметки (оставили только эталон)
    "fields": {"etalon_field": "Back"},
    # Кеш ответов (сек) (0 = отключить)
    "cache_ttl_sec": 600,
    # Лимиты длины для ввода и эталона
    "max_input_len": 800,
    "max_gold_len": 800,
    # Сеть/ретраи
    "base_url": "https://api.openai.com/v1/",
    "retries": 2,
    "backoff_ms": [300, 800],
    # Параметры генерации модели
    "temperature": 0.3,
    "max_tokens": 64,
}


def get_cfg():
    """Читает конфиг аддона и подмешивает дефолты. Нормализует base_url."""
    cfg = mw.addonManager.getConfig(__name__) or {}
    merged = DEFAULT_CFG.copy()
    merged.update(cfg)

    # подмешиваем вложенный блок полей
    merged["fields"] = {**DEFAULT_CFG["fields"], **cfg.get("fields", {})}

    # нормализуем base_url до https://.../v1/
    bu = (merged.get("base_url") or "").strip()
    if not bu:
        bu = "https://api.openai.com/v1/"
    if bu.startswith("http://"):
        # запрещаем незащищённый http
        bu = "https://api.openai.com/v1/"
    if bu.rstrip("/").endswith("openai.com"):
        bu = "https://api.openai.com/v1/"
    elif not bu.rstrip("/").endswith("/v1"):
        bu = bu.rstrip("/") + "/v1/"
    merged["base_url"] = bu

    return merged


def map_to_ease(e):
    """Нормализуем ease к 1..4, иначе 0 (не нажимать)."""
    try:
        e = int(e)
    except Exception:
        return 0
    return e if e in (1, 2, 3, 4) else 0
