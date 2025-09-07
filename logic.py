from aqt import mw

DEFAULT_CFG = {
    "openai_api_key": "",
    "model": "gpt-4o-mini",
    "timeout_sec": 12,
    "auto_answer": True,
    # Поля заметки
    # Оставили только эталон — sentence_field вырезан, т.к. не используется в коде
    "fields": {"etalon_field": "Back"},
    # (1) кеш ответов, сек (0 = выключить кеш)
    "cache_ttl_sec": 600,
    # (3) лимиты длины для ввода и эталона
    "max_input_len": 800,
    "max_gold_len": 800,
    # Сеть/ретраи
    # ВАЖНО: нормализуем до /v1/, чтобы не плодить «/chat/completions» без версии
    "base_url": "https://api.openai.com/v1/",
    "retries": 2,
    "backoff_ms": [300, 800],
    # Новые параметры, перенесённые из «жёстких» значений в конфиг
    "temperature": 0.3,
    "max_tokens": 64,
}


def get_cfg():
    """Читает конфиг аддона и подмешивает дефолты."""
    cfg = mw.addonManager.getConfig(__name__) or {}
    merged = DEFAULT_CFG.copy()
    merged.update(cfg)

    # аккуратно подмешиваем вложенный блок полей
    merged["fields"] = {**DEFAULT_CFG["fields"], **cfg.get("fields", {})}

    # нормализуем base_url до /v1/
    bu = (merged.get("base_url") or "").strip()
    if bu:
        if bu.rstrip("/").endswith("openai.com"):
            bu = "https://api.openai.com/v1/"
        elif not bu.rstrip("/").endswith("/v1"):
            bu = bu.rstrip("/") + "/v1/"
        merged["base_url"] = bu
    else:
        merged["base_url"] = "https://api.openai.com/v1/"

    return merged


def map_to_ease(e):
    """Нормализуем ease к 1..4, иначе 0 (не нажимать)."""
    try:
        e = int(e)
    except Exception:
        return 0
    return e if e in (1, 2, 3, 4) else 0


# ----------------- Фильтры колод/типов — УДАЛЕНЫ ПО ТВОЕЙ ПРОСЬБЕ -----------------
# Ранее здесь были:
#   _deck_name, _match_deck, is_deck_allowed, is_note_type_allowed
# Теперь логика не ограничивает работу по колодам и типам заметок.
