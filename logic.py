from aqt import mw

DEFAULT_CFG = {
    "openai_api_key": "",
    "model": "gpt-4o-mini",
    "lang": "en",
    "strictness": "medium",
    "timeout_sec": 12,
    "auto_answer": True,
    # Поля заметки
    "fields": {"sentence_field": "Front", "etalon_field": "Back"},
    # (1) кеш ответов, сек (0 = выключить кеш)
    "cache_ttl_sec": 600,
    # (3) лимиты длины для ввода и эталона
    "max_input_len": 800,
    "max_gold_len": 800,
    # Фильтры (необязательно использовать)
    "deck_whitelist": [],  # ["EN::Translate"]
    "deck_blacklist": [],  # ["ES::*"]
    "note_type_whitelist": [],  # ["Basic (type in the answer)"]
    "note_type_blacklist": [],
    # Сеть/ретраи
    "base_url": "https://api.openai.com",
    "retries": 2,
    "backoff_ms": [300, 800],
}


def get_cfg():
    """Читает конфиг аддона и подмешивает дефолты."""
    cfg = mw.addonManager.getConfig(__name__) or {}
    merged = DEFAULT_CFG.copy()
    merged.update(cfg)
    merged["fields"] = {**DEFAULT_CFG["fields"], **cfg.get("fields", {})}
    return merged


def map_to_ease(e):
    """Нормализуем ease к 1..4, иначе 0 (не нажимать)."""
    try:
        e = int(e)
    except Exception:
        return 0
    return e if e in (1, 2, 3, 4) else 0


# ----------------- Фильтры колод/типов (по желанию) -----------------


def _deck_name(card) -> str:
    """Возвращает полное имя колоды для карты (с учётом odid)."""
    try:
        d = mw.col.decks.get(card.odid or card.did)
        return d.get("name") or ""
    except Exception:
        return ""


def _match_deck(name: str, pattern: str) -> bool:
    """Простая маска: '*' в конце означает префиксное совпадение."""
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


def is_deck_allowed(card, cfg) -> bool:
    """True, если карта разрешена по спискам колод (whitelist/blacklist)."""
    name = _deck_name(card)
    wl = cfg.get("deck_whitelist") or []
    bl = cfg.get("deck_blacklist") or []
    if wl and not any(_match_deck(name, p) for p in wl):
        return False
    if bl and any(_match_deck(name, p) for p in bl):
        return False
    return True


def is_note_type_allowed(note, cfg) -> bool:
    """True, если тип заметки разрешён."""
    try:
        m = mw.col.models.get(note.mid)
        mname = m.get("name", "")
    except Exception:
        # если не смогли определить — не блокируем
        return True
    wl = cfg.get("note_type_whitelist") or []
    bl = cfg.get("note_type_blacklist") or []
    if wl and mname not in wl:
        return False
    if bl and mname in bl:
        return False
    return True
