"""
RAG и модуль обработки текста для Grand Mobile Support Copilot.
Включает:
- Нормализацию казахских cp1251 символов
- Индексацию и поиск правил из rules.json и taxonomy.json без перезаписи
- Извлечение никнеймов, пунктов правил, доказательств
- Определение языка (ru, kk, uz, en, es, pt, und)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Пути к файлам данных
DATA_DIR = Path(__file__).resolve().parent / "support_copilot" / "handout"
RULES_PATH = DATA_DIR / "rules.json"
TAXONOMY_PATH = DATA_DIR / "taxonomy.json"

# Таблица замены искажённых cp1251 казахских букв
CP1251_KZ_MAP = {
    'ќ': 'қ', 'Ќ': 'Қ',
    'ґ': 'ғ', 'Ґ': 'Ғ',
    'ѕ': 'ң', 'Ѕ': 'Ң',
    'є': 'ә', 'Є': 'Ә',
    'ј': 'ө', 'Ј': 'Ө',
    'ў': 'ұ', 'Ў': 'Ұ',
    'ї': 'ү', 'Ї': 'Ү',
}

# Регулярка для поиска никнеймов формата CRMP/SAMP: First_Last или First_Last123
NICK_PATTERN = re.compile(r'\b[A-Z][a-zA-Z0-9]+_[A-Za-z0-9]+\b')

# Регулярка для ссылок на правила (2.1, 5.2, 5.4.1, ЖБ 1.1.2 и т.д.)
RULE_REF_PATTERN = re.compile(
    r'\b(?:\d+\.\d+(?:\.\d+)?|GZ|DM|DB|SK|TK|PG|MG|RMT|НонРП|NonRP)\b',
    re.IGNORECASE
)

# Маркеры наличия доказательств в тексте
EVIDENCE_KEYWORDS = [
    "скрин", "скриншот", "видео", "пруф", "доказ", "fraps", "фрапс", 
    "док-ва", "доква", "сурет", "бейне", "<url>", "тайм", "time",
    "video", "screenshot", "proof", "prueba", "captura", "prova"
]


def normalize_kz_text(text: str) -> str:
    """
    Нормализует искажения cp1251 казахских букв,
    не повреждая при этом никнеймы и служебные токены.
    """
    if not text:
        return ""
    
    # Сохраняем токены вроде <url>, <email>, <digits>, <nick>
    tokens = {}
    def repl_token(match):
        key = f"__TOKEN_{len(tokens)}__"
        tokens[key] = match.group(0)
        return key

    temp_text = re.sub(r'<[a-zA-Z0-9_]+>', repl_token, text)

    # Применяем посимвольную замену
    chars = [CP1251_KZ_MAP.get(ch, ch) for ch in temp_text]
    normalized = "".join(chars)

    # Восстанавливаем токены
    for key, val in tokens.items():
        normalized = normalized.replace(key, val)

    return normalized


def detect_language(text: str, default: str = "ru") -> str:
    """
    Эвристика определения языка обращения:
    Поддерживаемые языки: ru, kk, uz, en, es, pt.
    """
    if not text or len(text.strip()) == 0:
        return "und"

    norm = normalize_kz_text(text.lower())

    # Специфичные буквы казахского языка
    kz_markers = {'ә', 'ғ', 'қ', 'ң', 'ө', 'ұ', 'ү', 'һ', 'і'}
    kz_count = sum(1 for ch in norm if ch in kz_markers)
    kz_common_words = {"сәлем", "көмек", "ойында", "қалай", "жоғалды", "ақша", "өтініш", "кеше", "қайда", "ертең", "алдап"}
    ru_words_in_text = set(re.findall(r'[а-яёәғқңөұүһі]+', norm))
    if kz_count > 0 or len(ru_words_in_text.intersection(kz_common_words)) >= 1:
        return "kk"

    # Узбекские маркеры (латиница)
    uz_words = {"salom", "iltimos", "yordam", "qanday", "nega", "mening", "aka", "uka", "xush", "rahmat", "pul", "sokyapti", "kun"}
    words_lat = set(re.findall(r'[a-zA-Z\']+', norm))
    if words_lat and len(words_lat.intersection(uz_words)) >= 1:
        return "uz"
    if "o'" in norm or "g'" in norm or "o‘" in norm or "g‘" in norm:
        return "uz"

    # Испанские маркеры
    es_words = {"hola", "por", "favor", "ayuda", "jugador", "trampa", "gracias", "queja", "juego", "cuenta", "dinero", "baneado", "denuncia"}
    es_found = words_lat.intersection(es_words)
    if len(es_found) >= 2 or any(ch in norm for ch in ['ñ', '¿', '¡']):
        return "es"

    # Португальские маркеры
    pt_words = {"olá", "ola", "ajuda", "jogador", "obrigado", "obrigada", "jogo", "conta", "dinheiro", "banido", "denúncia", "trapaceiro", "servidor"}
    pt_found = words_lat.intersection(pt_words)
    if len(pt_found) >= 2 or any(term in norm for term in ["ção", "ções", "não"]):
        return "pt"

    # Английские маркеры
    en_stopwords = {"the", "is", "are", "you", "my", "me", "he", "she", "they", "we", "player", "please", "banned", "why", "hello", "help", "unban", "account", "money"}
    en_found = words_lat.intersection(en_stopwords)
    if len(en_found) >= 2 or ("help" in words_lat and len(words_lat) <= 4):
        return "en"

    # По умолчанию для русскоязычного сервера и игрового сленга
    return default


class RuleKnowledgeBase:
    """Индексация и быстрый поиск по rules.json и taxonomy.json без коллизий"""

    def __init__(self, rules_path: Path = RULES_PATH, taxonomy_path: Path = TAXONOMY_PATH):
        self.rules_by_ref: Dict[str, str] = {}
        self.rules_by_category: Dict[str, Dict[str, str]] = {}
        self.rules_list: List[Dict[str, Any]] = []
        self.categories: Dict[str, Dict[str, Any]] = {}
        self.priorities: Dict[str, Dict[str, Any]] = {}
        self.needs_human_always: List[str] = [
            "report.account", "report.payment", "complaint.admin"
        ]
        self._load_taxonomy(taxonomy_path)
        self._load_rules(rules_path)

    def _load_taxonomy(self, path: Path):
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cat in data.get("categories", []):
                self.categories[cat["code"]] = cat
            for prio in data.get("priorities", []):
                self.priorities[prio["code"]] = prio
            if "needs_human_always" in data:
                self.needs_human_always = data["needs_human_always"]
        except Exception as e:
            print(f"Ошибка загрузки taxonomy.json: {e}")

    def _load_rules(self, path: Path):
        """
        Индексирует правила так, чтобы приоритет имели "Общие правила игры",
        а специфичные категории не перезаписывали ключевые пункты (2.1, 5.2, 5.4 и т.д.).
        """
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for rule_group in data.get("rules", []):
                cat_name = rule_group.get("category", "").strip()
                if cat_name not in self.rules_by_category:
                    self.rules_by_category[cat_name] = {}

                is_general = "Общие правила" in cat_name
                is_gz = "Зелёной Зоне" in cat_name
                is_complaints = "Правила жалоб" in cat_name

                for pt in rule_group.get("points", []):
                    ref = pt.get("ref", "").strip()
                    text = pt.get("text", "").strip()
                    if not ref:
                        continue

                    # Сохраняем в категории
                    self.rules_by_category[cat_name][ref] = text
                    self.rules_list.append({"ref": ref, "text": text, "category": cat_name})

                    # Основной индекс
                    if is_general:
                        # Общие правила игры — высший приоритет для "2.1", "5.2" и т.д.
                        self.rules_by_ref[ref] = text
                        self.rules_by_ref[f"ОП {ref}"] = text
                    elif is_gz:
                        self.rules_by_ref[f"GZ {ref}"] = text
                        if ref == "1.1":
                            self.rules_by_ref["GZ"] = text
                            self.rules_by_ref["ЗЗ"] = text
                    elif is_complaints:
                        self.rules_by_ref[f"ЖБ {ref}"] = text
                        if ref.startswith("1.1."):
                            self.rules_by_ref[ref] = text
                    else:
                        # Специализированные правила сохраняем с квалифицированным именем,
                        # чтобы не перетереть общие правила
                        qual_key = f"{cat_name} {ref}"
                        self.rules_by_ref[qual_key] = text
                        # Добавляем в общий индекс только если ref ещё свободен
                        if ref not in self.rules_by_ref:
                            self.rules_by_ref[ref] = text

        except Exception as e:
            print(f"Ошибка загрузки rules.json: {e}")

    def get_rule_text(self, ref: str) -> Optional[str]:
        """
        Получить точный текст правила по идентификатору (например, '2.1', '5.2', 'GZ', 'ЖБ 1.1.2').
        """
        if not ref:
            return None
        ref_clean = ref.strip()
        # Прямое совпадение
        if ref_clean in self.rules_by_ref:
            return self.rules_by_ref[ref_clean]
        # Проверка верхнего регистра (GZ, ЗЗ)
        if ref_clean.upper() in self.rules_by_ref:
            return self.rules_by_ref[ref_clean.upper()]
        # Проверка с префиксом ОП
        if f"ОП {ref_clean}" in self.rules_by_ref:
            return self.rules_by_ref[f"ОП {ref_clean}"]
        # Проверка с префиксом ЖБ
        if f"ЖБ {ref_clean}" in self.rules_by_ref:
            return self.rules_by_ref[f"ЖБ {ref_clean}"]
        return None

    def search_rules(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        """
        Поиск релевантных правил по тексту обращения или ключевым словам.
        Возвращает точные тексты без галлюцинаций.
        """
        query_norm = normalize_kz_text(query).lower()
        results = []

        # 1. Прямой поиск упоминания номеров правил
        found_refs = set(RULE_REF_PATTERN.findall(query))
        for ref in found_refs:
            ref_clean = ref.upper() if ref.upper() in ["GZ", "DM"] else ref
            text = self.get_rule_text(ref_clean)
            if text:
                results.append({"ref": ref_clean, "text": text, "match": "exact_ref"})

        # 2. Полнотекстовый поиск по ключевым маркерам
        kw_map = {
            "оскорб": ["2.1", "2.4"],
            "оск": ["2.1"],
            "мать": ["2.1"],
            "родн": ["2.1"],
            "мут": ["2.1", "3.1"],
            "чит": ["5.2"],
            "софт": ["5.2"],
            "аим": ["5.2"],
            "рван": ["5.2"],
            "вх": ["5.2"],
            "автокликер": ["5.2"],
            "бот": ["5.2"],
            "дм": ["5.4", "GZ"],
            "dm": ["5.4", "GZ"],
            "дб": ["5.4"],
            "убил": ["5.4"],
            "зелен": ["GZ"],
            "обман": ["4.1", "4.2"],
            "развод": ["4.1", "4.2"],
            "алдап": ["4.1"],
            "реал": ["4.4", "2.3"],
            "вирты": ["4.4"],
            "донат": ["1.4"],
            "взлом": ["1.4"],
            "пропал": ["1.4"],
            "застрял": ["3.1"],
            "текстур": ["3.1"],
            "админ": ["3.4"],
        }

        for kw, refs in kw_map.items():
            if kw in query_norm:
                for r in refs:
                    text = self.get_rule_text(r)
                    if text and not any(res["ref"] == r for res in results):
                        results.append({"ref": r, "text": text, "match": f"keyword:{kw}"})
                    if len(results) >= limit:
                        break
            if len(results) >= limit:
                break

        return results[:limit]


# Глобальный синглтон базы знаний
kb = RuleKnowledgeBase()


def extract_ticket_facts(messages: List[Dict[str, Any]], kind: str = "complaint") -> Dict[str, Any]:
    """
    Извлекает структурированные факты из списка сообщений тикета:
    - Никнейм нарушителя / игрока
    - Ссылки на правила
    - Наличие доказательств (attachments / keywords)
    - Язык сообщения
    - Полный и нормализованный текст
    """
    player_texts = []
    attachments_count = 0

    for msg in messages:
        if msg.get("role") == "player":
            player_texts.append(msg.get("text", ""))
            attachments_count += msg.get("attachments", 0)

    combined_text = " ".join(player_texts)
    norm_text = normalize_kz_text(combined_text)

    # 1. Поиск никнеймов
    nicks = NICK_PATTERN.findall(combined_text)
    nickname = nicks[0] if nicks else None

    # 2. Поиск ссылок на правила
    raw_refs = RULE_REF_PATTERN.findall(combined_text)
    rule_refs = list(dict.fromkeys(raw_refs))

    # 3. Наличие доказательств
    has_evidence_kw = any(kw in norm_text.lower() for kw in EVIDENCE_KEYWORDS)
    has_evidence = attachments_count > 0 or has_evidence_kw

    # 4. Язык
    lang = detect_language(combined_text)

    return {
        "nickname": nickname,
        "rule_refs": rule_refs,
        "has_evidence": has_evidence,
        "attachments_count": attachments_count,
        "lang": lang,
        "combined_text": combined_text,
        "normalized_text": norm_text,
    }
