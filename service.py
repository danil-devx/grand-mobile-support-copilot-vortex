"""
Основной сервис триажа обращений Grand Mobile Support Copilot.
Поддерживает:
1. Вызов LLM через OpenAI-совместимый API (OpenAI, Gemini, Groq, DeepSeek, OpenRouter)
2. Надёжный эвристический fallback-движок (работает без API-ключа или при сбое сети)

Строгое разделение типов обращений (Routing):
1. Вопросы по геймплею и навигации (report.question / report.help) — понятные инструкции с /gps, /stats, /help, без правил нарушений.
2. Жалобы и нарушения (complaint.* / ban) — правила нарушений из rules.json.
3. Прямой мат, токсичность и оскорбления в тикете — пресечение нарушений (правила 2.1, 3.2), выставление мута и эскалация администрации.
4. Оффтоп, флуд и бессмысленные сообщения — пресечение нецелевого использования (правило 3.1), предупреждение / мут репорта.
5. Кросс-канальная маршрутизация (Канал vs Суть текста):
   - Геймплейный вопрос в форму жалобы -> complaint.invalid, правило ЖБ 1.1.2, вежливое перенаправление в репорт.
   - Жалоба на игрока во внутриигровой репорт -> report.complaint, слежка + перенаправление в раздел жалоб.

Поддержка 6 языков: ru, kk, uz, en, es, pt.
Точные цитаты RAG из rules.json без галлюцинаций.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from rag import kb, extract_ticket_facts, normalize_kz_text
from prompt import SYSTEM_PROMPT, build_user_prompt

load_dotenv()

# Настройки LLM
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Инициализация клиента OpenAI при наличии ключа
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    except Exception as e:
        print(f"Предупреждение: не удалось инициализировать OpenAI клиент: {e}")


def _is_direct_abuse(norm: str, nickname: Optional[str] = None) -> bool:
    """
    Определяет, содержит ли сообщение прямой мат, токсичность или оскорбления
    в адрес администрации или в тексте тикета (а не жалобу на стороннего нарушителя).
    """
    norm = norm.strip().lower()

    # Мат и жесткие оскорбления
    profanity_patterns = [
        r'\b(хуй|хуя|хуе|хуи|пизд\w*|ебат\w*|ебал\w*|ебан\w*|блять|блядь|сук[аи]|сучк\w*|пидор\w*|пидар\w*|гандон\w*|гондон\w*|мудак\w*|долбоеб\w*|долбаеб\w*|шлюх\w*|даун\w*|лох\w*|урод\w*|тварь\w*|мраз\w*|уеб\w*)\b',
        r'\b(иди|пошел|пошли|соси|пососи)\s+(нахуй|в\s+пизду|хуй|член)\b',
        r'\b(ты|вы)\s+(даун\w*|лох\w*|дурак\w*|дебил\w*|пидор\w*|клоун\w*|мудак\w*|урод\w*|мраз\w*|тупой|тупые)\b',
        r'\b(админ\w*|хелпер\w*|разработчик\w*|саппорт\w*)\s+(лох\w*|пидор\w*|даун\w*|урод\w*|тупой|тупые|кончен\w*)\b',
        r'\b(sikey|onangni|chort|sikaman)\b',
        r'\b(fuck\s+you|bitch|stfu|asshole|idiot|dick)\b'
    ]
    has_profanity = any(re.search(pat, norm) for pat in profanity_patterns)
    if not has_profanity:
        return False

    # Явные прямые наезды на поддержку/админов: всегда прямой абьюз
    direct_attacks = [
        r'\b(ты|вы)\s+(даун\w*|лох\w*|дурак\w*|дебил\w*|пидор\w*|клоун\w*|мудак\w*|урод\w*|мраз\w*|тупой|тупые)\b',
        r'\b(админ\w*|хелпер\w*|саппорт\w*)\s+(лох\w*|пидор\w*|даун\w*|урод\w*|тупой|тупые|кончен\w*)\b',
        r'\b(иди|пошел|пошли|соси|пососи)\s+(нахуй|в\s+пизду|хуй|член)\b'
    ]
    if any(re.search(pat, norm) for pat in direct_attacks):
        return True

    # Проверяем, является ли это жалобой на другого игрока
    reporting_markers = [
        r'\b(игрок\w*|чел\w*|он\w*|нарушител\w*|айди|\bid\b|\bid\s*\d+)\b',
        r'\b(оскорбил\w*|оскнул\w*|обозвал\w*|написал\w*|сказал\w*|упомянул\w*)\b',
        r'\b(накаж\w*|забан\w*|замут\w*|проверьте|следите|посадите)\b',
        r'\b(скрин\w*|видео|доказ\w*|фрапс|тайм)\b'
    ]
    is_reporting_other = any(re.search(rm, norm) for rm in reporting_markers) or bool(nickname)
    return not is_reporting_other


def _is_nonsense_or_offtop(norm: str) -> bool:
    """
    Определяет бессмысленный оффтоп, флуд, клавиатурный спам, выпрашивание валюты/ТП или простое общение (п. 3.1).
    """
    norm = norm.strip().lower()
    if len(norm) <= 4 and norm not in ["дм", "дб", "вх", "гпс", "gps", "help", "жб"]:
        return True
    if re.search(r'(.)\1{3,}', norm):
        return True
    if any(km in norm for km in ["йцукен", "asdfg", "qwerty", "фывап", "zxcvb", "123456"]):
        return True
    # Выпрашивание или требования телепорта (п. 3.1)
    begging_patterns = [
        r'\b(дай|дайте|выдай\w*|подкинь\w*|накинь\w*)\s+(\w+\s+)?(денег|вирт\w*|рубл\w*|тачк\w*|машин\w*|донат\w*|ранг\w*|лвл\w*|админк\w*|гк)\b',
        r'\b(тп|телепорт\w*)\s+(меня|к|на|в)\b',
    ]
    if any(re.search(pat, norm) for pat in begging_patterns):
        return True
    # Простое общение / флуд / разговор вместо вопроса (оффтоп в репорт)
    chatting_patterns = [
        r'\b(привет\s+как\s+дела|че\s+делаешь|как\s+сам|поболтаем|расскажи\s+анекдот|ты\s+бот|кто\s+ты|ты\s+кто|ты\s+тут|ало\s+ты\s+тут|админ\s+ты\s+тут|скучно|мне\s+скучно)\b',
        r'^\s*(скучно|привет|ку|хай|куку|ау|пока|всем\s+ку|ало|алло|помогите|help)\s*$'
    ]
    if any(re.search(pat, norm) for pat in chatting_patterns):
        return True
    return False


def _is_complaint_intent(norm: str) -> bool:
    """
    Проверяет, содержит ли текст жалобу на нарушение правил или другого игрока.
    """
    complaint_pattern = re.compile(
        r'(\b(админ\w*|хелпер\w*|жалоб\w*|мут\w*|jail|деморган\w*)\b|\b(за)?(дм|дб|dm|db)\w*|'
        r'\b(чит\w*|софт\w*|аим\w*|рванка|вх|автокликер\w*|бот\w*|'
        r'убил\w*|застрелил\w*|таран\w*|стреля\w*|deathmatch|'
        r'оск\w*|оскорб\w*|родн\w*|мать|маму|шлюх\w*|сука|чмо|даун|sokyapti|'
        r'обман\w*|развод\w*|украл\w*|не отдал|кинул\w*|алдап|scam\w*|'
        r'реал|вирты|рубл\w*|тг|rmt|'
        r'мут просто так|посадил ни за что|варн ни за что|бан ни за что|разбан\w*|unban|banned)\b)',
        re.IGNORECASE
    )
    return bool(complaint_pattern.search(norm))


def _is_gameplay_question(norm: str) -> bool:
    """
    Проверяет, является ли текст игрока обычным вопросом по геймплею, навигации,
    механикам, работам или помощи, а не жалобой на нарушения или игроков.
    """
    # Если в сообщении есть прямой абьюз, оффтоп или жалоба на нарушение — это НЕ геймплейный вопрос
    if _is_direct_abuse(norm, None):
        return False

    if _is_nonsense_or_offtop(norm):
        return False

    if _is_complaint_intent(norm):
        return False

    if any(w in norm for w in ["админ", "хелпер", "деморган", "jail", "взлом", "взломали", "украли аккаунт", "пароль", "не могу зайти", "донат", "оплат", "не пришли гк", "деньги списали", "hacked", "donate"]):
        return False

    if re.search(r'\b(бан|бана|баны|забанен\w*|разбан\w*|варн\w*|мут\w*|жалоб\w*)\b', norm):
        return False

    # Маркеры вопросов по геймплею, навигации и базовой помощи
    gameplay_markers = [
        # Вопросительные маркеры
        "как", "где", "куда", "когда", "почему", "зачем", "сколько", "можно ли", "что делать",
        # Навигация и координаты
        "gps", "гпс", "навигатор", "карт", "найти", "место", "локаци", "где находится", "доехать", "добраться",
        # Государственные структуры и организации
        "полици", "мвд", "гибдд", "увд", "дпс", "участок", "полицейск",
        "больниц", "госпитал", "лечени", "врач", "медкарт", "вылечит",
        "мэри", "правительств", "губернатор", "фракци", "организац", "собеседован", "инвайт", "ранг", "повышен",
        # Автошкола и права
        "права", "автошкол", "лицензи", "экзамен", "сдать", "вождение",
        # Работы и заработок
        "шахт", "руд", "карьер", "завод", "ферм", "дальнобой", "грузчик", "работ", "устроит", "зарплат", "деньг", "доход",
        # Финансы
        "банк", "банкомат", "вклад", "депозит", "счет", "снять", "положить", "передать",
        # Транспорт
        "автосалон", "машин", "транспорт", "автомобил", "купить", "продать", "авторынок", "тюнинг", "бу рынок",
        # Прокачка
        "лвл", "уровен", "опыт", "exp", "стат", "скилл", "прокач",
        # Команды и справка
        "команд", "хелп", "помощ", "подскаж", "скаж", "инвентар", "квест",
        # KK
        "қалай", "қайда", "жұмыс", "ақша", "деңгей", "қанша", "көлік", "шахта", "банк", "полиция", "аурухана", "әкімдік",
        # UZ
        "qanday", "qayerda", "ish", "ishga", "daraja", "litsenziya", "qancha", "mashina", "bank", "shaxta", "politsiya", "shifoxona",
        # EN
        "how", "where", "what", "which", "job", "mine", "mining", "work", "level", "salary", "gps", "money", "buy", "car", "bank", "police", "hospital",
        # ES
        "cómo", "dónde", "trabajo", "mina", "nivel", "salario", "cuánto", "dinero", "coche", "auto", "banco", "policía", "policia", "hospital",
        # PT
        "como", "onde", "trabalho", "mina", "nível", "salário", "quanto", "dinheiro", "carro", "banco", "polícia", "policia", "hospital"
    ]
    return any(gm in norm for gm in gameplay_markers)


def _build_rules_cited(cited_refs: List[str]) -> List[Dict[str, str]]:
    """
    Формирует массив rules_cited с точными цитатами из rules.json.
    Каждый элемент: { "rule_id": "2.1", "quote": "Точная цитата из rules.json" }
    """
    result = []
    seen = set()
    for ref in cited_refs:
        if ref in seen:
            continue
        seen.add(ref)
        text = kb.get_rule_text(ref)
        if text:
            result.append({"rule_id": ref, "quote": text})
    return result


def _build_facts_extracted(
    kind: str,
    lang: str,
    facts: Dict[str, Any],
    cited_rules: List[str],
    is_mismatch: bool = False,
    intent_type: str = "complaint"
) -> List[str]:
    """
    Формирует структурированный список ключевых фактов, извлеченных из сообщения игрока.
    """
    items = []
    is_complaint = "comp" in str(kind).lower()
    kind_label = "Жалоба на игрока (COMPLAINT)" if is_complaint else "Репорт / Помощь (REPORT)"
    items.append(f"Канал обращения: {kind_label}")

    if intent_type == "gameplay":
        items.append("Суть обращения: Вопрос по навигации / игровому процессу / механике")
    elif intent_type == "help_stuck":
        items.append("Суть обращения: Просьба о технической помощи (застрял / баг)")
    elif intent_type == "direct_abuse":
        items.append("Суть обращения: Прямое оскорбление / ненормативная лексика в адрес проекта или администрации")
        items.append("Нарушение: Пункты 2.1 (оскорбления), 3.2 (хамство/неуважение к администрации)")
        items.append("Требуемое действие: Выдача MUTE от 15 до 60 минут или блокировка персонажа")
    elif intent_type == "offtop_spam":
        items.append("Суть обращения: Оффтоп / флуд / общение не по назначению репорта")
        items.append("Нарушение: Пункт 3.1 правил (использование репорта не по назначению)")
        items.append("Предупреждение: Наказание MUTE REPORT до 30 минут или предупреждение (варн)")
    else:
        items.append("Суть обращения: Заявление о нарушении правил игры другим игроком")

    if is_mismatch:
        items.append("Маршрутизация: Выявлено несоответствие выбранного канала и сути вопроса (требуется перенаправление)")

    if facts.get("nickname"):
        items.append(f"Указанный игрок / нарушитель: {facts['nickname']}")

    if facts.get("has_evidence"):
        att = facts.get("attachments_count", 0)
        if att > 0:
            items.append(f"Доказательства: прикреплены вложения ({att} шт.)")
        else:
            items.append("Доказательства: имеются ссылки/таймкоды в тексте сообщения")
    else:
        if intent_type not in ["gameplay", "direct_abuse", "offtop_spam"]:
            items.append("Доказательства: не приложены")

    if cited_rules:
        items.append(f"Затронутые пункты правил: {', '.join(cited_rules)}")
    else:
        if intent_type == "gameplay":
            items.append("Пункты правил: не применимы (обращение носит справочный характер)")

    items.append(f"Определённый язык: {lang}")
    return items


def _rule_based_triage(
    orig_kind: str,
    text: str,
    facts: Dict[str, Any],
    matched_rules: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Резервный (fallback) экспертный алгоритм триажа с поддержкой умного роутинга.
    Срабатывает мгновенно (< 5мс) без интернета и ключей.
    """
    norm = facts["normalized_text"].lower()
    lang = facts["lang"]
    nickname = facts["nickname"]
    has_evidence = facts["has_evidence"]

    reply_lang = lang if lang in ["ru", "kk", "uz", "en", "es", "pt"] else "ru"

    # =========================================================================
    # 0. ПРОВЕРКА НА ПРЯМОЙ МАТ, ТОКСИЧНОСТЬ И ОСКОРБЛЕНИЯ В АДРЕС ПРОЕКТА / АДМИНИСТРАЦИИ
    # =========================================================================
    if _is_direct_abuse(norm, nickname):
        category = "report.offtopic" if orig_kind != "complaint" else "complaint.invalid"
        cited_rules = ["2.1", "3.2"] if any(w in norm for w in ["админ", "хелпер", "вы", "саппорт", "сервер"]) else ["2.1"]
        priority = "p1"
        needs_human = True
        reasons = ["Обнаружено прямое оскорбление / ненормативная лексика в обращении. Нарушение правил 2.1, 3.2. Требуется выдача блокировки чата (MUTE) администратором."]

        draft_templates = {
            "ru": (
                "Внимание: Вы нарушаете правила проекта! В репорте и любых чатах строго запрещены ненормативная лексика, "
                "прямые и завуалированные оскорбления, а также хамство в адрес администрации (пункты 2.1 и 3.2 правил). "
                "За данное нарушение предусмотрена блокировка чата (MUTE от 15 до 60 минут) или блокировка персонажа. "
                "Обращение передано администрации для выдачи соответствующего наказания. "
                "Пожалуйста, соблюдайте субординацию и правила проекта."
            ),
            "kk": (
                "Назар аударыңыз: Сіз жоба ережелерін бұзып тұрсыз! Репортта және барлық чаттарда былапыт сөздер, "
                "қорлау және әкімшілікке құрметсіздік танытуға қатаң тыйым салынады (2.1 және 3.2 тармақтары). "
                "Бұл бұзушылық үшін чатты бұғаттау (15-тен 60 минутқа дейін MUTE) немесе кейіпкерді бұғаттау қарастырылған. "
                "Өтініш жаза қолдану үшін әкімшілікке берілді."
            ),
            "uz": (
                "Diqqat: Siz loyiha qoidalarini buzyapsiz! Report va barcha chatlarda behayo so'kinish, "
                "haqorat va ma'muriyatga hurmatsizlik qilish qat'iyan man etiladi (2.1 va 3.2-bandlar). "
                "Ushbu qoidabuzarlik uchun chat bloklanishi (15 dan 60 daqiqagacha MUTE) ko'zda tutilgan. "
                "Murojaat jazo qo'llash uchun ma'muriyatga topshirildi."
            ),
            "en": (
                "Warning: You are violating server rules! Profanity, direct or implied insults, "
                "and disrespect towards administration are strictly prohibited (rules 2.1 and 3.2). "
                "This violation results in a chat mute (MUTE from 15 to 60 minutes) or account suspension. "
                "Your ticket has been forwarded to administrators for appropriate sanctions."
            ),
            "es": (
                "¡Atención: Está infringiendo las reglas del servidor! El uso de lenguaje obsceno, "
                "insultos y falta de respeto a la administración están estrictamente prohibidos (reglas 2.1 y 3.2). "
                "Esta infracción conlleva el bloqueo del chat (MUTE de 15 a 60 minutos) o suspensión del personaje. "
                "El caso ha sido transferido a la administración para la sanción correspondiente."
            ),
            "pt": (
                "Atenção: Você está violando as regras do servidor! Palavrões, "
                "ofensas e desrespeito à administração são estritamente proibidos (regras 2.1 e 3.2). "
                "Esta infração acarreta o bloqueio do chat (MUTE de 15 a 60 minutos) ou suspensão do personagem. "
                "O tíquete foi transferido para a administração para sanção cabível."
            )
        }
        draft_text = draft_templates.get(reply_lang, draft_templates["ru"])
        rules_cited = _build_rules_cited(cited_rules)
        facts_extracted = _build_facts_extracted(orig_kind, reply_lang, facts, cited_rules, is_mismatch=False, intent_type="direct_abuse")
        return {
            "category": category,
            "lang": reply_lang,
            "priority": priority,
            "is_spam_or_duplicate": True,
            "facts_extracted": facts_extracted,
            "extracted": {"nickname": nickname, "rule_refs": cited_rules, "has_evidence": has_evidence, "server": None},
            "draft_reply": {"lang": reply_lang, "text": draft_text, "rules_cited": rules_cited, "cited": cited_rules},
            "needs_human": needs_human,
            "confidence": 0.95,
            "reasons": reasons
        }

    # =========================================================================
    # 0.1 ПРОВЕРКА НА БЕССМЫСЛЕННЫЙ ОФФТОП, ФЛУД ИЛИ ВЫПРАШИВАНИЕ (п. 3.1)
    # =========================================================================
    if _is_nonsense_or_offtop(norm):
        category = "report.offtopic"
        cited_rules = ["3.1"]
        priority = "p3"
        needs_human = False
        reasons = ["Оффтоп / флуд / нецелевое использование канала обращений. Нарушение пункта 3.1 правил."]

        draft_templates = {
            "ru": (
                "Внимание: Репорт используется не по назначению! Отправка оффтопа, бессмысленных сообщений, "
                "флуда, ведение простого общения вместо вопроса, а также просьбы о выдаче валюты или телепорте "
                "строго запрещены правилами сервера (пункт 3.1 правил). Подобные действия наказываются блокировкой репорта "
                "(MUTE REPORT до 30 минут) или предупреждением (варн). Пожалуйста, используйте репорт исключительно "
                "для конкретных игровых вопросов и навигации."
            ),
            "kk": (
                "Назар аударыңыз: Репорт өз мақсатында қолданылмай жатыр! Оффтоп, флуд, сұрақтың орнына жай сөйлесу "
                "және ақша немесе телепорт сұрауға ереже бойынша қатаң тыйым салынады (3.1 тармағы). "
                "Мұндай әрекеттер репортты бұғаттауға (MUTE REPORT 30 минутқа дейін) немесе варнға әкеледі. "
                "Өтінеміз, репортты тек ойын бойынша сұрақтар мен көмек үшін пайдаланыңыз."
            ),
            "uz": (
                "Diqqat: Report maqsadsiz ishlatilmoqda! Offtop, behuda xabarlar, savol o'rniga gaplashish "
                "hamda pul yoki teleport so'rash qat'iyan man etiladi (3.1-band). "
                "Bunday qoidabuzarlik report bloklanishiga (30 daqiqagacha MUTE REPORT) yoki ogohlantirishga (warn) sabab bo'ladi. "
                "Iltimos, reportdan faqat aniq o'yin savollari bo'yicha foydalaning."
            ),
            "en": (
                "Warning: The report system is being used improperly! Offtop, spam, casual chatting instead of asking questions, "
                "and begging for money or teleportation are strictly prohibited (rule 3.1). "
                "These actions will result in a report mute (MUTE REPORT up to 30 minutes) or a warning. "
                "Please use the report system solely for specific gameplay questions and navigation assistance."
            ),
            "es": (
                "¡Atención: El reporte se está usando de forma indebida! El offtop, spam, charlar en lugar de hacer una pregunta "
                "y pedir dinero o teletransporte están estrictamente prohibidos (regla 3.1). "
                "Estas acciones conllevan el bloqueo de reportes (MUTE REPORT hasta 30 minutos) o una advertencia. "
                "Utilice el reporte únicamente para dudas específicas sobre el juego."
            ),
            "pt": (
                "Atenção: O reporte está sendo usado de forma inadequada! Offtop, spam, conversas em vez de dúvidas "
                "e pedidos de dinheiro ou teletransporte são estritamente proibidos (regla 3.1). "
                "Essas ações acarretam o bloqueo de reportes (MUTE REPORT até 30 minutos) ou advertência (warn). "
                "Utilice o reporte apenas para dúvidas pontuais sobre o jogo."
            )
        }
        draft_text = draft_templates.get(reply_lang, draft_templates["ru"])
        rules_cited = _build_rules_cited(cited_rules)
        facts_extracted = _build_facts_extracted(orig_kind, reply_lang, facts, cited_rules, is_mismatch=False, intent_type="offtop_spam")
        return {
            "category": category,
            "lang": reply_lang,
            "priority": priority,
            "is_spam_or_duplicate": True,
            "facts_extracted": facts_extracted,
            "extracted": {"nickname": nickname, "rule_refs": cited_rules, "has_evidence": has_evidence, "server": None},
            "draft_reply": {"lang": reply_lang, "text": draft_text, "rules_cited": rules_cited, "cited": cited_rules},
            "needs_human": needs_human,
            "confidence": 0.90,
            "reasons": reasons
        }

    # =========================================================================
    # СЦЕНАРИЙ 1: Кросс-канальный Mismatch — Вопрос по геймплею/навигации через форму жалобы!
    # =========================================================================
    if orig_kind == "complaint" and _is_gameplay_question(norm):
        category = "complaint.invalid"
        cited_rules = ["ЖБ 1.1.2"]
        priority = "p3"
        needs_human = False
        reasons = ["Вопрос по геймплею подан через форму жалоб. Требуется вежливое перенаправление в репорт/поддержку."]

        draft_templates = {
            "ru": (
                "Здравствуйте! Для поиска нужных локаций, организаций и работ используйте встроенный навигатор: введите команду /gps. "
                "Также посмотреть уровень и статистику можно командой /stats, а список команд — /help.\n\n"
                "Обратите внимание: данный раздел предназначен исключительно для жалоб на нарушения других игроков (пункт 1.1.2 правил). "
                "Пожалуйста, задавайте игровые вопросы через внутриигровой репорт (/report > Вопрос) или в Discord-сообществе игры. Приятной игры!"
            ),
            "kk": (
                "Сәлеметсіз бе! Қажетті орындар мен жұмысты табу үшін /gps командасын енгізіп, навигатордан бағытты таңдаңыз. "
                "Кейіпкер деңгейін /stats, ал командаларды /help арқылы көре аласыз.\n\n"
                "Назар аударыңыз: бұл бөлім тек ойыншылардың ереже бұзушылықтары бойынша шағымдарға арналған (ереженің 1.1.2 тармағы). "
                "Ойын бойынша сұрақтарды /report мәзірі немесе ресми Discord арқылы қоюыңызды сұраймыз. Ойыныңыз сәтті өтсін!"
            ),
            "uz": (
                "Salom! Joylar va ishlarni topish uchun /gps buyrug'idan foydalaning. "
                "Darajangizni /stats, barcha buyruqlarni esa /help orqali bilib olishingiz mumkin.\n\n"
                "E'tibor bering: ushbu bo'lim faqat o'yinchilar ustidan shikoyatlar uchun mo'ljallangan (1.1.2-band). "
                "Iltimos, o'yin bo'yicha savollarni o'yin ichidagi /report menyusi orqali yuboring. Maroqli o'yin tilaymiz!"
            ),
            "en": (
                "Hello! To find locations and jobs, please use the in-game navigator: command /gps. "
                "You can also check your level and stats with /stats, and see all commands with /help.\n\n"
                "Please note: this section is intended exclusively for complaints against players (rule 1.1.2). "
                "For gameplay questions, please use the in-game report system (/report > Question) or our official Discord. Enjoy the game!"
            ),
            "es": (
                "¡Hola! Para encontrar ubicaciones y trabajos, usa el GPS del juego: comando /gps. "
                "También puedes ver tu nivel con /stats y los comandos con /help.\n\n"
                "Tenga en cuenta: esta sección está destinada exclusivamente a quejas sobre jugadores (regla 1.1.2). "
                "Para preguntas sobre el juego, utilice el reporte en el juego (/report > Pregunta). ¡Disfruta el juego!"
            ),
            "pt": (
                "Olá! Para encontrar locais e trabalhos, use o GPS do jogo: comando /gps. "
                "Você também pode conferir seu nível com /stats e os comandos com /help.\n\n"
                "Observe: esta seção é destinada exclusivamente a denúncias de jogadores (regra 1.1.2). "
                "Para dúvidas sobre o jogo, use o reporte no jogo (/report > Dúvida). Bom jogo!"
            )
        }
        draft_text = draft_templates.get(reply_lang, draft_templates["ru"])

        rules_cited = _build_rules_cited(cited_rules)
        facts_extracted = _build_facts_extracted(orig_kind, reply_lang, facts, cited_rules, is_mismatch=True, intent_type="gameplay")

        return {
            "category": category,
            "lang": reply_lang,
            "priority": priority,
            "is_spam_or_duplicate": False,
            "facts_extracted": facts_extracted,
            "extracted": {
                "nickname": nickname,
                "rule_refs": cited_rules,
                "has_evidence": has_evidence,
                "server": None
            },
            "draft_reply": {
                "lang": reply_lang,
                "text": draft_text,
                "rules_cited": rules_cited,
                "cited": cited_rules
            },
            "needs_human": needs_human,
            "confidence": 0.90,
            "reasons": reasons
        }

    # =========================================================================
    # СЦЕНАРИЙ 2: Вопрос по геймплею, навигации, механикам в репорте / помощи
    # =========================================================================
    if orig_kind in ["report", "question", "help"] and _is_gameplay_question(norm):
        # 1. Если игрок застрял в текстурах
        if any(w in norm for w in ["застрял", "текстур", "не могу пошевелит", "вытащите", "stuck", "tiqilib"]):
            category = "report.help"
            priority = "p2"
            needs_human = False
            reasons = ["Просьба о помощи с освобождением персонажа из текстур."]
            help_templates = {
                "ru": "Здравствуйте! Если вы застряли в текстурах или транспорте, администрация уже спешит на помощь! Также вы можете попробовать ввести команду /spback или дождаться помощи дежурного администратора.",
                "kk": "Сәлеметсіз бе! Егер кейіпкеріңіз текстурада қалып қойса, әкімшілік көмекке келеді! Қозғала алмасаңыз, /spback командасын қолданып көріңіз.",
                "uz": "Salom! Agar qahramoningiz teksturada qolib ketgan bo'lsa, navbatchi ma'muriyat yordam beradi!",
                "en": "Hello! If you are stuck in game textures, admin assistance is on the way. You can also try using the /spback command.",
                "es": "¡Hola! Si tu personaje está atrapado en las texturas, un administrador te ayudará en breve.",
                "pt": "Olá! Se o seu personagem estiver preso nas texturas, um administrador logo irá ajudá-lo."
            }
            draft_text = help_templates.get(reply_lang, help_templates["ru"])
            cited_rules = []
            rules_cited = []
            facts_extracted = _build_facts_extracted(orig_kind, reply_lang, facts, cited_rules, is_mismatch=False, intent_type="help_stuck")
            return {
                "category": category,
                "lang": reply_lang,
                "priority": priority,
                "is_spam_or_duplicate": False,
                "facts_extracted": facts_extracted,
                "extracted": {"nickname": nickname, "rule_refs": [], "has_evidence": has_evidence, "server": None},
                "draft_reply": {"lang": reply_lang, "text": draft_text, "rules_cited": rules_cited, "cited": []},
                "needs_human": needs_human,
                "confidence": 0.88,
                "reasons": reasons
            }

        # 2. Игровые механики, навигация, работы, организации
        category = "report.question"
        cited_rules = []  # Правила нарушений категорически не применяются
        priority = "p3"
        needs_human = False
        reasons = ["Вопрос по навигации / игровым механикам классифицирован как report.question, подготовлена четкая инструкция с /gps."]

        # 2.1 Шахта и добыча руды
        if any(w in norm for w in ["шахт", "руд", "карьер", "ископаем"]):
            job_templates = {
                "ru": (
                    "Здравствуйте! Чтобы устроиться и работать на Шахте, используйте встроенный навигатор: введите команду /gps "
                    "и выберите раздел «Поиск работы» → «Шахта» (на карте появится метка). "
                    "Работа на шахте доступна с начального уровня (1 lvl) без предварительного опыта. "
                    "Чтобы начать добычу руды, подойдите к маркеру взаимодействия в карьере. "
                    "Узнать текущий баланс, заработанные деньги и уровень персонажа можно через команду /stats, "
                    "а список всех доступных команд игры — /help. Желаем приятной и прибыльной игры на Grand Mobile!"
                ),
                "kk": (
                    "Сәлеметсіз бе! Шахтаға жұмысқа тұру үшін /gps командасын енгізіп, «Жұмыс іздеу» → «Шахта» бөлімін таңдаңыз. "
                    "Шахтада бастапқы 1-деңгейден бастап жұмыс істеуге болады. "
                    "Кейіпкер деңгейін және табысты /stats, ал барлық командаларды /help арқылы тексере аласыз. Ойыныңыз сәтті өтсін!"
                ),
                "uz": (
                    "Salom! Shaxtaga ishga joylashish uchun /gps buyrug'ini kiriting va «Ish qidirish» → «Shaxta» bo'limini tanlang. "
                    "Shaxtada boshlang'ich 1-darajadan boshlab ishlashingiz mumkin. "
                    "Darajangizni va maoshingizni /stats, barcha buyruqlar ro'yxatini esa /help orqali ko'rishingiz mumkin. "
                    "Grand Mobile-da maroqli o'yin tilaymiz!"
                ),
                "en": (
                    "Hello! To get a job at the Mine, open the in-game GPS: enter /gps → «Jobs» → «Mine» (a marker will appear on your map). "
                    "Working at the mine is available from level 1 without prior experience. Approach the interaction marker at the quarry to start mining. "
                    "Check your stats and salary with /stats, and see all commands with /help. Enjoy Grand Mobile!"
                ),
                "es": (
                    "¡Hola! Para conseguir trabajo en la Mina, usa el GPS: ingresa el comando /gps → «Búsqueda de trabajo» → «Mina». "
                    "El trabajo en la mina está disponible desde el nivel 1. "
                    "Verifica tu nivel y salario con /stats, y consulta todos los comandos con /help. ¡Buen juego en Grand Mobile!"
                ),
                "pt": (
                    "Olá! Para trabalhar na Mina, use o GPS do jogo: comando /gps → «Buscar emprego» → «Mina». "
                    "O trabalho na mina está disponível a partir do nível 1. "
                    "Confira seu saldo e nível com /stats, e veja todos os comandos com /help. Bom jogo no Grand Mobile!"
                )
            }
            draft_text = job_templates.get(reply_lang, job_templates["ru"])

        # 2.2 Полиция / МВД / ГИБДД
        elif any(w in norm for w in ["полици", "мвд", "гибдд", "увд", "дпс", "участок", "police"]):
            police_templates = {
                "ru": (
                    "Здравствуйте! Чтобы добраться до ближайшего отделения полиции (МВД / ГИБДД), откройте навигатор: введите команду /gps "
                    "и выберите раздел «Государственные организации» → «Полиция (МВД / ГИБДД)». На вашей карте появится метка с маршрутом. "
                    "В отделе полиции вы можете оплатить штрафы, подать заявление или узнать критерии для поступления на службу. "
                    "Проверить статус персонажа: /stats, список команд: /help. Приятной игры!"
                ),
                "kk": (
                    "Сәлеметсіз бе! Жақын маңдағы полиция бөлімін (ІІМ / ЖҚП) табу үшін /gps командасын енгізіп, "
                    "«Мемлекеттік ұйымдар» → «Полиция» бөлімін таңдаңыз. Картада бағыт пайда болады. "
                    "Полицияда айыппұл төлеуге және қызметке орналасу шарттарын білуге болады. Командалар: /help."
                ),
                "uz": (
                    "Salom! Yaqin atrofdagi politsiya bo'limini (IIV / YPX) topish uchun /gps buyrug'ini kiriting va "
                    "«Davlat tashkilotlari» → «Politsiya» bo'limini tanlang. Xaritada yo'nalish belgilanadi. "
                    "U yerda jarimalarni to'lash va xizmatga kirish shartlarini bilish mumkin. Buyruqlar: /help."
                ),
                "en": (
                    "Hello! To find the nearest Police Department (PD / Traffic Police), enter /gps → «Government Organizations» → «Police Department». "
                    "A navigation marker will appear on your map. You can pay fines and apply for service there. "
                    "Check your status with /stats, commands: /help."
                ),
                "es": (
                    "¡Hola! Para encontrar la comisaría de Policía más cercana, usa el GPS: ingresa /gps → «Organizaciones Estatales» → «Policía». "
                    "Aparecerá un marcador en tu mapa. Allí puedes pagar multas y solicitar unirte. Comandos: /help."
                ),
                "pt": (
                    "Olá! Para encontrar a delegacia de Polícia mais próxima, use o GPS: comando /gps → «Organizações Governamentais» → «Polícia». "
                    "Um marcador aparecerá no seu mapa. Lá você pode pagar multas e se alistar. Comandos: /help."
                )
            }
            draft_text = police_templates.get(reply_lang, police_templates["ru"])

        # 2.3 Больница / Центральная городская больница
        elif any(w in norm for w in ["больниц", "госпитал", "лечени", "врач", "медкарт", "hospital"]):
            hospital_templates = {
                "ru": (
                    "Здравствуйте! Найти Центральную городскую больницу можно через встроенный навигатор: введите команду /gps "
                    "и выберите «Государственные организации» → «Больница (ЦБ)». На карте появится маршрут до медицинского центра. "
                    "В больнице вы можете восстановить здоровье, оформить медицинскую карту у дежурного врача и приобрести рецепты. "
                    "Узнать статистику персонажа: /stats, помощь по командам: /help."
                ),
                "kk": (
                    "Сәлеметсіз бе! Орталық қалалық аурухананы табу үшін /gps командасын енгізіп, «Мемлекеттік ұйымдар» → «Аурухана» бөлімін таңдаңыз. "
                    "Ауруханада денсаулықты қалпына келтіруге және медициналық карта алуға болады. Командалар: /help."
                ),
                "uz": (
                    "Salom! Markaziy shahar shifoxonasini topish uchun /gps buyrug'ini kiriting va «Davlat tashkilotlari» → «Shifoxona» bo'limini tanlang. "
                    "Shifoxonada sog'ligingizni tiklashingiz va tibbiy karta olishingiz mumkin. Buyruqlar: /help."
                ),
                "en": (
                    "Hello! To find the Central Hospital, open the GPS: enter /gps → «Government Organizations» → «Hospital». "
                    "Follow the map marker. At the hospital, you can heal, receive a medical card, and consult doctors. Stats: /stats, commands: /help."
                ),
                "es": (
                    "¡Hola! Para encontrar el Hospital Central, usa el GPS: ingresa /gps → «Organizaciones Estatales» → «Hospital». "
                    "En el hospital puedes curarte y obtener tu tarjeta médica. Todos los comandos: /help."
                ),
                "pt": (
                    "Olá! Para encontrar o Hospital Central, use o GPS: comando /gps → «Organizações Governamentais» → «Hospital». "
                    "Lá você pode recuperar sua saúde e obter sua ficha médica. Todos os comandos: /help."
                )
            }
            draft_text = hospital_templates.get(reply_lang, hospital_templates["ru"])

        # 2.4 Банки и банкоматы
        elif any(w in norm for w in ["банк", "банкомат", "вклад", "депозит", "счет", "снять деньги", "bank"]):
            bank_templates = {
                "ru": (
                    "Здравствуйте! Ближайшее отделение банка можно найти через навигатор: введите команду /gps "
                    "и выберите раздел «Банки и банкоматы». В банке вы можете открыть вклад под процент, оплатить налоги "
                    "за дом или бизнес и снять заработную плату. Проверить баланс счета можно через /stats или в приложении банка в телефоне персонажа."
                ),
                "kk": (
                    "Сәлеметсіз бе! Жақын маңдағы банкті табу үшін /gps командасын енгізіп, «Банктер және банкоматтар» бөлімін таңдаңыз. "
                    "Банкте салым ашуға, салықтарды төлеуге және жалақыны шешіп алуға болады. Балансты /stats арқылы көре аласыз."
                ),
                "uz": (
                    "Salom! Yaqin atrofdagi bankni topish uchun /gps buyrug'ini kiriting va «Banklar va bankomatlar» bo'limini tanlang. "
                    "Bankda omonat ochish, soliq to'lash va oylikni yechish mumkin. Hisobingizni /stats orqali tekshiring."
                ),
                "en": (
                    "Hello! You can find the nearest bank using the in-game GPS: enter /gps → «Banks & ATMs». "
                    "At the bank, you can open a deposit account, pay taxes, and withdraw your salary. Check your balance with /stats."
                ),
                "es": (
                    "¡Hola! Puedes encontrar el banco más cercano usando el GPS del juego: comando /gps → «Bancos y cajeros». "
                    "En el banco puedes abrir un depósito, pagar impuestos y retirar tu salario. Consulta tu saldo con /stats."
                ),
                "pt": (
                    "Olá! Encontre o banco mais próximo pelo GPS: comando /gps → «Bancos e caixas eletrônicos». "
                    "No banco você pode abrir depósito, pagar impostos e sacar seu salário. Veja seu saldo com /stats."
                )
            }
            draft_text = bank_templates.get(reply_lang, bank_templates["ru"])

        # 2.5 Мэрия / Правительство
        elif any(w in norm for w in ["мэри", "правительств", "губернатор", "government"]):
            gov_templates = {
                "ru": (
                    "Здравствуйте! Чтобы найти здание Правительства (Мэрию), откройте навигатор: введите команду /gps "
                    "и выберите раздел «Государственные организации» → «Правительство (Мэрия)». "
                    "В Правительстве решаются государственные вопросы, регистрируются браки и проводятся приёмы граждан. "
                    "Статистика персонажа: /stats, список всех команд: /help. Приятной игры!"
                ),
                "kk": (
                    "Сәлеметсіз бе! Үкімет (Әкімдік) ғимаратын табу үшін /gps командасын енгізіп, «Мемлекеттік ұйымдар» → «Үкімет» бөлімін таңдаңыз. "
                    "Кейіпкер деңгейін /stats, ал командаларды /help арқылы көре аласыз."
                ),
                "uz": (
                    "Salom! Hukumat (Hokimiyat) binosini topish uchun /gps buyrug'ini kiriting va «Davlat tashkilotlari» → «Hukumat» bo'limini tanlang. "
                    "Darajangizni /stats, buyruqlarni esa /help orqali bilib oling."
                ),
                "en": (
                    "Hello! To find City Hall (Government), open the GPS: enter /gps → «Government Organizations» → «Government». "
                    "Check character stats with /stats and commands with /help."
                ),
                "es": (
                    "¡Hola! Para encontrar el Ayuntamiento (Gobierno), usa el GPS: ingresa /gps → «Organizaciones Estatales» → «Gobierno». "
                    "Consulta estadísticas con /stats y comandos con /help."
                ),
                "pt": (
                    "Olá! Para encontrar a Prefeitura (Governo), use o GPS: comando /gps → «Organizações Governamentais» → «Governo». "
                    "Confira status com /stats e comandos com /help."
                )
            }
            draft_text = gov_templates.get(reply_lang, gov_templates["ru"])

        # 2.6 Автошкола и права
        elif any(w in norm for w in ["автошкол", "права", "лицензи", "экзамен", "вожден"]):
            autoschool_templates = {
                "ru": (
                    "Здравствуйте! Автошколу для сдачи экзамена на права можно найти через встроенный навигатор: "
                    "введите команду /gps и выберите «Государственные организации / Важные места» → «Автошкола». "
                    "В здании автошколы необходимо пройти теоретический тест и практический экзамен по вождению. "
                    "Проверить наличие имеющихся лицензий можно через команду /stats. Список команд: /help."
                ),
                "kk": (
                    "Сәлеметсіз бе! Жүргізуші куәлігін алу үшін автомектепті /gps → «Автомектеп» бөлімінен таба аласыз. "
                    "Теориялық және практикалық емтихан тапсыру қажет. Лицензияларды /stats арқылы тексеріңіз."
                ),
                "uz": (
                    "Salom! Haydovchilik guvohnomasini olish uchun avtomaktabni /gps → «Avtomaktab» orqali topishingiz mumkin. "
                    "Nazariy va amaliy imtihon topshiriladi. Litsenziyalaringizni /stats orqali tekshiring."
                ),
                "en": (
                    "Hello! You can find the Driving School via GPS: enter /gps → «Driving School». "
                    "Pass the theory and practical driving tests to earn your license. Check licenses with /stats."
                ),
                "es": (
                    "¡Hola! Puedes encontrar la Autoescuela con el GPS: comando /gps → «Autoescuela». "
                    "Aprueba el examen teórico y práctico para obtener tu licencia. Consulta con /stats."
                ),
                "pt": (
                    "Olá! Encontre a Autoescola pelo GPS: comando /gps → «Autoescola». "
                    "Passe nos exames teórico e prático para obter sua carteira de motorista. Veja com /stats."
                )
            }
            draft_text = autoschool_templates.get(reply_lang, autoschool_templates["ru"])

        # 2.7 Автосалоны и покупка авто
        elif any(w in norm for w in ["автосалон", "авторынок", "бу рынок", "купить машину", "автомобил"]):
            autosalon_templates = {
                "ru": (
                    "Здравствуйте! Чтобы выбрать и приобрести автомобиль, откройте навигатор: введите команду /gps "
                    "и перейдите в раздел «Автосалоны и авторынки». Вы можете выбрать автосалон низкого, среднего или высокого класса, "
                    "либо отправиться на Б/У авторынок для покупки авто у других игроков. Проверить баланс и слоты: /stats."
                ),
                "kk": (
                    "Сәлеметсіз бе! Көлік сатып алу үшін /gps командасын енгізіп, «Автосалондар және автонарықтар» бөлімін таңдаңыз. "
                    "Қаражатыңызды /stats арқылы тексере аласыз."
                ),
                "uz": (
                    "Salom! Mashina sotib olish uchun /gps buyrug'ini kiriting va «Avtosalonlar va avtobozor» bo'limini tanlang. "
                    "Mablag'ingizni /stats orqali tekshiring."
                ),
                "en": (
                    "Hello! To buy a car, use the GPS: enter /gps → «Car Dealerships & Car Market». "
                    "You can choose between low, medium, high class salons or the used car market. Stats: /stats."
                ),
                "es": (
                    "¡Hola! Para comprar un coche, usa el GPS: ingresa /gps → «Concesionarios de autos y mercado». "
                    "Puedes elegir entre varias clases de vehículos. Consulta tu saldo con /stats."
                ),
                "pt": (
                    "Olá! Para comprar um carro, use o GPS: comando /gps → «Concessionárias e feira de veículos». "
                    "Escolha o modelo de acordo com seu orçamento. Veja saldo com /stats."
                )
            }
            draft_text = autosalon_templates.get(reply_lang, autosalon_templates["ru"])

        # 2.8 Общая навигация и команды игры
        else:
            draft_templates = {
                "ru": (
                    "Здравствуйте! Для поиска нужной локации или работы используйте встроенный навигатор: введите команду /gps "
                    "и выберите интересующий раздел (Поиск работы, Государственные организации, Банки, Автосалоны). "
                    "На миникарте появится красный маркер. Уровень персонажа и текущую зарплату можно посмотреть командой /stats, "
                    "а список всех доступных команд — /help. Приятной игры на Grand Mobile!"
                ),
                "kk": (
                    "Сәлеметсіз бе! Қажетті орынды немесе жұмысты табу үшін ойын навигаторын қолданыңыз: /gps командасын енгізіп, "
                    "қажетті бағытты таңдаңыз. Кейіпкер деңгейі мен жалақысын /stats командасымен, ал барлық қолжетімді командаларды /help арқылы көре аласыз. "
                    "Grand Mobile-да сәтті ойын тілейміз!"
                ),
                "uz": (
                    "Salom! Kerakli ish yoki joyni topish uchun o'yin navigatoridan foydalaning: /gps buyrug'ini kiriting va kerakli bo'limni tanlang. "
                    "Qahramon darajasi va maoshini /stats buyrug'i orqali, barcha buyruqlar ro'yxatini esa /help orqali ko'rishingiz mumkin. "
                    "Grand Mobile-da maroqli o'yin tilaymiz!"
                ),
                "en": (
                    "Hello! To find any location or job, please use the in-game GPS: enter /gps and choose the desired category (Jobs, Bank, Government). "
                    "A marker will guide you on the map. Check your level and stats with /stats, and see all available commands with /help. "
                    "Enjoy playing Grand Mobile!"
                ),
                "es": (
                    "¡Hola! Para encontrar cualquier ubicación o trabajo, usa el GPS del juego: ingresa el comando /gps y elige la categoría deseada (Búsqueda de trabajo, Banco, Gobierno). "
                    "Puedes verificar tu nivel y estadísticas con /stats, y ver todos los comandos con /help. "
                    "¡Disfruta jugando en Grand Mobile!"
                ),
                "pt": (
                    "Olá! Para encontrar qualquer local ou trabalho, use o GPS do jogo: digite o comando /gps e escolha a categoria desejada (Buscar emprego, Banco, Governo). "
                    "Você pode conferir seu nível e estatísticas com /stats, и ver todos os comandos com /help. "
                    "Bom jogo no Grand Mobile!"
                )
            }
            draft_text = draft_templates.get(reply_lang, draft_templates["ru"])

        rules_cited = []
        facts_extracted = _build_facts_extracted(orig_kind, reply_lang, facts, cited_rules, is_mismatch=False, intent_type="gameplay")

        return {
            "category": category,
            "lang": reply_lang,
            "priority": priority,
            "is_spam_or_duplicate": False,
            "facts_extracted": facts_extracted,
            "extracted": {
                "nickname": nickname,
                "rule_refs": [],
                "has_evidence": has_evidence,
                "server": None
            },
            "draft_reply": {
                "lang": reply_lang,
                "text": draft_text,
                "rules_cited": rules_cited,
                "cited": []
            },
            "needs_human": needs_human,
            "confidence": 0.88,
            "reasons": reasons
        }

    # =========================================================================
    # СЦЕНАРИЙ 3: Кросс-канальный Mismatch — Жалоба на нарушение через репорт / вопрос
    # =========================================================================
    if orig_kind in ["report", "question", "help"] and _is_complaint_intent(norm):
        category = "report.complaint"
        cited_rules = []

        if any(w in norm for w in ["чит", "софт", "аим", "рванка", "вх", "soft", "cheat"]):
            cited_rules = ["5.2"]
        elif re.search(r'\b(за)?(дм|дб|dm|db)\w*', norm) or any(w in norm for w in ["убил", "застрелил", "таран"]):
            cited_rules = ["5.4"]
            if any(w in norm for w in ["зз", "зелен", "gz"]):
                cited_rules.append("GZ")
        elif any(w in norm for w in ["обман", "развод", "украл", "не отдал", "алдап", "scam"]):
            cited_rules = ["4.1"]
        elif any(w in norm for w in ["оскорб", "родн", "мать", "оск", "сука", "чмо"]):
            cited_rules = ["2.1"]

        for r in facts["rule_refs"]:
            if kb.get_rule_text(r) and r not in cited_rules:
                cited_rules.append(r)

        rule_str = f" ({', '.join(cited_rules)})" if cited_rules else ""
        priority = "p2"
        needs_human = True
        reasons = ["Жалоба на нарушение подана через внутриигровой репорт. Направлена дежурной администрации, требуется перенаправление на форум при наличии видео."]

        draft_templates = {
            "ru": (
                f"Здравствуйте! Ваше обращение передано дежурным администраторам для наблюдения за игроком в игре. "
                f"Если у вас имеются видеодоказательства нарушения{rule_str}, пожалуйста, оформите официальную жалобу "
                "в специальном разделе жалоб в лаунчере или на форуме для проверки логов и выдачи строгого наказания."
            ),
            "kk": (
                f"Сәлеметсіз бе! Ақпарат ойыншыны бақылау үшін кезекші әкімшілікке берілді. "
                f"Егер сізде ереже бұзушылық бойынша бейнедәлелдер болса{rule_str}, "
                "толық тексеру және жаза қолдану үшін лаунчердегі арнайы шағымдар бөліміне жүгінуіңізді сұраймыз."
            ),
            "uz": (
                f"Salom! O'yinchini kuzatish uchun ma'lumot navbatchi administratorlarga uzatildi. "
                f"Agar sizda qoidabuzarlik bo'yicha dalillar bo'lsa{rule_str}, "
                "qat'iy jazo qo'llash uchun iltimos, maxsus shikoyatlar bo'limiga rasmiy ariza qoldiring."
            ),
            "en": (
                f"Hello! The report has been passed to active duty administrators to monitor the player. "
                f"If you have recorded evidence of the violation{rule_str}, "
                "please submit an official complaint in the dedicated complaints section in the launcher for log review and sanctions."
            ),
            "es": (
                f"¡Hola! El reporte ha sido transferido a la administración de turno para monitorear al jugador. "
                f"Si tiene pruebas en video de la infracción{rule_str}, "
                "por favor presente una queja oficial en la sección de denuncias para la revisión de registros y sanciones."
            ),
            "pt": (
                f"Olá! O reporte foi transferido para os administradores de plantão monitorarem o jogador. "
                f"Se você tiver provas em vídeo da infração{rule_str}, "
                "envie uma denúncia oficial na seção de denúncias para análise e sanções formais."
            )
        }
        draft_text = draft_templates.get(reply_lang, draft_templates["ru"])

        rules_cited = _build_rules_cited(cited_rules)
        facts_extracted = _build_facts_extracted(orig_kind, reply_lang, facts, cited_rules, is_mismatch=True, intent_type="complaint")

        return {
            "category": category,
            "lang": reply_lang,
            "priority": priority,
            "is_spam_or_duplicate": False,
            "facts_extracted": facts_extracted,
            "extracted": {
                "nickname": nickname,
                "rule_refs": cited_rules,
                "has_evidence": has_evidence,
                "server": None
            },
            "draft_reply": {
                "lang": reply_lang,
                "text": draft_text,
                "rules_cited": rules_cited,
                "cited": cited_rules
            },
            "needs_human": needs_human,
            "confidence": 0.85,
            "reasons": reasons
        }

    # =========================================================================
    # СЦЕНАРИЙ 4: Обычная классификация (Жалобы в complaint, тех. помощь в report)
    # =========================================================================
    kind = orig_kind if orig_kind == "complaint" else "report"
    category = "complaint.other" if kind == "complaint" else "report.question"
    cited_rules = []

    if kind == "complaint":
        # Жалобы на администрацию — высокий приоритет
        if any(w in norm for w in ["админ", "хелпер", "администратор", "мут просто так", "посадил ни за что", "бан ни за что", "варн ни за что"]):
            category = "complaint.admin"
            cited_rules = ["3.4"]
        elif any(w in norm for w in ["чит", "софт", "аим", "рванка", "бот", "автокликер", "вх", "soft", "cheat", "aimbot"]):
            category = "complaint.cheat"
            cited_rules = ["5.2"]
        elif re.search(r'\b(за)?(дм|дб|dm|db)\w*', norm) or any(w in norm for w in ["убил", "застрелил", "таран", "нанесение урона", "deathmatch"]):
            category = "complaint.dm"
            cited_rules = ["5.4"]
            if any(w in norm for w in ["зз", "зелен", "gz", "green"]):
                cited_rules.append("GZ")
        elif any(w in norm for w in ["обман", "развод", "украл", "не отдал", "кинул на", "алдап", "scam"]):
            category = "complaint.scam"
            cited_rules = ["4.1", "4.2"]
        elif any(w in norm for w in ["реал", "рубл", "тг", "телеграм", "продажа за", "купил вирты", "аккаунт за", "rmt"]):
            category = "complaint.rmt_ads"
            cited_rules = ["4.4", "2.3"]
        elif any(w in norm for w in ["оскорб", "мать", "родн", "сын", "даун", "шлюх", "оск", "сука", "чмо", "sokyapti"]):
            category = "complaint.insult"
            cited_rules = ["2.1"]
        elif len(norm.strip()) < 5 or (not has_evidence and not nickname):
            category = "complaint.invalid"
            cited_rules = ["ЖБ 1.1.2"]
    else:  # report / question
        if any(w in norm for w in ["застрял", "текстур", "помогите выбраться", "спавн", "тп", "перевернулся", "stuck"]):
            category = "report.help"
            cited_rules = ["3.1"]
        elif any(w in norm for w in ["донат", "оплат", "гк", "купил", "не пришли", "чек", "деньги списали", "donate"]):
            category = "report.payment"
            cited_rules = []
        elif any(w in norm for w in ["взлом", "пароль", "пропал", "украли", "не могу зайти", "аккаунт", "hacked", "account"]):
            category = "report.account"
            cited_rules = ["1.4"]
        elif any(w in norm for w in ["привет", "ау", "дай", "помогите", "пж", "hello", "hi", "pls", "help"]) and len(norm.split()) <= 2:
            category = "report.offtopic"
            cited_rules = ["3.1"]
        else:
            category = "report.question"

    # Добавляем правила, которые указал игрок в тексте обращения
    for r in facts["rule_refs"]:
        if kb.get_rule_text(r) and r not in cited_rules:
            cited_rules.append(r)

    # Привязка категории к явно указанным пунктам правил из taxonomy.json
    for r in cited_rules:
        if r == "5.2" and kind == "complaint":
            category = "complaint.cheat"
        elif r in ["5.3", "5.4", "GZ"] and kind == "complaint":
            category = "complaint.dm"
        elif r in ["2.1", "2.4", "2.6"] and kind == "complaint":
            category = "complaint.insult"
        elif r in ["4.1", "4.2"] and kind == "complaint":
            category = "complaint.scam"
        elif r in ["2.3", "4.4"] and kind == "complaint":
            category = "complaint.rmt_ads"
        elif r == "3.4" and kind == "complaint":
            category = "complaint.admin"
        elif r == "1.4" and kind == "report":
            category = "report.account"

    # Определение приоритета
    if category in ["report.account", "report.payment"] or "взлом" in norm:
        priority = "p0"
    elif category in ["complaint.cheat", "complaint.scam", "complaint.rmt_ads", "complaint.admin"]:
        priority = "p1"
    elif kind == "complaint" and has_evidence:
        priority = "p1"
    elif category in ["complaint.insult", "complaint.dm", "report.help", "report.complaint"]:
        priority = "p2"
    else:
        priority = "p3"

    # Нужен ли человек (needs_human)
    needs_human = False
    reasons = []

    if category in kb.needs_human_always:
        needs_human = True
        reasons.append(f"Категория {category} требует обязательного рассмотрения администратором")
    elif kind == "complaint":
        needs_human = True
        reasons.append("Решение о выдаче наказания игроку принимает администратор")
    elif "разбан" in norm or "бан" in norm or "unban" in norm:
        needs_human = True
        reasons.append("Вопросы блокировок и апелляций рассматриваются администратором")

    # Формирование черновика ответа на языке игрока
    draft_text = ""

    if reply_lang == "kk":
        if kind == "complaint":
            target = f" {nickname} ойыншысына" if nickname else ""
            rule_str = f" ({', '.join(cited_rules)} тармағы бойынша)" if cited_rules else ""
            ev_str = " Дәлелдер тіркелді." if has_evidence else " Жаза қолдану үшін дәлелдер қажет."
            draft_text = (
                f"Сәлеметсіз бе! Шағымыңыз{target} қабылданды{rule_str}.{ev_str} "
                "Шешімді сервер әкімшісі қабылдайды."
            )
        else:
            if category == "report.help":
                draft_text = "Сәлеметсіз бе! Әкімшілік көмек көрсету үшін репортты қабылдады. Өтінеміз, күте тұрыңыз."
            elif category == "report.payment":
                draft_text = "Сәлеметсіз бе! Төлемдерге қатысты мәселелер техникалық қолдау бөліміне тексеруге жіберілді."
            elif category == "report.account":
                draft_text = "Сәлеметсіз бе! Аккаунт бойынша өтініш тіркеліп, әкімшіге логтарды тексеруге берілді (ереженің 1.4 тармағы)."
            elif category == "report.offtopic":
                draft_text = "Сәлеметсіз бе! Репортты тек ойын мәселелері бойынша пайдалануыңызды сұраймыз (ереженің 3.1 тармағы)."
            else:
                draft_text = "Сәлеметсіз бе! Өтінішіңіз үшін рақмет. Сұрақ кезекші әкімшілерге берілді."

    elif reply_lang == "uz":
        if kind == "complaint":
            target = f" o'yinchi {nickname} ustidan" if nickname else ""
            rule_str = f" ({', '.join(cited_rules)}-band)" if cited_rules else ""
            ev_str = " Dalillar qabul qilindi." if has_evidence else " Jazoni qo'llash uchun dalillar zarur."
            draft_text = (
                f"Salom! Sizning shikoyatingiz{target} ko'rib chiqish uchun qabul qilindi{rule_str}.{ev_str} "
                "Yakuniy qarorni server administratori qabul qiladi."
            )
        else:
            if category == "report.help":
                draft_text = "Salom! Administrator sizga yordam berish uchun kelmoqda. Iltimos, joyingizda qoling."
            elif category == "report.payment":
                draft_text = "Salom! To'lovlar bo'yicha murojaatingiz texnik mutaxassislarga tekshirish uchun topshirildi."
            elif category == "report.account":
                draft_text = "Salom! Akkaunt bo'yicha murojaat ro'yxatga olindi va administratorga yuborildi (1.4-band)."
            elif category == "report.offtopic":
                draft_text = "Salom! Iltimos, reportdan faqat o'yin masalalari bo'yicha foydalaning (3.1-band)."
            else:
                draft_text = "Salom! Murojaatingiz uchun rahmat. Savol navbatchi administratorlarga uzatildi."

    elif reply_lang == "en":
        if kind == "complaint":
            target = f" against player {nickname}" if nickname else ""
            rule_str = f" regarding rule {', '.join(cited_rules)}" if cited_rules else ""
            ev_str = " Evidence has been received." if has_evidence else " Evidence is required for sanctions."
            draft_text = (
                f"Hello! Your complaint{target}{rule_str} has been received for review.{ev_str} "
                "The final decision will be made by a server administrator."
            )
        else:
            if category == "report.help":
                draft_text = "Hello! An administrator is on the way to help you. Please stay at your location."
            elif category == "report.payment":
                draft_text = "Hello! Payment and donation inquiries have been forwarded to technical specialists for verification."
            elif category == "report.account":
                draft_text = "Hello! Your account issue has been registered and forwarded to an administrator for log review (rule 1.4)."
            elif category == "report.offtopic":
                draft_text = "Hello! Please use the report system only for game-related questions (rule 3.1)."
            else:
                draft_text = "Hello! Thank you for your inquiry. An administrator will respond shortly."

    elif reply_lang == "es":
        if kind == "complaint":
            target = f" contra el jugador {nickname}" if nickname else ""
            rule_str = f" según la regla {', '.join(cited_rules)}" if cited_rules else ""
            ev_str = " Pruebas recibidas." if has_evidence else " Se requieren pruebas para aplicar sanciones."
            draft_text = (
                f"¡Hola! Su queja{target}{rule_str} ha sido recibida para su revisión.{ev_str} "
                "La decisión final será tomada por el administrador del servidor."
            )
        else:
            if category == "report.help":
                draft_text = "¡Hola! Un administrador ya va en camino para ayudarte. Por favor, quédate en tu ubicación."
            elif category == "report.payment":
                draft_text = "¡Hola! Los problemas relacionados con pagos y donaciones han sido transferidos para su verificación."
            elif category == "report.account":
                draft_text = "¡Hola! Su solicitud sobre la cuenta ha sido registrada y enviada para la revisión de registros (regla 1.4)."
            elif category == "report.offtopic":
                draft_text = "¡Hola! Por favor, utilice el reporte únicamente para asuntos del juego (regla 3.1)."
            else:
                draft_text = "¡Hola! Gracias por contactarnos. Un administrador responderá pronto."

    elif reply_lang == "pt":
        if kind == "complaint":
            target = f" contra o jogador {nickname}" if nickname else ""
            rule_str = f" referente à regra {', '.join(cited_rules)}" if cited_rules else ""
            ev_str = " Provas recebidas." if has_evidence else " São necessárias provas para aplicar sanções."
            draft_text = (
                f"Olá! Sua denúncia{target}{rule_str} foi recebida para análise.{ev_str} "
                "A decisão final será tomada pelo administrador do servidor."
            )
        else:
            if category == "report.help":
                draft_text = "Olá! Um administrador já está a caminho para ajudar. Por favor, permaneça no local."
            elif category == "report.payment":
                draft_text = "Olá! As questões de pagamento e doação foram encaminhadas para verificação técnica."
            elif category == "report.account":
                draft_text = "Olá! Sua solicitação sobre a conta foi registrada e encaminhada para verificação (regra 1.4)."
            elif category == "report.offtopic":
                draft_text = "Olá! Por favor, utilize o reporte apenas para assuntos do jogo (regra 3.1)."
            else:
                draft_text = "Olá! Obrigado pelo contato. Um administrador responderá em breve."

    else:  # ru
        if kind == "complaint":
            nick_str = f" на игрока {nickname}" if nickname else ""
            rules_str = f" по пункту {', '.join(cited_rules)}" if cited_rules else ""
            ev_str = " Доказательства получены." if has_evidence else " Для наказания требуются доказательства."
            draft_text = (
                f"Здравствуйте! Ваша жалоба{nick_str}{rules_str} принята к рассмотрению.{ev_str} "
                "Решение о наказании будет принято администратором."
            )
        else:
            if category == "report.help":
                draft_text = "Здравствуйте! Администрация уже спешит на помощь. Пожалуйста, оставайтесь на месте."
            elif category == "report.payment":
                draft_text = "Здравствуйте! Вопросы по платежам и донату переданы специалистам для проверки."
            elif category == "report.account":
                draft_text = "Здравствуйте! Обращение по поводу аккаунта зарегистрировано и передано на проверку логов (пункт 1.4 правил)."
            elif category == "report.offtopic":
                draft_text = "Здравствуйте! Пожалуйста, используйте репорт только по игровым вопросам (пункт 3.1 правил)."
            elif category == "report.question":
                draft_text = "Здравствуйте! Для поиска локаций и работы используйте команду /gps. Проверить уровень и баланс можно через /stats, а команды — /help. Приятной игры на Grand Mobile!"
            else:
                draft_text = "Здравствуйте! Спасибо за обращение. Вопрос передан администраторам."

    # Точные цитаты из rules.json без галлюцинаций
    rules_cited = _build_rules_cited(cited_rules)

    # Список ключевых фактов
    facts_extracted = _build_facts_extracted(orig_kind, reply_lang, facts, cited_rules, is_mismatch=False, intent_type="complaint" if kind == "complaint" else "help_stuck")

    return {
        "category": category,
        "lang": reply_lang,
        "priority": priority,
        "is_spam_or_duplicate": category in ["complaint.invalid", "report.offtopic"],
        "facts_extracted": facts_extracted,
        "extracted": {
            "nickname": nickname,
            "rule_refs": cited_rules,
            "has_evidence": has_evidence,
            "server": None
        },
        "draft_reply": {
            "lang": reply_lang,
            "text": draft_text,
            "rules_cited": rules_cited,
            "cited": cited_rules
        },
        "needs_human": needs_human,
        "confidence": 0.82 if client else 0.75,
        "reasons": reasons if reasons else ["Обращение классифицировано по содержанию и ключевым признакам"]
    }


def triage_ticket(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Основная точка входа для триажа одного обращения.
    Принимает payload формата POST /triage:
    { "kind": "complaint"|"report"|"question", "messages": [ ... ] }
    """
    orig_kind_raw = payload.get("kind") or "REPORT"
    orig_kind = "complaint" if "comp" in str(orig_kind_raw).lower() else "report"
    messages = payload.get("messages", [])
    if not messages and payload.get("text"):
        messages = [{"role": "player", "text": str(payload["text"]), "attachments": 0}]

    # 1. Извлечение фактов и RAG
    facts = extract_ticket_facts(messages, kind=orig_kind)
    matched_rules = kb.search_rules(facts["combined_text"], limit=3)

    # 2. Если есть LLM-клиент — пробуем запросить нейросеть
    if client:
        try:
            user_prompt = build_user_prompt(
                kind=orig_kind,
                text=facts["combined_text"],
                attachments_count=facts["attachments_count"],
                matched_rules=matched_rules
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                timeout=4.5  # p95 <= 5s по правилам хакатона
            )

            result_json = json.loads(response.choices[0].message.content)

            # Валидация обязательных полей контракта
            required_keys = ["category", "lang", "facts_extracted", "draft_reply"]
            if all(k in result_json for k in required_keys):
                if result_json.get("category") in kb.needs_human_always:
                    result_json["needs_human"] = True
                
                draft = result_json.get("draft_reply", {})
                if "rules_cited" not in draft or not draft["rules_cited"]:
                    cited_list = draft.get("cited", [])
                    draft["rules_cited"] = _build_rules_cited(cited_list)
                
                if "extracted" not in result_json:
                    result_json["extracted"] = {
                        "nickname": facts["nickname"],
                        "rule_refs": [r["rule_id"] for r in draft.get("rules_cited", [])],
                        "has_evidence": facts["has_evidence"],
                        "server": None
                    }
                if "cited" not in draft:
                    draft["cited"] = [r["rule_id"] for r in draft.get("rules_cited", [])]

                return result_json

        except Exception as e:
            print(f"Ошибка LLM инференса, переключение на fallback: {e}")

    # 3. Экспертный Fallback (работает всегда)
    return _rule_based_triage(orig_kind, facts["combined_text"], facts, matched_rules)
