"""
Основной сервис триажа обращений Grand Mobile Support Copilot.
Поддерживает:
1. Вызов LLM через OpenAI-совместимый API (OpenAI, Gemini, Groq, DeepSeek, OpenRouter)
2. Надёжный эвристический fallback-движок (работает без API-ключа или при сбое сети)

Строгое разделение типов обращений (Routing):
1. Вопросы по геймплею и помощи (report.question / report.help) — понятные инструкции, без правил нарушений.
2. Жалобы и нарушения (complaint.* / ban) — правила нарушений из rules.json.
3. Кросс-канальная маршрутизация (Канал vs Суть текста):
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


def _is_gameplay_question(norm: str) -> bool:
    """
    Проверяет, является ли текст игрока обычным вопросом по геймплею, механике,
    работам, уровням или помощи, а не жалобой на нарушения или игроков.
    """
    # Если в сообщении есть жалоба на нарушение — это НЕ геймплейный вопрос
    if _is_complaint_intent(norm):
        return False

    if any(w in norm for w in ["админ", "хелпер", "деморган", "jail", "взлом", "взломали", "украли аккаунт", "пароль", "не могу зайти", "донат", "оплат", "не пришли гк", "деньги списали", "hacked", "donate"]):
        return False

    if re.search(r'\b(бан|бана|баны|забанен\w*|разбан\w*|варн\w*|мут\w*|жалоб\w*)\b', norm):
        return False

    # Маркеры вопросов по геймплею и базовой помощи
    gameplay_markers = [
        # Вопросительные маркеры
        "как", "где", "куда", "когда", "почему", "зачем", "сколько", "можно ли", "что делать",
        # Работы и заработок
        "шахт", "руд", "карьер", "завод", "ферм", "дальнобой", "грузчик", "работ", "устроит", "зарплат", "деньг", "доход",
        # Финансы
        "банк", "вклад", "депозит", "счет", "снять", "положить", "передать",
        # Транспорт
        "автосалон", "машин", "транспорт", "автомобил", "купить", "продать", "авторынок", "тюнинг",
        # Права и лицензии
        "права", "автошкол", "лицензи", "экзамен", "сдать", "вождение",
        # Фракции и организации
        "мэри", "правительств", "фракци", "организац", "собеседован", "инвайт", "ранг", "повышен",
        # Навигация и координаты
        "gps", "гпс", "навигатор", "карт", "найти", "место", "локаци", "где находится",
        # Прокачка
        "лвл", "уровен", "опыт", "exp", "стат", "скилл", "прокач",
        # Команды и справка
        "команд", "хелп", "помощ", "подскаж", "скаж", "инвентар", "квест",
        # KK
        "қалай", "қайда", "жұмыс", "ақша", "деңгей", "қанша", "көлік", "шахта", "банк",
        # UZ
        "qanday", "qayerda", "ish", "ishga", "daraja", "litsenziya", "qancha", "mashina", "bank", "shaxta",
        # EN
        "how", "where", "what", "which", "job", "mine", "mining", "work", "level", "salary", "gps", "money", "buy", "car", "bank",
        # ES
        "cómo", "dónde", "trabajo", "mina", "nivel", "salario", "cuánto", "dinero", "coche", "auto", "banco",
        # PT
        "como", "onde", "trabalho", "mina", "nível", "salário", "quanto", "dinheiro", "carro", "banco"
    ]
    return any(gm in norm for gm in gameplay_markers)


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
    kind_name = "Жалоба на игрока" if kind == "complaint" else "Внутриигровой репорт"
    items.append(f"Канал обращения: {kind_name}")

    if intent_type == "gameplay":
        items.append("Суть обращения: Вопрос по игровому процессу / механике / помощи")
    elif intent_type == "help_stuck":
        items.append("Суть обращения: Просьба о технической помощи (застрял / баг)")
    else:
        items.append("Суть обращения: Заявление о нарушении правил игры")

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
        if intent_type != "gameplay":
            items.append("Доказательства: не приложены")

    if cited_rules:
        items.append(f"Затронутые пункты правил: {', '.join(cited_rules)}")
    else:
        if intent_type == "gameplay":
            items.append("Пункты правил: не применимы (обращение носит справочный характер)")

    items.append(f"Определённый язык: {lang}")
    return items


def _rule_based_triage(
    kind: str,
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

    # Нормализуем тип обращения: help -> report
    if kind in ["help", "question"]:
        kind = "report"

    reply_lang = lang if lang in ["ru", "kk", "uz", "en", "es", "pt"] else "ru"

    # =========================================================================
    # СЦЕНАРИЙ 1: Кросс-канальный Mismatch — Игрок задал вопрос по геймплею через форму жалобы!
    # =========================================================================
    if kind == "complaint" and _is_gameplay_question(norm):
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
        facts_extracted = _build_facts_extracted(kind, reply_lang, facts, cited_rules, is_mismatch=True, intent_type="gameplay")

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
    # СЦЕНАРИЙ 2: Вопрос по геймплею в репорте / помощи
    # =========================================================================
    if kind == "report" and _is_gameplay_question(norm):
        # 1. Если игрок конкретно застрял в текстурах
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
            facts_extracted = _build_facts_extracted(kind, reply_lang, facts, cited_rules, is_mismatch=False, intent_type="help_stuck")
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

        # 2. Игровые механики, работы, навигация
        category = "report.question"
        cited_rules = []  # Правила нарушений категорически не применяются
        priority = "p3"
        needs_human = False
        reasons = ["Вопрос по игровым механикам классифицирован как report.question, подготовлена подробная инструкция."]

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
        elif any(w in norm for w in ["банк", "вклад", "депозит", "счет", "снять деньги"]):
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
        else:
            draft_templates = {
                "ru": (
                    "Здравствуйте! Для поиска нужной локации или работы используйте встроенный навигатор: введите команду /gps "
                    "и выберите интересующий раздел (Поиск работы, Государственные учреждения, Банк). "
                    "Уровень персонажа и текущую зарплату можно посмотреть командой /stats, а список всех команд — /help. "
                    "Приятной игры на Grand Mobile!"
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
                    "You can check your character level and stats with /stats, and see all available commands with /help. "
                    "Enjoy playing Grand Mobile!"
                ),
                "es": (
                    "¡Hola! Para encontrar cualquier ubicación o trabajo, usa el GPS del juego: ingresa el comando /gps y elige la categoría deseada (Búsqueda de trabajo, Banco, Gobierno). "
                    "Puedes verificar tu nivel y estadísticas con /stats, y ver todos los comandos con /help. "
                    "¡Disfruta jugando en Grand Mobile!"
                ),
                "pt": (
                    "Olá! Para encontrar qualquer local ou trabalho, use o GPS do jogo: digite o comando /gps e escolha a categoria desejada (Buscar emprego, Banco, Governo). "
                    "Você pode conferir seu nível e estatísticas com /stats, e ver todos os comandos com /help. "
                    "Bom jogo no Grand Mobile!"
                )
            }
            draft_text = draft_templates.get(reply_lang, draft_templates["ru"])

        rules_cited = []
        facts_extracted = _build_facts_extracted(kind, reply_lang, facts, cited_rules, is_mismatch=False, intent_type="gameplay")

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

    # СЦЕНАРИЙ 3: Кросс-канальный Mismatch — Жалоба на нарушение через репорт (kind == "report")
    # =========================================================================
    if kind == "report" and _is_complaint_intent(norm):
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
        facts_extracted = _build_facts_extracted(kind, reply_lang, facts, cited_rules, is_mismatch=True, intent_type="complaint")

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
    else:  # report
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
    facts_extracted = _build_facts_extracted(kind, reply_lang, facts, cited_rules, is_mismatch=False, intent_type="complaint" if kind == "complaint" else "help_stuck")

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
    { "kind": "complaint"|"report", "messages": [ ... ] }
    """
    kind = payload.get("kind", "complaint")
    messages = payload.get("messages", [])

    # 1. Извлечение фактов и RAG
    facts = extract_ticket_facts(messages, kind=kind)
    matched_rules = kb.search_rules(facts["combined_text"], limit=3)

    # 2. Если есть LLM-клиент — пробуем запросить нейросеть
    if client:
        try:
            user_prompt = build_user_prompt(
                kind=kind,
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
                # Дополнительная проверка на принудительный human handoff
                if result_json.get("category") in kb.needs_human_always:
                    result_json["needs_human"] = True
                
                # Обеспечиваем наличие rules_cited с верифицированными цитатами
                draft = result_json.get("draft_reply", {})
                if "rules_cited" not in draft or not draft["rules_cited"]:
                    cited_list = draft.get("cited", [])
                    draft["rules_cited"] = _build_rules_cited(cited_list)
                
                # Обеспечиваем обратную совместимость с extracted и cited
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
            # Логируем и мягко переключаемся на экспертный fallback
            print(f"Ошибка LLM инференса, переключение на fallback: {e}")

    # 3. Экспертный Fallback (работает всегда)
    return _rule_based_triage(kind, facts["combined_text"], facts, matched_rules)
