"""
Скрипт пакетной обработки тикетов для команды:
make predict IN=tickets.jsonl OUT=pred.jsonl
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Гарантируем UTF-8 в Windows-терминале
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Добавляем корневую директорию проекта в sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from service import triage_ticket
from rag import kb


def process_file(in_path: str, out_path: str, limit: int = None):
    p_in = Path(in_path)
    p_out = Path(out_path)

    if not p_in.exists():
        print(f"Ошибка: входной файл {in_path} не найден!")
        sys.exit(1)

    print(f"Запуск обработки: {in_path} -> {out_path}")
    start_time = time.time()
    
    total = 0
    needs_human_count = 0
    categories_stat = {}
    lang_stat = {}

    with open(p_in, "r", encoding="utf-8") as f_in, open(p_out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            try:
                ticket = json.loads(line)
            except Exception:
                continue

            # Подготовка формата для triage_ticket
            kind = ticket.get("kind", "complaint")
            images_count = ticket.get("images_count", 0)
            messages = ticket.get("messages", [])
            reported_nick = ticket.get("reported_nick")
            rule_refs_raw = ticket.get("rule_refs_raw")

            # Дополняем сообщения количеством вложений, если не указано
            for m in messages:
                if m.get("role") == "player" and "attachments" not in m:
                    m["attachments"] = images_count

            payload = {
                "kind": kind,
                "messages": messages
            }

            # Инференс
            res = triage_ticket(payload)

            # Если в тикете был указан reported_nick, а в тексте не нашли — обогащаем
            if reported_nick and res.get("extracted", {}).get("nickname") is None:
                res["extracted"]["nickname"] = reported_nick
                # обновляем facts_extracted
                res["facts_extracted"] = [
                    f"Указанный нарушитель: {reported_nick}" if "Нарушитель:" in f else f
                    for f in res["facts_extracted"]
                ]

            # Если в тикете был rule_refs_raw, а в правилах пусто — обогащаем
            if rule_refs_raw and not res.get("extracted", {}).get("rule_refs"):
                raw_clean = rule_refs_raw.strip()
                rule_text = kb.get_rule_text(raw_clean)
                if rule_text:
                    res["extracted"]["rule_refs"] = [raw_clean]
                    if not any(rc.get("rule_id") == raw_clean for rc in res.get("draft_reply", {}).get("rules_cited", [])):
                        res["draft_reply"].setdefault("rules_cited", []).append({
                            "rule_id": raw_clean,
                            "quote": rule_text
                        })

            # Сохраняем ticket_id, если он был во входных данных
            if "ticket_id" in ticket:
                res["ticket_id"] = ticket["ticket_id"]

            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")

            total += 1
            if res.get("needs_human"):
                needs_human_count += 1
            cat = res.get("category", "unknown")
            categories_stat[cat] = categories_stat.get(cat, 0) + 1
            lang = res.get("lang", "unknown")
            lang_stat[lang] = lang_stat.get(lang, 0) + 1

            if total % 100 == 0:
                print(f"Обработано {total} тикетов...")

            if limit and total >= limit:
                break

    elapsed = time.time() - start_time
    avg_lat = (elapsed / total * 1000) if total > 0 else 0
    human_pct = (needs_human_count / total * 100) if total > 0 else 0

    print("\n--- ИТОГИ ОБРАБОТКИ ---")
    print(f"Всего тикетов: {total}")
    print(f"Общее время: {elapsed:.2f} сек (в среднем {avg_lat:.1f} мс / тикет)")
    print(f"Доля needs_human: {human_pct:.1f}% ({needs_human_count}/{total})")
    print(f"Языки: {dict(lang_stat)}")
    print(f"Топ-5 категорий: {sorted(categories_stat.items(), key=lambda x: -x[1])[:5]}")
    print(f"Результат сохранён в: {out_path}")


if __name__ == "__main__":
    # Поддерживаем как Make-синтаксис (IN=... OUT=...), так и стандартные флаги (--in ... --out ...)
    in_val = None
    out_val = None
    limit_val = None

    for arg in sys.argv[1:]:
        if arg.startswith("IN=") or arg.startswith("in="):
            in_val = arg.split("=", 1)[1]
        elif arg.startswith("OUT=") or arg.startswith("out="):
            out_val = arg.split("=", 1)[1]
        elif arg.startswith("LIMIT=") or arg.startswith("limit="):
            try:
                limit_val = int(arg.split("=", 1)[1])
            except ValueError:
                pass

    if in_val and out_val:
        process_file(in_val, out_val, limit_val)
    else:
        parser = argparse.ArgumentParser(description="Grand Mobile Support Copilot - Predict")
        parser.add_argument("--in", dest="in_file", required=True, help="Входной jsonl файл")
        parser.add_argument("--out", dest="out_file", required=True, help="Выходной jsonl файл")
        parser.add_argument("--limit", dest="limit", type=int, default=None, help="Ограничить количество тикетов для теста")
        args = parser.parse_args()

        process_file(args.in_file, args.out_file, args.limit)
