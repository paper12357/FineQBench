import os
import re
import json

data_dirs = ["football_players", "football_clubs"]

data = {}

output_json_path = "dblotus\\lotus_football.json"

def extract_name_from_filename(filename: str) -> str:
    start = filename.find("_")
    end = filename.find("__")
    if start == -1 or end == -1 or end <= start:
        name = os.path.splitext(filename)[0]
        name = re.sub(r'^\d+_?', '', name)
        return name.replace("_", " ").strip()
    name = filename[start + 1:end]
    return name.replace("_", " ").strip()

def extract_sections_by_major_heading(text: str):
    pattern = re.compile(r'^(={2}(?!\=))\s*(.+?)\s*(\1)(?!\=)\s*$', flags=re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        content = text.strip()
        if len(content) > 30:
            return [("General", content, False)]
        else:
            return []

    sections = []
    first = matches[0]
    if first.start() > 0:
        preface = text[: first.start()].strip()
        if len(preface) > 30:
            sections.append(("Introduction", preface, True)) 

    for i, m in enumerate(matches):
        heading_raw = m.group(2).strip()
        start_content = m.end()
        end_content = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start_content:end_content].strip()
        if len(content) < 30:
            continue
        sections.append((heading_raw, content, False))

    return sections

for data_dir in data_dirs:
    if not os.path.exists(data_dir):
        continue

    lower = os.path.basename(data_dir).lower()
    is_player_dir = "player" in lower
    is_club_dir = "club" in lower

    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(".txt"):
            continue

        filepath = os.path.join(data_dir, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            continue

        name = extract_name_from_filename(filename)
        sections = extract_sections_by_major_heading(text)

        for heading, content, is_preface in sections:
            if is_preface:
                if is_player_dir and not is_club_dir:
                    table_name = "Player Introduction"
                elif is_club_dir and not is_player_dir:
                    table_name = "Club Introduction"
                else:
                    table_name = f"{os.path.basename(data_dir)} Introduction"

                prefix = f"== {name}'s Introduction ==\n"
                paragraph_text = prefix + content
            else:
                if is_player_dir and not is_club_dir:
                    table_name = f"Player {heading}"
                elif is_club_dir and not is_player_dir:
                    table_name = f"Club {heading}"
                else:
                    table_name = f"{os.path.basename(data_dir)} {heading}"

                prefix = f"== {name}'s {heading} ==\n"
                paragraph_text = prefix + content

            data.setdefault(table_name, []).append(paragraph_text)

total_tables = len(data)
total_paragraphs = sum(len(v) for v in data.values())
print(f"Total tables: {total_tables}, total paragraphs: {total_paragraphs}")

data = {k: v for k, v in data.items() if len(v) >= 10}

try:
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=2)
    print(f"save to path {output_json_path}")
except Exception as e:
    print(f"fail to save to path {output_json_path} : {e}")
