import os
import json
import re
import time
import requests
from urllib.parse import quote, unquote
from tools.http_request import safe_get
from tools.wiki_api import WIKI_API, HEADERS

session = requests.Session()
session.headers.update(HEADERS)

def get_page_text(title: str):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json"
    }
    r = safe_get(WIKI_API, params=params, headers=HEADERS)
    if r is None:
        return ""

    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("extract", "")


def get_page_images(title: str):
    params = {
        "action": "query",
        "prop": "images",
        "imlimit": "max",
        "titles": title,
        "format": "json"
    }
    r = safe_get(WIKI_API, params=params, headers=HEADERS)
    if r is None:
        return []

    page = next(iter(r.json()["query"]["pages"].values()))
    images = page.get("images", [])
    return [img["title"] for img in images]


def get_image_info(filename: str):
    """通过文件名获取图片真实 URL 和简介"""
    params = {
        "action": "query",
        "titles": filename,
        "prop": "imageinfo",
        "iiprop": "url|comment|extmetadata",
        "format": "json"
    }
    r = safe_get(WIKI_API, params=params, headers=HEADERS)
    if r is None:
        return None

    page = next(iter(r.json()["query"]["pages"].values()))
    if "imageinfo" not in page:
        return None

    info = page["imageinfo"][0]
    return {
        "url": info.get("url"),
        "comment": info.get("comment"),
        "metadata": info.get("extmetadata")
    }


import requests
from io import BytesIO
from PIL import Image
import os

MAX_SIZE_BYTES = 2 * 1024 * 1024  # 2MB

def _try_download(url, save_path, timeout=30):
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                print(f"HTTP {r.status_code} for url: {url}")
                return False

            img_data = BytesIO(r.content)
            if len(img_data.getbuffer()) <= MAX_SIZE_BYTES:
                with open(save_path, "wb") as f:
                    f.write(img_data.getbuffer())
                return True
            img = Image.open(img_data)

            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            max_dim = 2048
            w, h = img.size
            scale = min(1, max_dim / max(w, h))
            if scale < 1:
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            quality = 85
            while quality >= 30:
                temp_bytes = BytesIO()
                img.save(temp_bytes, format="JPEG", quality=quality)
                if temp_bytes.getbuffer().nbytes <= MAX_SIZE_BYTES:
                    with open(save_path, "wb") as f:
                        f.write(temp_bytes.getbuffer())
                    return True
                quality -= 10
            img.save(save_path, format="JPEG", quality=30)
            return True

    except requests.RequestException as e:
        return False
    except Exception as e:
        return False

def download_image_with_fallback(img_url: str, filename: str, save_dir: str, max_retries=3):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return True

    attempts = []
    attempts.append(img_url)

    try:
        parts = img_url.split("/")
        last = parts[-1]
        quoted_last = quote(last, safe='')
        parts[-1] = quoted_last
        quoted_url = "/".join(parts)
        if quoted_url != img_url:
            attempts.append(quoted_url)
    except Exception:
        pass

    if filename.lower().startswith("file:"):
        file_only = filename[len("file:"):]
    else:
        file_only = filename
    special_url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + quote(file_only, safe='')
    attempts.append(special_url)

    for idx, u in enumerate(attempts):
        for trial in range(max_retries):
            wait = (2 ** trial) * 0.5
            if trial > 0:
                time.sleep(wait)
            print(f"try downloading image, attempt {idx + 1}/{len(attempts)}, trial {trial + 1}/{max_retries}: {u}")
            ok = _try_download(u, save_path)
            if ok:
                return True
    return False


def parse_title(title: str, base_path: str):
    title_clean = title.replace("/", "_")
    entry_dir = os.path.join(base_path, title_clean)
    os.makedirs(entry_dir, exist_ok=True)

    print(f"\n=== Processing: {title} ===")

    text = get_page_text(title)
    with open(os.path.join(entry_dir, f"{title_clean}__txt.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    img_files = get_page_images(title)

    # 只保留 jpg/jpeg
    img_files = [
        name for name in img_files
        if name.lower().endswith(".jpg") or name.lower().endswith(".jpeg")
    ]

    image_meta_list = []
    os.makedirs(os.path.join(entry_dir, "images"), exist_ok=True)

    for filename in img_files:
        info = get_image_info(filename)
        if info is None or not info.get("url"):
            continue

        img_url = info["url"]
        save_filename = img_url.split("/")[-1]
        save_filename = unquote(save_filename)
        save_filename = re.sub(r'[<>:"/\\|?*]', '_', save_filename)

        ok = download_image_with_fallback(
            img_url,
            save_filename,
            os.path.join(entry_dir, "images")
        )

        if ok:
            print(f"Downloaded: {save_filename}")

        image_meta_list.append({
            "file": filename,
            "url": img_url,
            "comment": info.get("comment"),
            "metadata": info.get("metadata"),
        })

    with open(os.path.join(entry_dir, "images.json"), "w", encoding="utf-8") as f:
        json.dump(image_meta_list, f, ensure_ascii=False, indent=2)


def parse_txt_file(input_txt: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    with open(input_txt, "r", encoding="utf-8") as f:
        for line in f:
            title = line.strip()
            if not title:
                continue
            parse_title(title, output_path)

if __name__ == "__main__":
    parse_txt_file("entity_data/geography_wiki.txt", "./geography")
