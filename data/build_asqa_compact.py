#!/usr/bin/env python3

import json
import random
import re
import time
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup
from datasets import load_dataset

session = requests.Session()
session.headers.update({
    "User-Agent": "ASQA-RAG-Project/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\[(?:\d+|edit|citation needed|source)\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def polite_sleep(base: float = 0.8, jitter: float = 0.7) -> None:
    time.sleep(base + random.uniform(0, jitter))


def request_text(url: str, retries: int = 4, timeout: int = 20) -> str:
    last_err = None

    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout, allow_redirects=True)

            if resp.status_code in (429, 500, 502, 503, 504):
                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.text

        except Exception as err:
            last_err = err
            wait = (2 ** attempt) + random.uniform(0.5, 1.5)
            time.sleep(wait)

    raise last_err


def get_question(ex: Dict[str, Any]) -> str:
    return clean_text(ex.get("ambiguous_question", ""))


def get_answer(ex: Dict[str, Any]) -> str:
    annotations = ex.get("annotations", [])
    if isinstance(annotations, list) and annotations:
        first = annotations[0]
        if isinstance(first, dict):
            return clean_text(first.get("long_answer", ""))
    return ""


def get_wikipages_from_json(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = ex.get("wikipages", [])
    if isinstance(pages, list):
        return [p for p in pages if isinstance(p, dict)]
    return []


def get_wikipage_title(page: Dict[str, Any]) -> str:
    return clean_text(page.get("title", ""))


def get_wikipage_url(page: Dict[str, Any]) -> str:
    if not isinstance(page, dict):
        return ""

    for key in ("wikipedia_link", "url", "link", "page_url", "source_url"):
        url = clean_text(page.get(key, ""))
        if url:
            return url

    title = get_wikipage_title(page)
    if title:
        return "https://en.wikipedia.org/wiki/" + requests.utils.quote(title.replace(" ", "_"), safe="")

    return ""


def html_to_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    content = soup.select_one("#mw-content-text") or soup.select_one("main") or soup.body or soup

    for selector in [
        ".reference",
        ".mw-editsection",
        ".reflist",
        ".hatnote",
        ".mw-jump-link",
        ".catlinks",
        ".navbox",
        ".vertical-navbox",
        ".metadata",
        ".mw-empty-elt",
    ]:
        for tag in content.select(selector):
            tag.decompose()

    blocks = []
    for el in content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "figcaption"]):
        txt = clean_text(el.get_text(" ", strip=True))
        if txt:
            blocks.append(txt)

    cleaned = []
    prev = None
    for block in blocks:
        if block != prev:
            cleaned.append(block)
        prev = block

    return "\n\n".join(cleaned).strip()


def fetch_wikipedia_page(url: str, max_chars: int = 25000) -> str:
    if not url:
        return ""

    try:
        html = request_text(url)
        text = html_to_clean_text(html)

        if text:
            print(f"[SUCCESS] fetched page | chars: {len(text)}")
        else:
            print("[EMPTY] page text")

        return text[:max_chars]

    except Exception as err:
        print(f"[ERROR] fetching page '{url}' -> {err}")
        return ""


def build_context(ex: Dict[str, Any], max_pages: int = 5) -> str:
    pages = get_wikipages_from_json(ex)

    passages = []
    seen = set()

    for page in pages[:max_pages]:
        url = get_wikipage_url(page)
        title = get_wikipage_title(page)

        if not url:
            continue

        label = title or url
        print(f"Fetching Wikipedia page: {label}")

        page_text = fetch_wikipedia_page(url)
        if page_text and page_text not in seen:
            seen.add(page_text)
            if title:
                passages.append(f"{title}\n{page_text}")
            else:
                passages.append(page_text)

        polite_sleep(base=1.0, jitter=1.0)

    return "\n\n".join(passages)


def convert_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    question = get_question(ex)
    answer = get_answer(ex)
    context = build_context(ex)

    return {
        "context": context,
        "prompts": [question],
        "responses": [answer],  # only long_answer
    }


def main():
    print("Downloading ASQA from Hugging Face...")
    ds = load_dataset("din0s/asqa", split="dev")

    out_path = "asqa_test.jsonl"
    kept = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            question = get_question(ex)
            answer = get_answer(ex)

            if not question or not answer:
                skipped += 1
                continue

            sample = convert_example(ex)

            if not sample["context"]:
                skipped += 1
                continue

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            kept += 1

            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1} examples")

            polite_sleep(base=0.5, jitter=0.8)

    print(f"Saved {kept} samples to {out_path}")
    print(f"Skipped {skipped} samples")


if __name__ == "__main__":
    main()