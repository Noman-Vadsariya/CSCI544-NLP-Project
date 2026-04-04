#!/usr/bin/env python3

import json
import random
import re
import time
from typing import Any, Dict, List

import requests
from datasets import load_dataset

SEARCH_API = "https://en.wikipedia.org/w/rest.php/v1/search/title"
SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"

session = requests.Session()
session.headers.update({
    "User-Agent": "ASQA-RAG-Project/1.0 (contact: your_email@example.com)",
    "Accept": "application/json"
})


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()


def polite_sleep(base: float = 0.8, jitter: float = 0.7) -> None:
    time.sleep(base + random.uniform(0, jitter))


def request_json(url: str, params: Dict[str, Any] = None, retries: int = 4, timeout: int = 15) -> Dict[str, Any]:
    params = params or {}
    last_err = None

    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)

            if resp.status_code in (429, 500, 502, 503, 504):
                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

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


def get_titles_from_wikipages(ex: Dict[str, Any]) -> List[str]:
    titles = []
    for page in ex.get("wikipages", []):
        if isinstance(page, dict):
            title = clean_text(page.get("title", ""))
            if title:
                titles.append(title)
    return list(dict.fromkeys(titles))


def get_titles_from_annotations(ex: Dict[str, Any]) -> List[str]:
    titles = []
    annotations = ex.get("annotations", [])
    if isinstance(annotations, list):
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            for kb in ann.get("knowledge", []):
                if isinstance(kb, dict):
                    title = clean_text(kb.get("title", ""))
                    if title:
                        titles.append(title)
    return list(dict.fromkeys(titles))


def get_annotation_context(ex: Dict[str, Any]) -> str:
    passages = []
    seen = set()

    annotations = ex.get("annotations", [])
    if isinstance(annotations, list):
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            for kb in ann.get("knowledge", []):
                if isinstance(kb, dict):
                    content = clean_text(kb.get("content", ""))
                    if content and content not in seen:
                        seen.add(content)
                        passages.append(content)

    return "\n\n".join(passages)


def wikipedia_search(query: str, limit: int = 5) -> List[str]:
    if not query:
        return []

    params = {
        "q": query,
        "limit": limit
    }

    try:
        data = request_json(SEARCH_API, params=params)
        titles = []
        for item in data.get("pages", []):
            title = clean_text(item.get("title", ""))
            if title:
                titles.append(title)
        return list(dict.fromkeys(titles))
    except Exception as err:
        print(f"Wikipedia search failed for query '{query}': {err}")
        return []


def fetch_wikipedia_summary(title: str, max_chars: int = 1500) -> str:
    url = SUMMARY_API.format(requests.utils.quote(title.replace(" ", "_"), safe=""))

    try:
        data = request_json(url)
        text = clean_text(data.get("extract", ""))
        print(text[:100] + "..." if text else "[No summary]")
        if text:
            print(f"[SUCCESS] {title} | chars: {len(text)}")
        else:
            print(f"[EMPTY] {title}")

        return text[:max_chars]

    except Exception as err:
        print(f"[ERROR] {title} -> {err}")
        return ""


def build_context(ex: Dict[str, Any], max_pages: int = 5) -> str:
    question = get_question(ex)

    titles = []
    titles.extend(get_titles_from_wikipages(ex))
    titles.extend(get_titles_from_annotations(ex))

    # Always make a Wikipedia query from the question
    titles.extend(wikipedia_search(question, limit=max_pages))
    titles = list(dict.fromkeys([t for t in titles if t]))

    passages = []
    seen = set()

    for title in titles[:max_pages]:
        print(f"Fetching Wikipedia page: {title}")
        page_text = fetch_wikipedia_summary(title)
        if page_text and page_text not in seen:
            seen.add(page_text)
            passages.append(f"{title}. {page_text}")

        polite_sleep(base=1.0, jitter=1.0)

    # Keep annotation knowledge as extra evidence, but do not rely on it alone
    ann_context = get_annotation_context(ex)
    if ann_context:
        passages.append(ann_context)

    return "\n\n".join(passages)


def convert_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    question = get_question(ex)
    answer = get_answer(ex)
    context = build_context(ex)

    return {
        "context": context,
        "prompts": [question],
        "responses": [answer]
    }


def main():
    print("Downloading ASQA from Hugging Face...")
    ds = load_dataset("din0s/asqa", split="train")

    out_path = "asqa_final.jsonl"
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