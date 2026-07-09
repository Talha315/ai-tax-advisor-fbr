from pathlib import Path
from urllib.parse import urljoin, urlparse
import json
import re
import time

import requests
from bs4 import BeautifulSoup


START_URL = "https://www.fbr.gov.pk/act-rules-ordinances/131226"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "fbr_documents.json"

# None means download everything found.
# For testing, you can change it to 5.
MAX_DOCUMENTS = None

REQUEST_DELAY = 1

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def clean_filename(title: str) -> str:
    """
    Convert document title into safe filename.
    """
    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", "_", title)
    title = title.strip("_")

    if not title:
        title = "document"

    return title[:120] + ".pdf"


def get_soup(url: str) -> BeautifulSoup:
    """
    Download HTML page and return parsed BeautifulSoup object.
    """
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def normalize_category(text: str):
    """
    Detect category headings from the FBR page.
    """
    text = text.strip().lower()

    if text == "acts":
        return "acts"

    if text == "ordinance" or text == "ordinances":
        return "ordinances"

    if text == "rules":
        return "rules"

    return None


def is_direct_document_url(url: str) -> bool:
    """
    Detect direct PDF/download links.
    FBR often uses download1.fbr.gov.pk for downloadable files.
    """
    lower_url = url.lower()
    parsed = urlparse(lower_url)

    return (
        "download1.fbr.gov.pk" in parsed.netloc
        or lower_url.endswith(".pdf")
    )


def extract_main_page_links():
    """
    Extract links from the main FBR page and assign them to:
    acts, ordinances, or rules.
    """
    soup = get_soup(START_URL)

    links = []
    current_category = None

    for tag in soup.find_all(["h2", "h3", "h4", "a"]):
        text = tag.get_text(" ", strip=True)

        if tag.name in ["h2", "h3", "h4"]:
            detected_category = normalize_category(text)
            if detected_category:
                current_category = detected_category
            continue

        if tag.name == "a" and current_category:
            href = tag.get("href")

            if not href or not text:
                continue

            full_url = urljoin(START_URL, href)

            links.append({
                "category": current_category,
                "title": text,
                "url": full_url,
                "source_page": START_URL
            })

    return links


def extract_document_links_from_detail_page(page_url: str, parent_title: str, category: str):
    """
    Some FBR links are not direct PDFs.
    They open detail pages that contain the real download links.
    """
    documents = []

    try:
        soup = get_soup(page_url)
    except requests.RequestException as error:
        print(f"Could not open detail page: {page_url}")
        print(f"Reason: {error}")
        return documents

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        text = a_tag.get_text(" ", strip=True)

        full_url = urljoin(page_url, href)

        if is_direct_document_url(full_url):
            title = text if text and text.lower() not in ["download", "view", "pdf"] else parent_title

            documents.append({
                "category": category,
                "title": title,
                "url": full_url,
                "source_page": page_url
            })

    return documents


def collect_all_document_links():
    """
    Collect all downloadable document links from:
    1. Main FBR page
    2. Detail pages inside Acts / Ordinance / Rules
    """
    main_links = extract_main_page_links()
    all_documents = []

    print(f"Found {len(main_links)} links on main FBR page.")

    for index, link in enumerate(main_links, start=1):
        category = link["category"]
        title = link["title"]
        url = link["url"]

        print(f"[{index}/{len(main_links)}] Checking: {category} -> {title}")

        if is_direct_document_url(url):
            all_documents.append({
                "category": category,
                "title": title,
                "url": url,
                "source_page": link["source_page"]
            })
        else:
            detail_docs = extract_document_links_from_detail_page(
                page_url=url,
                parent_title=title,
                category=category
            )
            all_documents.extend(detail_docs)

        time.sleep(REQUEST_DELAY)

    # Remove duplicates by URL
    unique_documents = {}
    for document in all_documents:
        unique_documents[document["url"]] = document

    documents = list(unique_documents.values())

    if MAX_DOCUMENTS is not None:
        documents = documents[:MAX_DOCUMENTS]

    return documents


def download_file(url: str, output_path: Path):
    """
    Download one document and save it locally.
    """
    response = requests.get(url, headers=HEADERS, stream=True, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


def main():
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    documents = collect_all_document_links()

    print("\n===================================")
    print(f"Total downloadable documents found: {len(documents)}")
    print("===================================\n")

    metadata = []

    for index, doc in enumerate(documents, start=1):
        category = doc["category"]
        title = doc["title"]
        url = doc["url"]

        category_dir = RAW_PDF_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)

        filename = clean_filename(title)
        output_path = category_dir / filename

        print(f"[{index}/{len(documents)}] Downloading: {category} -> {title}")

        if output_path.exists():
            print(f"Already exists, skipping: {output_path}")
        else:
            try:
                download_file(url, output_path)
                print(f"Saved: {output_path}")
            except requests.RequestException as error:
                print(f"Failed: {url}")
                print(f"Reason: {error}")
                continue

        metadata.append({
            "category": category,
            "title": title,
            "file_url": url,
            "source_page": doc["source_page"],
            "local_path": str(output_path)
        })

        time.sleep(REQUEST_DELAY)

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("\nDownload completed.")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()