from pathlib import Path
import json
import re

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PDF_DIR = PROJECT_ROOT / "data" / "raw_pdfs"
TEXT_OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted_text"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "extraction_report.json"


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_pdf_category(pdf_path):
    """
    Example:
    data/raw_pdfs/acts/file.pdf -> acts
    data/raw_pdfs/rules/file.pdf -> rules
    """
    relative_path = pdf_path.relative_to(RAW_PDF_DIR)
    return relative_path.parts[0]


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))

    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text is None:
            text = ""

        text = clean_text(text)

        pages_text.append(
            f"\n\n--- Page {page_number} ---\n\n{text}"
        )

    full_text = "\n".join(pages_text).strip()

    return full_text, len(reader.pages)


def save_text_file(pdf_path, text):
    category = get_pdf_category(pdf_path)

    output_folder = TEXT_OUTPUT_DIR / category
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = output_folder / f"{pdf_path.stem}.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)

    return output_path


def main():
    TEXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_PDF_DIR.rglob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    report = []

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n[{index}/{len(pdf_files)}] Processing: {pdf_path.name}")

        try:
            text, page_count = extract_text_from_pdf(pdf_path)
            output_path = save_text_file(pdf_path, text)

            character_count = len(text)
            possible_scanned_pdf = character_count < 500

            report.append({
                "category": get_pdf_category(pdf_path),
                "pdf_path": str(pdf_path),
                "text_path": str(output_path),
                "pages": page_count,
                "character_count": character_count,
                "possible_scanned_pdf": possible_scanned_pdf
            })

            print(f"Saved: {output_path}")
            print(f"Pages: {page_count}")
            print(f"Characters: {character_count}")

            if possible_scanned_pdf:
                print("Warning: Very little text extracted. This may be scanned.")

        except Exception as error:
            print(f"Failed: {pdf_path.name}")
            print(f"Reason: {error}")

            report.append({
                "category": get_pdf_category(pdf_path),
                "pdf_path": str(pdf_path),
                "text_path": None,
                "pages": 0,
                "character_count": 0,
                "possible_scanned_pdf": True,
                "error": str(error)
            })

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nText extraction completed.")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()