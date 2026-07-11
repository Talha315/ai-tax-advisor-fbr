from pathlib import Path
import os
import re

from dotenv import load_dotenv
from groq import Groq

from retriever import load_collection, load_embedding_model, search_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

TOP_K = 6
MAX_CONTEXT_CHARS = 12000
MAX_HISTORY_TOKENS = 1000

# Chroma distance: smaller means better.
MAX_ALLOWED_DISTANCE = 0.95


SYSTEM_PROMPT = """
You are an AI Tax Advisor Assistant for Pakistani business owners.

Follow these rules strictly:

1. Answer only using the provided FBR context.
2. Do not use outside knowledge.
3. If the answer is not clearly available in the FBR context, say:
   "I don't know based on the available FBR documents."
4. Use simple plain English.
5. Do not give personal legal or tax advice.
6. Every important claim must include a citation.
7. Citation format must be:
   Section [X], [Act/Rules/Ordinance Name], [Year]
8. Do not invent section numbers, years, or Act names.
9. If the exact section is not visible in the context, say:
   "The exact section is not clearly visible in the retrieved text."
10. If the retrieved documents are from different years or versions, mention that.
11. Keep the answer short, clear, and practical.
"""


def check_environment():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")


def get_groq_client():
    check_environment()
    return Groq(api_key=GROQ_API_KEY)


def estimate_tokens(text):
    """
    Rough token estimate.
    Good enough for trimming chat history.
    """
    if not text:
        return 0

    return max(1, len(text) // 4)


def trim_messages(messages, max_tokens=MAX_HISTORY_TOKENS):
    """
    Keep chat history under max_tokens.

    Strategy:
    - Preserve latest message.
    - Remove oldest messages first.
    - System prompt is not stored here, so it is always preserved.
    """
    if not messages:
        return []

    latest_message = messages[-1]
    older_messages = messages[:-1]

    kept_messages = [latest_message]
    token_count = estimate_tokens(latest_message.get("content", ""))

    for message in reversed(older_messages):
        message_tokens = estimate_tokens(message.get("content", ""))

        if token_count + message_tokens > max_tokens:
            continue

        kept_messages.insert(0, message)
        token_count += message_tokens

    return kept_messages


def tokenize(text):
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    stop_words = {
        "the", "is", "are", "am", "a", "an", "of", "to", "in", "on",
        "for", "and", "or", "by", "with", "from", "under", "what",
        "who", "when", "where", "how", "does", "do", "shall", "be",
        "this", "that", "it", "tell", "explain"
    }

    return {
        word for word in words
        if word not in stop_words and len(word) > 2
    }


def keyword_overlap(question, context):
    question_words = tokenize(question)
    context_words = tokenize(context)

    if not question_words:
        return 0.0

    matched_words = question_words.intersection(context_words)
    return len(matched_words) / len(question_words)


def infer_document_name_and_year(source_file):
    file_name = source_file.lower()

    known_documents = [
        ("income_tax_ordinance", "Income Tax Ordinance", "2001"),
        ("income_tax_rules", "Income Tax Rules", "2002"),
        ("sales_tax_act", "Sales Tax Act", "1990"),
        ("sales_tax_rules", "Sales Tax Rules", "2006"),
        ("customs_act", "Customs Act", "1969"),
        ("customs_rules", "Customs Rules", "year not clear"),
        ("federal_excise_act", "Federal Excise Act", "2005"),
        ("federal_excise_rules", "Federal Excise Rules", "2005"),
        (
            "islamabad_capital_territory_tax_on_services",
            "Islamabad Capital Territory Tax on Services Ordinance",
            "2001",
        ),
    ]

    for key, document_name, year in known_documents:
        if key in file_name:
            return document_name, year

    readable_name = source_file.replace(".txt", "").replace("_", " ").title()

    year_match = re.search(r"(19|20)\d{2}", source_file)
    year = year_match.group(0) if year_match else "year not clear"

    return readable_name, year


def extract_section_reference(text):
    """
    Extract a visible section/rule reference.
    We do not invent section numbers.
    """
    patterns = [
        r"\b[Ss]ection\s+([0-9]+[A-Z]?)\b",
        r"\b[Ss]ec\.?\s+([0-9]+[A-Z]?)\b",
        r"\b[Rr]ule\s+([0-9]+[A-Z]?)\b",
        r"\b([0-9]{1,3}[A-Z]?)\.\s+[A-Z][A-Za-z ,\-\(\)]{5,120}",
        r"\b([0-9]{1,3}[A-Z]?)\s*\[[^\]]+\]",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if not match:
            continue

        found_text = match.group(0).strip()

        if found_text.lower().startswith("section"):
            return found_text

        if found_text.lower().startswith("sec"):
            return f"Section {match.group(1)}"

        if found_text.lower().startswith("rule"):
            return found_text

        return f"Section {match.group(1)}"

    return "Section not clearly visible"


def make_source_block(source_number, chunk):
    metadata = chunk["metadata"]
    text = chunk["text"]

    source_file = metadata.get("source_file", "unknown_file")
    category = metadata.get("category", "unknown")
    language = metadata.get("language", "unknown")
    page_start = metadata.get("page_start", "")
    page_end = metadata.get("page_end", "")
    chunk_index = metadata.get("chunk_index", "")

    document_name, year = infer_document_name_and_year(source_file)
    section_reference = extract_section_reference(text)

    citation = f"{section_reference}, {document_name}, {year}"

    return f"""
[Source {source_number}]
Citation: {citation}
Source file: {source_file}
Document name: {document_name}
Year: {year}
Category: {category}
Language: {language}
Pages: {page_start}-{page_end}
Chunk index: {chunk_index}
Vector distance: {chunk.get("distance")}

Text:
{text}
""".strip()


def format_context(chunks):
    blocks = []
    total_chars = 0

    for source_number, chunk in enumerate(chunks, start=1):
        block = make_source_block(source_number, chunk)

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n" + ("-" * 80 + "\n\n").join(blocks)


def context_is_strong_enough(question, chunks, context):
    """
    ChromaDB always returns something.
    This blocks clearly unrelated questions before calling Groq.
    """
    if not chunks:
        return False

    best_distance = chunks[0].get("distance", 999)

    if best_distance > MAX_ALLOWED_DISTANCE:
        return False

    overlap = keyword_overlap(question, context)

    if overlap < 0.08:
        return False

    return True


def build_user_prompt(question, context, chat_history):
    trimmed_history = trim_messages(chat_history, MAX_HISTORY_TOKENS)

    history_text = ""

    for message in trimmed_history:
        role = message.get("role", "user")
        content = message.get("content", "")

        history_text += f"{role.upper()}:\n{content}\n\n"

    return f"""
Chat history:
{history_text}

FBR context:
{context}

User question:
{question}

Answer the user's question using only the FBR context.

Required format:

Answer:
[plain English answer]

Citations:
- Section [X], [Act/Rules/Ordinance Name], [Year]

Important:
- Use only the citations provided in the sources.
- Do not invent section numbers.
- Do not answer from general knowledge.
- If the answer is not available in the FBR context, say:
  "I don't know based on the available FBR documents."
"""


def call_groq_stream(client, messages):
    """
    Stream response from Groq.
    """
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1200,
        stream=True,
    )

    for chunk in response:
        piece = chunk.choices[0].delta.content

        if piece:
            yield piece


def answer_question_stream(question, collection, embedding_model, groq_client, chat_history):
    chunks = search_documents(
        question=question,
        collection=collection,
        model=embedding_model,
        top_k=TOP_K,
    )

    context = format_context(chunks)

    if not context_is_strong_enough(question, chunks, context):
        yield "I don't know based on the available FBR documents."
        return

    user_prompt = build_user_prompt(
        question=question,
        context=context,
        chat_history=chat_history,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    for piece in call_groq_stream(groq_client, messages):
        yield piece


def main():
    print("Loading ChromaDB collection...")
    collection = load_collection()

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print("Starting Groq client...")
    groq_client = get_groq_client()

    chat_history = []

    print("\nRAG Tax Advisor is ready.")
    print("Type your question. Type 'exit' to stop.\n")

    while True:
        question = input("Ask a tax question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        if not question:
            continue

        chat_history.append({
            "role": "user",
            "content": question,
        })

        print("\n" + "=" * 90)
        print("ANSWER")
        print("=" * 90)

        answer_parts = []

        try:
            for piece in answer_question_stream(
                question=question,
                collection=collection,
                embedding_model=embedding_model,
                groq_client=groq_client,
                chat_history=chat_history,
            ):
                print(piece, end="", flush=True)
                answer_parts.append(piece)

            final_answer = "".join(answer_parts)

            chat_history.append({
                "role": "assistant",
                "content": final_answer,
            })

            chat_history = trim_messages(chat_history, MAX_HISTORY_TOKENS)

        except Exception as error:
            print("\n\nSomething went wrong.")
            print(f"Reason: {error}")

        print("\n" + "=" * 90 + "\n")


if __name__ == "__main__":
    main()