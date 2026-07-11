from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from retriever import load_collection, load_embedding_model
from rag_chain import (
    get_groq_client,
    answer_question_stream,
    trim_messages,
    MAX_HISTORY_TOKENS,
)


st.set_page_config(
    page_title="FBR Tax Advisor",
    page_icon="📄",
    layout="wide"
)


@st.cache_resource
def load_rag_resources():
    collection = load_collection()
    embedding_model = load_embedding_model()
    groq_client = get_groq_client()

    return collection, embedding_model, groq_client


def reset_chat():
    st.session_state.messages = []
    st.session_state.chat_history = []


def initialize_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def show_sidebar():
    with st.sidebar:
        st.title("FBR Tax Advisor")

        st.markdown(
            """
            This assistant answers tax questions using the indexed FBR documents only.
            
            It should:
            - answer in simple English
            - cite FBR sources
            - refuse when the answer is not in the corpus
            """
        )

        st.divider()

        st.write("**Current settings**")
        st.write(f"Chat history limit: {MAX_HISTORY_TOKENS} tokens")
        st.write("Retriever: ChromaDB")
        st.write("LLM: Groq")

        st.divider()

        if st.button("Clear chat"):
            reset_chat()
            st.rerun()


def show_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main():
    initialize_session()
    show_sidebar()

    st.title("📄 FBR Tax Advisor Assistant")
    st.caption("Ask Pakistani tax questions using the indexed FBR corpus.")

    try:
        collection, embedding_model, groq_client = load_rag_resources()
    except Exception as error:
        st.error("The RAG system could not be loaded.")
        st.exception(error)
        return

    show_chat_history()

    user_question = st.chat_input("Ask a tax question...")

    if user_question:
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            full_answer = ""

            try:
                for piece in answer_question_stream(
                    question=user_question,
                    collection=collection,
                    embedding_model=embedding_model,
                    groq_client=groq_client,
                    chat_history=st.session_state.chat_history,
                ):
                    full_answer += piece
                    answer_placeholder.markdown(full_answer + "▌")

                answer_placeholder.markdown(full_answer)

            except Exception as error:
                full_answer = (
                    "Something went wrong while generating the answer. "
                    "Please check the terminal logs or API configuration."
                )
                answer_placeholder.error(full_answer)
                st.exception(error)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer
        })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_answer
        })

        st.session_state.chat_history = trim_messages(
            st.session_state.chat_history,
            MAX_HISTORY_TOKENS
        )


if __name__ == "__main__":
    main()