"""
Task 4 - Interactive Full-Stack AI Chatbot with Streamlit
==========================================================
A browser chat app that wraps a RAG pipeline (Task 2 logic, self-contained
here) in a Streamlit UI with streaming token output, chat history and clear
session-state management. Without uploaded documents it works as a normal
chatbot with conversation memory; once PDF/text files are indexed it answers
from them and shows source citations.

Run (Windows 11, PowerShell):
    streamlit run task4.py
(running "python task4.py" also works - it launches Streamlit for you)
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Iterator, List

try:
    import chromadb
    import streamlit as st
    from google import genai
    from google.genai import errors, types
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependency: {exc}\nRun: pip install streamlit google-genai chromadb "
          "langchain-community langchain-text-splitters pypdf")
    sys.exit(1)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
EMBED_BATCH = 50
HISTORY_TURNS = 8   # last N user/assistant exchanges sent to the model
NOT_FOUND_MESSAGE = "I could not find the answer in the uploaded documents."

CHAT_INSTRUCTION = "You are a helpful, concise assistant. Use the conversation history for context."
RAG_INSTRUCTION = f"""You answer using ONLY the numbered context passages in the latest user message.
- If they do not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}"
- Never guess or use outside knowledge. Use the conversation history only to understand follow-up questions.
- Be concise and cite sources in brackets, e.g. [report.pdf, page 3]."""


class GeminiError(Exception):
    """Any problem talking to the Gemini API, with a user-friendly message."""


class GeminiClient:
    MAX_CANDIDATES = 4
    EXCLUDE = ("image", "tts", "audio", "live", "embedding", "robotics",
               "computer-use", "native", "vision", "imagen", "veo")

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise GeminiError(
                "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey, "
                'run `setx GEMINI_API_KEY "your-key-here"` in PowerShell, then close ALL terminals, '
                "reopen one and start the app again.")
        self.client = genai.Client(api_key=api_key)
        self.models, self.embed_model = self._discover_models()
        self.index = 0
        self.use_task_type = True

    @property
    def model(self) -> str:
        return self.models[min(self.index, len(self.models) - 1)]

    @staticmethod
    def _explain(exc: Exception) -> str:
        code = getattr(exc, "code", None)
        if code in (400, 401, 403) and "API key" in str(exc):
            return "The Gemini API key is invalid. Check GEMINI_API_KEY."
        if code == 429:
            return "Free-tier quota or rate limit reached. Wait a while and retry."
        if code == 404:
            return "The selected Gemini model is not available."
        if code in (500, 503):
            return "Gemini service is temporarily unavailable. Try again later."
        return f"Gemini API error: {exc}"

    @staticmethod
    def _rank(name: str):
        match = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        version = float(match.group(1)) if match else 0.0
        return ("lite" in name, not ("preview" in name or "exp" in name), version)

    def _discover_models(self):
        gen_override = os.environ.get("GEMINI_MODEL", "").strip()
        emb_override = os.environ.get("GEMINI_EMBED_MODEL", "").strip()
        gen_models, emb_models = [], []
        try:
            for m in self.client.models.list():
                name = (m.name or "").replace("models/", "", 1)
                actions = m.supported_actions or []
                if "embedding" in name and any(a in actions for a in ("embedContent", "batchEmbedContents")):
                    emb_models.append(name)
                elif ("generateContent" in actions and name.startswith("gemini-") and "flash" in name
                      and not any(w in name for w in self.EXCLUDE)):
                    gen_models.append(name)
        except errors.APIError as exc:
            raise GeminiError(self._explain(exc)) from exc
        gen_models.sort(key=self._rank, reverse=True)
        gen_models = [gen_override] if gen_override else gen_models[: self.MAX_CANDIDATES]
        emb_models.sort(key=lambda n: ("gemini-embedding" in n, n), reverse=True)
        embed_model = emb_override or (emb_models[0] if emb_models else "")
        if not gen_models:
            raise GeminiError("No Gemini Flash model available. Set GEMINI_MODEL manually.")
        if not embed_model:
            raise GeminiError("No embedding model available. Set GEMINI_EMBED_MODEL manually.")
        return gen_models, embed_model

    @staticmethod
    def _transient(exc, attempt) -> float:
        """Seconds to wait if the error is a short-lived rate limit/outage, else 0."""
        code = getattr(exc, "code", None)
        text = str(exc)
        if code not in (429, 500, 503) or attempt >= 2 or "PerDay" in text:
            return 0
        found = re.search(r"retry in ([\d.]+)s", text)
        return min(float(found.group(1)) + 1 if found else 10, 30)

    def stream(self, contents, system_instruction: str) -> Iterator[str]:
        """Yield text chunks. Falls back to another model if none has been produced yet."""
        config = types.GenerateContentConfig(system_instruction=system_instruction)
        last_exc = None
        while self.index < len(self.models):
            for attempt in range(3):
                produced = False
                try:
                    for chunk in self.client.models.generate_content_stream(
                            model=self.models[self.index], contents=contents, config=config):
                        if chunk.text:
                            produced = True
                            yield chunk.text
                    return
                except errors.APIError as exc:
                    last_exc = exc
                    if produced:
                        raise GeminiError(self._explain(exc)) from exc
                    if getattr(exc, "code", None) in (400, 401):
                        raise GeminiError(self._explain(exc)) from exc
                    wait = self._transient(exc, attempt)
                    if wait:
                        time.sleep(wait)
                        continue
                    break
            self.index += 1
        raise GeminiError(self._explain(last_exc) if last_exc else "No model could be used.")

    def embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start:start + EMBED_BATCH]
            for attempt in range(3):
                try:
                    config = types.EmbedContentConfig(task_type=task_type) if self.use_task_type else None
                    resp = self.client.models.embed_content(
                        model=self.embed_model, contents=batch, config=config)
                    vectors.extend(list(e.values) for e in resp.embeddings)
                    break
                except errors.APIError as exc:
                    if getattr(exc, "code", None) == 400 and self.use_task_type \
                            and "API key" not in str(exc):
                        self.use_task_type = False
                        continue
                    wait = self._transient(exc, attempt)
                    if wait:
                        time.sleep(wait)
                        continue
                    raise GeminiError(self._explain(exc)) from exc
            else:
                raise GeminiError("Embedding failed after several attempts.")
        return vectors


# --------------------------------------------------------------------------
# Cached resources and session state
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Gemini ...")
def get_gemini() -> GeminiClient:
    return GeminiClient()


@st.cache_resource
def get_chroma():
    return chromadb.EphemeralClient()  # in-memory; documents vanish when the server stops


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])          # [{"role", "content", "sources"}]
    ss.setdefault("session_id", uuid.uuid4().hex[:12])
    ss.setdefault("indexed_hashes", set())
    ss.setdefault("doc_names", [])


def get_collection():
    return get_chroma().get_or_create_collection(
        f"s_{st.session_state.session_id}", metadata={"hnsw:space": "cosine"})


def reset_documents() -> None:
    try:
        get_chroma().delete_collection(f"s_{st.session_state.session_id}")
    except Exception:
        pass
    st.session_state.indexed_hashes = set()
    st.session_state.doc_names = []


def index_uploads(gemini: GeminiClient, uploads) -> None:
    ss = st.session_state
    collection = get_collection()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for up in uploads:
        data = up.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        if digest in ss.indexed_hashes:
            continue  # already indexed - saves free-tier embedding calls
        suffix = os.path.splitext(up.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            if suffix == ".pdf":
                pages = PyPDFLoader(tmp_path).load()
            else:
                try:
                    pages = TextLoader(tmp_path, encoding="utf-8").load()
                except UnicodeDecodeError:
                    pages = TextLoader(tmp_path, encoding="latin-1").load()
        except Exception as exc:
            st.sidebar.error(f"Could not read {up.name}: {exc}")
            continue
        finally:
            os.unlink(tmp_path)
        for p in pages:
            p.metadata["source"] = up.name
        chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
        if not chunks:
            st.sidebar.warning(f"No extractable text in {up.name} (scanned PDF?).")
            continue
        all_texts = [c.page_content for c in chunks]
        all_embeds = [gemini.embed([t], "RETRIEVAL_DOCUMENT")[0] for t in all_texts]
        collection.add(
            ids=[f"{digest[:16]}-{i}" for i in range(len(chunks))],
            embeddings=all_embeds,
            documents=all_texts,
            metadatas=[{"source": up.name, "page": int(c.metadata["page"]) + 1 if "page" in c.metadata else 0} for c in chunks]
        )
        ss.indexed_hashes.add(digest)
        ss.doc_names.append(up.name)


def retrieve(gemini: GeminiClient, question: str):
    """Return (context_text, source_labels) for the question, or ('', []) if no documents."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return "", []
    qvec = gemini.embed([question], "RETRIEVAL_QUERY")[0]
    res = collection.query(query_embeddings=[qvec], n_results=min(TOP_K, total),
                           include=["documents", "metadatas"])
    passages, sources = [], []
    for n, (text, meta) in enumerate(zip(res["documents"][0], res["metadatas"][0]), start=1):
        label = f"{meta['source']}, page {meta['page']}" if meta["page"] else meta["source"]
        passages.append(f"[{n}] ({label})\n{text}")
        if label not in sources:
            sources.append(label)
    return "\n\n".join(passages), sources


def build_contents(history: list, augmented_last: str) -> list:
    """Convert stored messages to Gemini contents (recent turns only); last user turn is augmented."""
    recent = history[-(HISTORY_TURNS * 2):]
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]  # conversation must start with a user turn
    contents = []
    for i, msg in enumerate(recent):
        role = "user" if msg["role"] == "user" else "model"
        text = augmented_last if i == len(recent) - 1 else msg["content"]
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
    return contents


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def render_sidebar(gemini: GeminiClient) -> None:
    ss = st.session_state
    with st.sidebar:
        st.header("Settings")
        st.caption(f"Model: {gemini.model}")
        st.subheader("Documents (optional)")
        uploads = st.file_uploader("Upload PDF or text files", type=["pdf", "txt", "md"],
                                   accept_multiple_files=True)
        if st.button("Index uploaded documents", disabled=not uploads):
            try:
                with st.spinner("Reading and embedding documents ..."):
                    index_uploads(gemini, uploads)
                st.success("Documents ready.")
            except GeminiError as exc:
                st.error(str(exc))
        if ss.doc_names:
            st.write("Indexed: " + ", ".join(ss.doc_names))
            st.info("Answers now come only from these documents.")
            if st.button("Remove documents"):
                reset_documents()
                st.rerun()
        st.divider()
        if st.button("Clear chat history"):
            ss.messages = []
            st.rerun()


def main_app() -> None:
    st.set_page_config(page_title="Gemini Chatbot", page_icon=":speech_balloon:")
    st.title("Gemini AI Chatbot")
    init_state()
    try:
        gemini = get_gemini()
    except GeminiError as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar(gemini)
    ss = st.session_state

    if not ss.messages:
        st.caption("Ask me anything. Upload documents in the sidebar to chat with them.")
    for msg in ss.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.write(f"- {s}")

    prompt = st.chat_input("Type your message ...")
    if not prompt:
        return

    ss.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            context, sources = retrieve(gemini, prompt)
            if context:
                augmented = f"Context passages:\n\n{context}\n\nQuestion: {prompt}"
                instruction = RAG_INSTRUCTION
            else:
                augmented, instruction = prompt, CHAT_INSTRUCTION
            contents = build_contents(ss.messages, augmented)
            answer = st.write_stream(gemini.stream(contents, instruction))
            if not answer:
                answer = "The model returned an empty reply. Please try again."
                st.markdown(answer)
            shown_sources = [] if (not context or NOT_FOUND_MESSAGE in answer) else sources
            if shown_sources:
                with st.expander("Sources"):
                    for s in shown_sources:
                        st.write(f"- {s}")
            ss.messages.append({"role": "assistant", "content": answer, "sources": shown_sources})
        except GeminiError as exc:
            st.error(str(exc))
            ss.messages.pop()  # remove the unanswered user message so history stays consistent


def main() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        running_in_streamlit = get_script_run_ctx() is not None
    except Exception:
        running_in_streamlit = False
    if running_in_streamlit:
        main_app()
    else:
        print("Starting Streamlit ... (press Ctrl+C to stop)")
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])


if __name__ == "__main__":
    main()