"""
Task 2 - Document Q&A Assistant with RAG & Vector Embeddings
=============================================================
Loads PDF/text documents, splits them recursively, creates Gemini embeddings,
stores them in a persistent ChromaDB collection, retrieves relevant chunks and
answers questions strictly from the documents, citing source file and page.

Usage (Windows 11, PowerShell):
    python task2.py --docs report.pdf notes.txt        (index, then ask questions)
    python task2.py --docs docs_folder --question "What is the refund policy?"
    python task2.py                                    (ask questions on the existing index)
    python task2.py --reset                            (delete the stored index)
"""

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import List

try:
    import chromadb
    from google import genai
    from google.genai import errors, types
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependency: {exc}\nRun: pip install google-genai chromadb "
          "langchain-community langchain-text-splitters pypdf")
    sys.exit(1)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
EMBED_BATCH = 50
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
NOT_FOUND_MESSAGE = "I could not find the answer in the provided documents."

SYSTEM_INSTRUCTION = f"""You answer questions using ONLY the numbered context passages provided.
- If the passages do not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}"
- Never use outside knowledge and never guess.
- Be concise. Mention the supporting source and page in brackets, e.g. [report.pdf, page 3]."""


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
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and run in PowerShell:\n"
                '  setx GEMINI_API_KEY "your-key-here"\n'
                "then close and reopen the terminal.")
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
        """Find usable generation models and an embedding model (never hardcoded blindly)."""
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
        # Prefer the gemini-embedding family, newest name last alphabetically.
        emb_models.sort(key=lambda n: ("gemini-embedding" in n, n), reverse=True)
        embed_model = emb_override or (emb_models[0] if emb_models else "")
        if not gen_models:
            raise GeminiError("No Gemini Flash model available. Set GEMINI_MODEL manually.")
        if not embed_model:
            raise GeminiError("No embedding model available. Set GEMINI_EMBED_MODEL manually.")
        return gen_models, embed_model

    def _retry_wait(self, exc, attempt) -> bool:
        """Sleep and return True if the error is a transient rate limit / outage."""
        code = getattr(exc, "code", None)
        text = str(exc)
        if code not in (429, 500, 503) or attempt >= 2 or "PerDay" in text:
            return False
        found = re.search(r"retry in ([\d.]+)s", text)
        wait = min(float(found.group(1)) + 1 if found else 15, 60)
        print(f"  [rate limited / busy - waiting {wait:.0f}s]")
        time.sleep(wait)
        return True

    def generate(self, contents, config=None):
        last_exc = None
        while self.index < len(self.models):
            for attempt in range(3):
                try:
                    return self.client.models.generate_content(
                        model=self.models[self.index], contents=contents, config=config)
                except errors.APIError as exc:
                    last_exc = exc
                    code = getattr(exc, "code", None)
                    if code in (400, 401):
                        raise GeminiError(self._explain(exc)) from exc
                    if self._retry_wait(exc, attempt):
                        continue
                    break
            self.index += 1
        raise GeminiError(self._explain(last_exc) if last_exc else "No model could be used.")

    def embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Embed texts in batches. task_type: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
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
                        self.use_task_type = False  # model may not support task_type
                        continue
                    if self._retry_wait(exc, attempt):
                        continue
                    raise GeminiError(self._explain(exc)) from exc
            else:
                raise GeminiError("Embedding failed after several attempts.")
        return vectors


# --------------------------------------------------------------------------
# Loading, chunking, indexing
# --------------------------------------------------------------------------
def collect_files(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS))
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        else:
            print(f"Skipping (not found or unsupported type): {p}")
    return files


def load_file(path: Path):
    """Return LangChain Documents for a PDF or text file (one per PDF page)."""
    if path.suffix.lower() == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    else:
        try:
            docs = TextLoader(str(path), encoding="utf-8").load()
        except UnicodeDecodeError:
            docs = TextLoader(str(path), encoding="latin-1").load()
    for d in docs:
        d.metadata["source"] = path.name
    return docs


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_files(gemini: GeminiClient, collection, files: List[Path]) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for path in files:
        digest = file_hash(path)
        if collection.get(where={"file_hash": digest}, limit=1)["ids"]:
            print(f"Already indexed (skipping, no API calls used): {path.name}")
            continue
        try:
            pages = load_file(path)
        except Exception as exc:  # corrupt/encrypted PDF etc.
            print(f"Could not read {path.name}: {exc}")
            continue
        chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
        if not chunks:
            print(f"No extractable text in {path.name} (scanned PDF?). Skipping.")
            continue
        print(f"Embedding {len(chunks)} chunks from {path.name} ...")
        vectors = gemini.embed([c.page_content for c in chunks], "RETRIEVAL_DOCUMENT")
        collection.add(
            ids=[f"{digest[:16]}-{i}" for i in range(len(chunks))],
            embeddings=vectors,
            documents=[c.page_content for c in chunks],
            metadatas=[{
                "source": path.name,
                "page": int(c.metadata.get("page", -1)) + 1 if "page" in c.metadata else 0,
                "file_hash": digest,
            } for c in chunks],
        )


# --------------------------------------------------------------------------
# Retrieval + answering
# --------------------------------------------------------------------------
def answer_question(gemini: GeminiClient, collection, question: str):
    """Return (answer_text, list_of_source_strings)."""
    total = collection.count()
    if total == 0:
        return "The index is empty. Add documents with --docs.", []
    query_vec = gemini.embed([question], "RETRIEVAL_QUERY")[0]
    result = collection.query(query_embeddings=[query_vec], n_results=min(TOP_K, total),
                              include=["documents", "metadatas"])
    passages, sources = [], []
    for n, (text, meta) in enumerate(zip(result["documents"][0], result["metadatas"][0]), start=1):
        where = f"{meta['source']}, page {meta['page']}" if meta["page"] else meta["source"]
        passages.append(f"[{n}] ({where})\n{text}")
        if where not in sources:
            sources.append(where)
    prompt = "Context passages:\n\n" + "\n\n".join(passages) + f"\n\nQuestion: {question}"
    config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
    response = gemini.generate(prompt, config)
    answer = (response.text or "").strip() or NOT_FOUND_MESSAGE
    return answer, ([] if answer == NOT_FOUND_MESSAGE else sources)


def print_answer(answer: str, sources: List[str]) -> None:
    print(f"\nAnswer: {answer}")
    if sources:
        print("Sources: " + "; ".join(sources))
    print()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="RAG Q&A over PDF/text documents using Gemini.")
    parser.add_argument("--docs", nargs="+", help="files and/or folders (.pdf, .txt, .md)")
    parser.add_argument("--db", default="chroma_db", help="folder for the vector store")
    parser.add_argument("--question", help="ask one question and exit")
    parser.add_argument("--reset", action="store_true", help="delete the stored index and exit")
    args = parser.parse_args()

    try:
        gemini = GeminiClient()
        store = chromadb.PersistentClient(path=args.db)
        name = "rag_" + re.sub(r"[^a-zA-Z0-9_]", "_", gemini.embed_model)[:55]
        if args.reset:
            try:
                store.delete_collection(name)
            except Exception:
                pass
            print("Index deleted.")
            return
        collection = store.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

        if args.docs:
            files = collect_files(args.docs)
            if not files:
                print("No supported documents found.")
                sys.exit(1)
            index_files(gemini, collection, files)
        if collection.count() == 0:
            print("No documents indexed yet. Run: python task2.py --docs <file-or-folder>")
            sys.exit(1)

        print(f"Using model '{gemini.model}' and embeddings '{gemini.embed_model}'. "
              f"{collection.count()} chunks indexed.")
        if args.question:
            print_answer(*answer_question(gemini, collection, args.question))
            return
        print("Ask questions about your documents. Type 'exit' to quit.")
        while True:
            question = input("Question> ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if question:
                print_answer(*answer_question(gemini, collection, question))
    except GeminiError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()