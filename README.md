# Gemini LLM Python Internship Tasks

Four self-contained Python projects built with the free Google Gemini API (`google-genai` SDK). Each task is a single `.py` file in the repo root.

| Task | Project | File |
|------|---------|------|
| 1 | Structured LLM Prompt Pipeline & JSON Extractor | `task1.py` |
| 2 | Document Q&A Assistant with RAG & Vector Embeddings | `task2.py` |
| 3 | Multi-Tool Autonomous AI Agent | `task3.py` |
| 4 | Interactive Full-Stack AI Chatbot with Streamlit | `task4.py` |

## Setup (Windows 11)

1. Install Python 3.9 or newer.
2. Get a free Gemini API key at https://aistudio.google.com/apikey
3. Set it as an environment variable in PowerShell:
   ```powershell
   setx GEMINI_API_KEY "your-key-here"
   ```
4. Close and reopen the terminal.

The key is read only from `GEMINI_API_KEY` and is never stored in the code.

Optional overrides: `GEMINI_MODEL` (generation model) and `GEMINI_EMBED_MODEL` (embedding model, Tasks 2 and 4). By default each script discovers an available Gemini Flash model automatically and falls back to another if one hits its quota.

---

## Task 1: Structured LLM Prompt Pipeline & JSON Extractor

Extracts structured data from unstructured text (invoices or resumes) using parameterized prompt templates, few-shot examples, and strict Pydantic validation. Invalid output triggers up to two corrective retries.

**Install**
```powershell
pip install google-genai pydantic
```

**Run**
```powershell
python task1.py --type invoice --sample
python task1.py --type resume --file my_resume.txt --output result.json
```

**Test:** run both built-in samples, then try an empty file and an unset API key to see the error handling.

### Screenshots of Task 1

<img width="451" height="186" alt="WhatsApp Image 2026-10-26 at 5 04 32 AM" src="https://github.com/user-attachments/assets/85d9f943-b178-46b7-bc6f-e02b59c6627e" />
<img width="437" height="193" alt="WhatsApp Image 2026-10-26 at 5 04 33 AM" src="https://github.com/user-attachments/assets/53d5a4e1-950e-45b0-ac0a-541d9e72ebd1" />
<img width="471" height="191" alt="WhatsApp Image 2026-10-26 at 5 04 33 AM (1)" src="https://github.com/user-attachments/assets/783979c8-207e-434d-88ae-b0f688cb74b9" />


---

## Task 2: Document Q&A Assistant with RAG & Vector Embeddings

Loads PDF/text files, splits them recursively, creates Gemini embeddings, stores them in ChromaDB, and answers questions only from the documents, with source file and page citations. Already-indexed files are skipped to save free-tier quota.

**Install**
```powershell
pip install google-genai chromadb langchain-community langchain-text-splitters pypdf
```

**Run**
```powershell
python task2.py --docs report.pdf notes.txt
python task2.py --docs docs_folder --question "What is the refund policy?"
python task2.py --reset
```

**Test:** ask a question answered in your PDF and check the page citation, then ask something unrelated and confirm it says the answer was not found.

### Screenshots of Task 2

<img width="986" height="157" alt="WhatsApp Image 2026-10-26 at 5 17 28 AM" src="https://github.com/user-attachments/assets/1fb26f58-610e-4e91-9851-196f9a0311dc" />
<img width="492" height="124" alt="WhatsApp Image 2026-10-26 at 5 17 28 AM (1)" src="https://github.com/user-attachments/assets/d62fe7ec-f112-4e46-bf3c-83687aaea433" />
<img width="450" height="136" alt="WhatsApp Image 2026-10-26 at 5 17 29 AM" src="https://github.com/user-attachments/assets/81fe0b75-8204-48db-a3bf-2911408a9307" />


---

## Task 3: Multi-Tool Autonomous AI Agent

A console agent using Gemini function calling with three tools: DuckDuckGo web search, a safe math calculator, and a local file reader (restricted to the current folder). It prints its reasoning chain: Thought, Action, Observation, Final Answer.

**Install**
```powershell
pip install google-genai ddgs
```

**Run**
```powershell
python task3.py
python task3.py --query "What is 15% of the population of France?"
```

**Test:** ask a math question, a current-events question, and "Summarize notes.txt" (with a `notes.txt` in the folder).

### Screenshots of Task 3

<img width="988" height="188" alt="WhatsApp Image 2026-10-26 at 5 35 05 AM" src="https://github.com/user-attachments/assets/a2dfedca-e754-473a-9308-3f2278bec077" />
<img width="707" height="179" alt="WhatsApp Image 2026-10-26 at 5 35 05 AM (1)" src="https://github.com/user-attachments/assets/5010c916-3e05-4eca-99c8-467cb225c2d6" />
<img width="984" height="201" alt="WhatsApp Image 2026-10-26 at 5 35 06 AM" src="https://github.com/user-attachments/assets/54f4e55c-8a5a-44be-af35-1586e66917dc" />


---

## Task 4: Interactive Full-Stack AI Chatbot with Streamlit

A browser chatbot with streaming responses, conversation memory, and session state management. Upload PDF/text files in the sidebar to chat with them (RAG with source citations), or use it as a normal chatbot.

**Install**
```powershell
pip install streamlit google-genai chromadb langchain-community langchain-text-splitters pypdf
```

**Run**
```powershell
streamlit run task4.py
```

**Test:** chat and ask a follow-up to check memory, upload and index a PDF, ask about it, then clear the chat.

### Screenshots of Task 4

<img width="1354" height="644" alt="WhatsApp Image 2026-10-26 at 6 02 46 AM" src="https://github.com/user-attachments/assets/50c3da62-6379-4e6a-ae07-671fb4ff543f" />
<img width="1348" height="621" alt="WhatsApp Image 2026-10-26 at 6 02 46 AM (1)" src="https://github.com/user-attachments/assets/70a4b078-310b-4913-b7c8-8d71f6a2a595" />
<img width="287" height="508" alt="WhatsApp Image 2026-10-26 at 6 02 47 AM (1)" src="https://github.com/user-attachments/assets/c083f36e-9ff9-4ab3-ac0b-9308a82d042f" />
<img width="826" height="577" alt="WhatsApp Image 2026-10-26 at 6 02 47 AM" src="https://github.com/user-attachments/assets/1c7aba6d-69ab-433c-a213-0cd8305bfb4d" />


---

## Notes

- The Gemini free tier has rate and daily limits, and free-tier content may be used by Google to improve its products. Do not upload private documents.
- Never commit your API key. Suggested `.gitignore`:
  ```
  chroma_db/
  __pycache__/
  ```
