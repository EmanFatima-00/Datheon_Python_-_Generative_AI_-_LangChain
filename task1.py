"""
Task 1 - Structured LLM Prompt Pipeline & JSON Extractor
=========================================================
Extracts structured data (invoice or resume) from unstructured text using the
Google Gemini API (free tier), parameterized prompt templates, few-shot
examples, and strict Pydantic validation of the JSON output.

Usage examples (Windows 11, PowerShell):
    python task1.py --type invoice --sample
    python task1.py --type resume --file my_resume.txt --output result.json
    python task1.py                       (interactive: asks type, then paste text)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from string import Template
from typing import List, Optional

try:
    from google import genai
    from google.genai import errors, types
    from pydantic import (BaseModel, ConfigDict, Field, ValidationError,
                          field_validator, model_validator)
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependency: {exc}\nRun: pip install google-genai pydantic")
    sys.exit(1)

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
MAX_ATTEMPTS = 3            # 1 call + up to 2 corrective retries (conserves free quota)
MAX_INPUT_CHARS = 20000     # truncate very long documents to save tokens


# --------------------------------------------------------------------------
# Gemini wrapper: model discovery, rate-limit handling, model fallback
# --------------------------------------------------------------------------
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
        self.models = self._discover_models()
        self.index = 0

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

    def _rank(self, name: str):
        match = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        version = float(match.group(1)) if match else 0.0
        # Prefer Flash-Lite (largest free quota), then stable over preview, then newest.
        return ("lite" in name, not ("preview" in name or "exp" in name), version)

    def _discover_models(self) -> List[str]:
        override = os.environ.get("GEMINI_MODEL", "").strip()
        if override:
            return [override]
        found = []
        try:
            for m in self.client.models.list():
                name = (m.name or "").replace("models/", "", 1)
                if "generateContent" not in (m.supported_actions or []):
                    continue
                if not name.startswith("gemini-") or "flash" not in name:
                    continue
                if any(word in name for word in self.EXCLUDE):
                    continue
                found.append(name)
        except errors.APIError as exc:
            raise GeminiError(self._explain(exc)) from exc
        if not found:
            raise GeminiError("No Gemini Flash model is available for this key. "
                              "Set GEMINI_MODEL to a model id from AI Studio.")
        found.sort(key=self._rank, reverse=True)
        return found[: self.MAX_CANDIDATES]

    def generate(self, contents, config=None):
        """Call generate_content with retries; fall back to the next model if needed."""
        last_exc = None
        while self.index < len(self.models):
            model = self.models[self.index]
            for attempt in range(3):
                try:
                    return self.client.models.generate_content(
                        model=model, contents=contents, config=config)
                except errors.APIError as exc:
                    last_exc = exc
                    code = getattr(exc, "code", None)
                    text = str(exc)
                    if code in (400, 401):
                        raise GeminiError(self._explain(exc)) from exc
                    if code in (429, 500, 503):
                        daily = "PerDay" in text or "per day" in text.lower()
                        if daily or attempt == 2:
                            break  # try the next model (quotas are per model)
                        found = re.search(r"retry in ([\d.]+)s", text)
                        wait = min(float(found.group(1)) + 1 if found else 15, 60)
                        print(f"  [rate limited / busy - waiting {wait:.0f}s]")
                        time.sleep(wait)
                        continue
                    if code in (403, 404):
                        break
                    raise GeminiError(self._explain(exc)) from exc
            self.index += 1
        raise GeminiError(self._explain(last_exc) if last_exc else "No model could be used.")


# --------------------------------------------------------------------------
# Pydantic schemas (strict validation)
# --------------------------------------------------------------------------
def _check_iso_date(value: Optional[str], field: str) -> Optional[str]:
    if value is None:
        return value
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{field} must be a real date in YYYY-MM-DD format, got {value!r}")
    return value


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=1)
    quantity: float = Field(ge=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    invoice_number: str = Field(min_length=1)
    invoice_date: str
    due_date: Optional[str] = None
    vendor_name: str = Field(min_length=1)
    customer_name: str = Field(min_length=1)
    currency: str
    line_items: List[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    discount: float = Field(default=0, ge=0)
    tax: float = Field(default=0, ge=0)
    shipping: float = Field(default=0, ge=0)
    total: float = Field(ge=0)

    @field_validator("invoice_date")
    @classmethod
    def _v_invoice_date(cls, v):
        return _check_iso_date(v, "invoice_date")

    @field_validator("due_date")
    @classmethod
    def _v_due_date(cls, v):
        return _check_iso_date(v, "due_date")

    @field_validator("currency")
    @classmethod
    def _v_currency(cls, v):
        if not re.fullmatch(r"[A-Z]{3}", v):
            raise ValueError("currency must be a 3-letter ISO code such as USD")
        return v

    @model_validator(mode="after")
    def _v_totals(self):
        for item in self.line_items:
            if abs(item.quantity * item.unit_price - item.amount) > 0.02:
                raise ValueError(f"line item '{item.description}': quantity x unit_price != amount")
        if abs(sum(i.amount for i in self.line_items) - self.subtotal) > 0.02:
            raise ValueError("subtotal does not equal the sum of line item amounts")
        expected = self.subtotal - self.discount + self.tax + self.shipping
        if abs(expected - self.total) > 0.02:
            raise ValueError("total must equal subtotal - discount + tax + shipping")
        return self


_PERIOD = r"\d{4}(-\d{2})?|Present"


class Education(BaseModel):
    model_config = ConfigDict(extra="forbid")
    institution: str = Field(min_length=1)
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_year: Optional[int] = Field(default=None, ge=1950, le=2100)
    end_year: Optional[int] = Field(default=None, ge=1950, le=2100)

    @model_validator(mode="after")
    def _v_years(self):
        if self.start_year and self.end_year and self.end_year < self.start_year:
            raise ValueError("end_year cannot be before start_year")
        return self


class Experience(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    summary: Optional[str] = None

    @field_validator("start_date", "end_date")
    @classmethod
    def _v_period(cls, v):
        if v is not None and not re.fullmatch(_PERIOD, v):
            raise ValueError("date must be YYYY-MM, YYYY or 'Present'")
        return v


class Resume(BaseModel):
    model_config = ConfigDict(extra="forbid")
    full_name: str = Field(min_length=1)
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)

    @field_validator("email")
    @classmethod
    def _v_email(cls, v):
        if v is not None and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("email is not a valid address")
        return v


MODELS = {"invoice": Invoice, "resume": Resume}

# --------------------------------------------------------------------------
# Parameterized prompt template + few-shot examples
# --------------------------------------------------------------------------
PROMPT_TEMPLATE = Template("""You are a precise information-extraction engine.
Task: extract $doc_type data from the document below.

Rules:
- Output ONLY one JSON object that matches the schema. No markdown, no commentary.
- Use null (or [] for lists) when a value is missing; never invent data.
- Numbers must be plain numbers (no currency symbols or thousands separators).
$extra_rules

JSON schema:
$schema

Examples:
$examples

Now extract from this document:
<document>
$document
</document>
$feedback""")

EXTRA_RULES = {
    "invoice": "- Dates must be YYYY-MM-DD. currency is a 3-letter ISO code. "
               "total = subtotal - discount + tax + shipping.",
    "resume": "- Job dates are YYYY-MM, YYYY or \"Present\". Years in education are integers.",
}

FEW_SHOT = {
    "invoice": [(
        "INVOICE #A-17\nDate: March 3, 2025   Due: April 2, 2025\nFrom: Bright Pens Ltd\n"
        "Bill to: Sam Lee\n2 x Notebook @ $10.00 = $20.00\nTax: $2.00\nTotal: $22.00 USD",
        {"invoice_number": "A-17", "invoice_date": "2025-03-03", "due_date": "2025-04-02",
         "vendor_name": "Bright Pens Ltd", "customer_name": "Sam Lee", "currency": "USD",
         "line_items": [{"description": "Notebook", "quantity": 2, "unit_price": 10.0, "amount": 20.0}],
         "subtotal": 20.0, "discount": 0, "tax": 2.0, "shipping": 0, "total": 22.0})],
    "resume": [(
        "Ana Cruz - ana@mail.com - Lisbon\nSkills: Python, SQL\n"
        "Data Analyst at Acme (Jan 2022 - Present): built dashboards.\n"
        "BSc Statistics, Univ. of Porto, 2017-2021",
        {"full_name": "Ana Cruz", "email": "ana@mail.com", "phone": None, "location": "Lisbon",
         "summary": None, "skills": ["Python", "SQL"],
         "education": [{"institution": "Univ. of Porto", "degree": "BSc", "field_of_study": "Statistics",
                        "start_year": 2017, "end_year": 2021}],
         "experience": [{"company": "Acme", "title": "Data Analyst", "start_date": "2022-01",
                         "end_date": "Present", "summary": "Built dashboards."}],
         "certifications": [], "languages": []})],
}

SAMPLES = {
    "invoice": """INVOICE No. INV-2025-0042
Issued: 12 January 2025        Payment due: 11 February 2025
Vendor: NorthWind Supplies, 14 Harbor Rd, Boston
Billed to: Priya Nair Consulting

Description                 Qty   Unit price   Amount
Standing desk                2     $350.00     $700.00
Ergonomic chair              1     $220.00     $220.00
Cable organiser kit          4      $12.50      $50.00

Subtotal: $970.00
Discount: $70.00
Tax (8%): $72.00
Shipping: $30.00
TOTAL DUE: $1,002.00 USD""",
    "resume": """Jordan Miles
jordan.miles@example.com | +1 555 0100 | Austin, TX
Backend developer with 4 years of experience building APIs.

EXPERIENCE
Software Engineer, Orbit Labs (June 2021 - Present)
 - Built REST services in Python and FastAPI.
Junior Developer, PixelForge (2019-08 to 2021-05)
 - Maintained internal tools.

EDUCATION
B.S. Computer Science, University of Texas, 2015 - 2019

SKILLS: Python, SQL, Docker, Git
CERTIFICATIONS: AWS Cloud Practitioner
LANGUAGES: English, Spanish""",
}


def build_prompt(doc_type: str, document: str, feedback: str = "") -> str:
    """Fill the parameterized template for the chosen document type."""
    model_cls = MODELS[doc_type]
    examples = []
    for text, data in FEW_SHOT[doc_type]:
        examples.append(f"<document>\n{text}\n</document>\nJSON:\n{json.dumps(data)}")
    return PROMPT_TEMPLATE.substitute(
        doc_type=doc_type,
        extra_rules=EXTRA_RULES[doc_type],
        schema=json.dumps(model_cls.model_json_schema()),
        examples="\n\n".join(examples),
        document=document,
        feedback=feedback,
    )


# --------------------------------------------------------------------------
# Extraction with validation and corrective retry
# --------------------------------------------------------------------------
class ExtractionError(Exception):
    pass


def clean_json_text(raw: str) -> str:
    """Remove accidental markdown fences around the JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    return re.sub(r"\s*```$", "", raw).strip()


def summarize_validation(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, e['loc'])) or 'root'}: {e['msg']}" for e in exc.errors())


def extract(gemini: GeminiClient, doc_type: str, document: str) -> BaseModel:
    config = types.GenerateContentConfig(response_mime_type="application/json")
    feedback = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = build_prompt(doc_type, document, feedback)
        print(f"Calling Gemini model '{gemini.model}' (attempt {attempt}/{MAX_ATTEMPTS}) ...")
        response = gemini.generate(prompt, config)
        raw = response.text
        if not raw:
            feedback = "\nYour previous reply was empty. Return the JSON object only.\n"
            continue
        try:
            return MODELS[doc_type].model_validate(json.loads(clean_json_text(raw)))
        except json.JSONDecodeError as exc:
            problem = f"not valid JSON ({exc})"
        except ValidationError as exc:
            problem = summarize_validation(exc)
        print(f"  Output rejected: {problem}")
        feedback = (f"\nYour previous output was rejected: {problem}\n"
                    "Return corrected JSON only.\n")
    raise ExtractionError(f"Could not get valid JSON after {MAX_ATTEMPTS} attempts.")


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------
def load_document(args, doc_type: str) -> str:
    if args.sample:
        text = SAMPLES[doc_type]
    elif args.file:
        if not os.path.isfile(args.file):
            raise SystemExit(f"File not found: {args.file}")
        try:
            with open(args.file, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            raise SystemExit(f"Cannot read file: {exc}")
    else:
        print("Paste the document text, then press Enter, Ctrl+Z, Enter (Windows) to finish:")
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        raise SystemExit("The document is empty.")
    if len(text) > MAX_INPUT_CHARS:
        print(f"Warning: document truncated to {MAX_INPUT_CHARS} characters.")
        text = text[:MAX_INPUT_CHARS]
    return text


def ask_type() -> str:
    while True:
        choice = input("Document type (invoice/resume): ").strip().lower()
        if choice in MODELS:
            return choice
        print("Please type 'invoice' or 'resume'.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Extract validated JSON from text using Gemini.")
    parser.add_argument("--type", choices=list(MODELS), help="document type")
    parser.add_argument("--file", help="path to a UTF-8 text file")
    parser.add_argument("--sample", action="store_true", help="use the built-in sample document")
    parser.add_argument("--output", help="output JSON path (default: <type>_output.json)")
    args = parser.parse_args()

    try:
        doc_type = args.type or ask_type()
        document = load_document(args, doc_type)
        gemini = GeminiClient()
        result = extract(gemini, doc_type, document)
    except (GeminiError, ExtractionError) as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)

    output_json = result.model_dump_json(indent=2)
    out_path = args.output or f"{doc_type}_output.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(output_json)
    print("\nValidated JSON:\n" + output_json)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
    