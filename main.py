import os
import json
import base64
import logging
from io import BytesIO

import pytesseract
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from pypdf import PdfReader
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_bytes

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Analyzer API",
    description="Extracts and structures text from PDF and DOCX files using OpenAI.",
    version="1.0.0",
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "gpt-4o-mini"
PDF_MAGIC_BYTES = b"%PDF"
DOCX_MAGIC_BYTES = b"PK"

SUPPORTED_FILE_TYPES = {"pdf", "docx"}

CV_JSON_TEMPLATE = {
    "NAME": "",
    "CITY_STATE": "",
    "LAST_POSITION": "",
    "SUMMARY": "",
    "EDUCATION": [{"DEGREE": "", "COURSE": "", "INSTITUTION": "", "YEAR": ""}],
    "CERTIFICATIONS": [{"CERTIFICATION": "", "INSTITUTION": "", "YEAR": ""}],
    "LANGUAGES": [{"LANGUAGE": "", "LEVEL": ""}],
    "EXPERIENCES": [{"POSITION": "", "COMPANY": "", "PERIOD": "", "DESCRIPTION": ""}],
    "ACTIVITIES": [{"ACTIVITY": ""}],
    "TECHNOLOGIES": [{"TECHNOLOGY": ""}],
}

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResumeAnalysisRequest(BaseModel):
    file_base64: str = Field(..., description="Base64-encoded PDF or DOCX file.")


class DocumentAnalysisRequest(BaseModel):
    file_base64: str = Field(..., description="Base64-encoded PDF or DOCX file.")
    json_structure: dict = Field(
        ..., description="Target JSON structure that should be filled from the document."
    )
    prompt: str = Field(
        ..., description="Instructions that guide how the model should fill the structure."
    )


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def detect_file_type(file_bytes: bytes) -> str:
    """Return 'pdf', 'docx', or 'unknown' based on magic bytes."""
    if file_bytes[:4] == PDF_MAGIC_BYTES:
        return "pdf"
    if file_bytes[:2] == DOCX_MAGIC_BYTES:
        return "docx"
    return "unknown"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract selectable text from a PDF using pypdf."""
    reader = PdfReader(BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "".join(pages_text)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract plain text from a Word document."""
    document = Document(BytesIO(file_bytes))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text_via_ocr(file_bytes: bytes) -> str:
    """Fallback OCR extraction for scanned / image-based PDFs."""
    images = convert_from_bytes(file_bytes)
    return "".join(pytesseract.image_to_string(image) for image in images)


def extract_text(file_bytes: bytes) -> str:
    """
    Detect the file type and extract all readable text.
    Falls back to OCR when a PDF contains no selectable text.

    Raises:
        HTTPException: If the file type is not supported.
    """
    file_type = detect_file_type(file_bytes)

    if file_type == "pdf":
        text = extract_text_from_pdf(file_bytes)
        if not text.strip():
            logger.info("No selectable text found in PDF — falling back to OCR.")
            text = extract_text_via_ocr(file_bytes)
        return text

    if file_type == "docx":
        return extract_text_from_docx(file_bytes)

    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=f"Unsupported file type. Expected one of: {SUPPORTED_FILE_TYPES}.",
    )


# ---------------------------------------------------------------------------
# OpenAI helper
# ---------------------------------------------------------------------------

def call_openai(prompt: str) -> dict:
    """
    Send a prompt to OpenAI and return a parsed JSON dict.

    Raises:
        HTTPException: If the model response cannot be parsed as JSON.
    """
    response = openai_client.responses.create(model=MODEL_NAME, input=prompt)
    raw_text = response.output_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse OpenAI response as JSON: %s", raw_text)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The model returned an invalid JSON response.",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/resumes/analyze",
    summary="Analyze a resume",
    response_description="Structured resume data as JSON.",
    status_code=status.HTTP_200_OK,
)
async def analyze_resume(request: ResumeAnalysisRequest) -> dict:
    """
    Receive a base64-encoded PDF or DOCX resume, extract its text,
    and return structured information using a fixed JSON schema.
    """
    logger.info("Received request to analyze resume.")

    file_bytes = base64.b64decode(request.file_base64)
    document_text = extract_text(file_bytes)

    if not document_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract any text from the provided file.",
        )

    json_template = json.dumps(CV_JSON_TEMPLATE, indent=4)

    prompt = f"""
Extract the information from the following resume and return ONLY valid JSON.
Any data not found should be returned as empty text.

{json_template}

Do not add explanations. Return ONLY valid JSON.

Resume:
{document_text}
"""

    result = call_openai(prompt)
    logger.info("Resume analysis completed successfully.")
    return result


@app.post(
    "/documents/analyze",
    summary="Analyze a document with a custom schema",
    response_description="Document data structured according to the provided JSON schema.",
    status_code=status.HTTP_200_OK,
)
async def analyze_document(request: DocumentAnalysisRequest) -> dict:
    """
    Receive a base64-encoded PDF or DOCX file, a target JSON structure,
    and a prompt. Returns the document data mapped to the given structure.
    """
    logger.info("Received request to analyze document with custom schema.")

    file_bytes = base64.b64decode(request.file_base64)
    document_text = extract_text(file_bytes)

    if not document_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract any text from the provided file.",
        )

    json_structure = json.dumps(request.json_structure, ensure_ascii=False, indent=4)

    prompt = f"""
{request.prompt}

Fill the following JSON structure using the document content below.
Any data not found should be returned as empty text or empty array as appropriate.
Return ONLY valid JSON, no explanations.

JSON structure:
{json_structure}

Document:
{document_text}
"""

    result = call_openai(prompt)
    logger.info("Document analysis completed successfully.")
    return result
