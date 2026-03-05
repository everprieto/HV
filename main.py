from fastapi import FastAPI
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO
import pytesseract
from pdf2image import convert_from_bytes
import os
import json
import base64

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CVRequest(BaseModel):
    file_base64: str


# -------- EXTRAER TEXTO PDF --------
def extraer_texto_pdf(file_bytes):

    reader = PdfReader(BytesIO(file_bytes))
    texto = ""

    for page in reader.pages:
        contenido = page.extract_text()
        if contenido:
            texto += contenido

    return texto


# -------- EXTRAER TEXTO WORD --------
def extraer_texto_docx(file_bytes):

    doc = Document(BytesIO(file_bytes))
    texto = ""

    for p in doc.paragraphs:
        texto += p.text + "\n"

    return texto


# -------- OCR PARA PDF ESCANEADO --------
def extraer_texto_ocr(file_bytes):

    images = convert_from_bytes(file_bytes)

    texto = ""

    for img in images:
        texto += pytesseract.image_to_string(img)

    return texto


# -------- DETECTAR TIPO DE ARCHIVO --------
def detectar_tipo_archivo(file_bytes):

    if file_bytes[:4] == b'%PDF':
        return "pdf"

    if file_bytes[:2] == b'PK':
        return "docx"

    return "unknown"


@app.post("/analizar-cv")
async def analizar_cv(request: CVRequest):

    try:

        # 1️⃣ Decodificar archivo
        file_bytes = base64.b64decode(request.file_base64)

        # 2️⃣ Detectar tipo
        tipo = detectar_tipo_archivo(file_bytes)

        # 3️⃣ Extraer texto
        if tipo == "pdf":

            texto_cv = extraer_texto_pdf(file_bytes)

            # si el PDF no tiene texto usar OCR
            if not texto_cv.strip():
                texto_cv = extraer_texto_ocr(file_bytes)

        elif tipo == "docx":

            texto_cv = extraer_texto_docx(file_bytes)

        else:
            return {"error": "Tipo de archivo no soportado"}

        if not texto_cv.strip():
            return {"error": "No se pudo extraer texto del archivo"}

        # 4️⃣ Enviar a OpenAI
        response = client.responses.create(
            model="gpt-4o-mini",
            input=f"""
Extract the information from the following resume and return
ONLY valid JSON.

Any data not found should be returned as empty text.

{{
    "NAME": "",
    "CITY_STATE": "",
    "LAST_POSITION": "",
    "SUMMARY": "",
    "EDUCATION": [
        {{"DEGREE": "", "COURSE": "", "INSTITUTION": "", "YEAR": ""}}
    ],
    "CERTIFICATIONS": [
        {{"CERTIFICATION": "", "INSTITUTION": "", "YEAR": ""}}
    ],
    "LANGUAGES": [
        {{"LANGUAGE": "", "LEVEL": ""}}
    ],
    "EXPERIENCES": [
        {{"POSITION": "", "COMPANY": "", "PERIOD": "", "DESCRIPTION": ""}}
    ],
    "ACTIVITIES": [
        {{"ACTIVITY": ""}}
    ],
    "TECHNOLOGIES": [
        {{"TECHNOLOGY": ""}}
    ]
}}

Do not add explanations.
Return ONLY valid JSON.

CV:
{texto_cv}
"""
        )

        texto_respuesta = response.output_text
        texto_respuesta = texto_respuesta.replace("```json", "").replace("```", "").strip()

        return json.loads(texto_respuesta)

    except Exception as e:

        return {"error": str(e)}