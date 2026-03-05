from fastapi import FastAPI
from pydantic import BaseModel
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO
import os
import json
import base64

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 👇 Modelo que recibirá el JSON
class CVRequest(BaseModel):
    file_base64: str


def extraer_texto_pdf(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    texto = ""
    for page in reader.pages:
        contenido = page.extract_text()
        if contenido:
            texto += contenido
    return texto


@app.post("/analizar-cv")
async def analizar_cv(request: CVRequest):

    try:
        # 🔹 1️⃣ Decodificar Base64
        pdf_bytes = base64.b64decode(request.file_base64)

        # 🔹 2️⃣ Extraer texto
        texto_cv = extraer_texto_pdf(pdf_bytes)

        if not texto_cv.strip():
            return {"error": "No se pudo extraer texto del PDF"}

        # 🔹 3️⃣ Enviar a OpenAI
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
    "EDUCATION": [{{"DEGREE": "", "COURSE": "", "INSTITUTION": "", "YEAR": ""}}],
    "CERTIFICATIONS": [{{"CERTIFICATION": "", "INSTITUTION": "", "YEAR": ""}}],
    "LANGUAGES": [{{"LANGUAGE": "", "LEVEL": ""}}],
    "EXPERIENCES": [{{"POSITION": "", "COMPANY": "", "PERIOD": "", "DESCRIPTION": ""}}],
    "ACTIVITIES": [{{"ACTIVITY": ""}}],
    "TECHNOLOGIES": [{{"TECHNOLOGY": ""}}]
}}

Do not add explanations.
Do not add additional text.
Only valid JSON.

CV:
{texto_cv}
"""
        )

        texto_respuesta = response.output_text
        texto_respuesta = texto_respuesta.replace("```json", "").replace("```", "").strip()

        return json.loads(texto_respuesta)

    except Exception as e:
        return {"error": str(e)}