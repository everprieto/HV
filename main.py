from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from io import BytesIO

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extraer_texto_pdf(file):
    reader = PdfReader(file)
    texto = ""
    for page in reader.pages:
        contenido = page.extract_text()
        if contenido:
            texto += contenido
    return texto

@app.post("/analizar-cv")
async def analizar_cv(file: UploadFile = File(...)):

    content = await file.read()

    print("First 20 bytes:", content[:20])
    print("Starts with %PDF?:", content.startswith(b"%PDF"))

    pdf_stream = BytesIO(content)
    pdf_stream.seek(0)   # 👈 fuerza posición 0

    reader = PdfReader(pdf_stream)

    print("Number of pages:", len(reader.pages))

    texto = ""
    for page in reader.pages:
        contenido = page.extract_text()
        print("Page text length:", len(contenido) if contenido else 0)
        if contenido:
            texto += contenido

    return {"texto_length": len(texto)}