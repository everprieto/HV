from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

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

    print("Content type:", file.content_type)
    print("First 10 bytes:", content[:10])
    print("Size:", len(content))

    return {"size": len(content)}