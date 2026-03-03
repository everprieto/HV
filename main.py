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

    texto_cv = extraer_texto_pdf(file.file)

    if not texto_cv.strip():
        return {"error": "No se pudo extraer texto del PDF"}

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


    #return {"resultado": response.output_text}
    
    texto_respuesta = response.output_text
    texto_respuesta = texto_respuesta.replace("```json", "").replace("```", "").strip()

    return json.loads(texto_respuesta)