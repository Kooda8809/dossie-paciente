import os
import json
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# A importação da biblioteca nova do Google
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalisarRequest(BaseModel):
    username: str

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not APIFY_TOKEN:
    raise RuntimeError("APIFY_TOKEN não definido")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY não definida")

# Inicia o cliente com a sintaxe nova
client = genai.Client(api_key=GEMINI_API_KEY)


def extrair_instagram(username: str):
    run_resp = requests.post(
        "https://api.apify.com/v2/acts/apify~instagram-scraper/runs",
        params={"token": APIFY_TOKEN},
        json={
            "directUrls": [f"https://www.instagram.com/{username}/"],
            "resultsType": "details",
            "resultsLimit": 5,
        },
    )
    run_resp.raise_for_status()
    run_id = run_resp.json()["data"]["id"]

    url_status = f"https://api.apify.com/v2/actor-runs/{run_id}"
    while True:
        s = requests.get(url_status, params={"token": APIFY_TOKEN}).json()
        status = s["data"]["status"]
        if status == "SUCCEEDED":
            break
        if status in ("FAILED", "ABORTED", "TIMED-OUT"):
            raise HTTPException(502, f"Apify falhou com status {status}")
        time.sleep(2)

    items = requests.get(
        f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items",
        params={"token": APIFY_TOKEN},
    ).json()

    if not items:
        raise HTTPException(400, "Perfil não encontrado ou sem dados.")

    profile = items[0]
    bio = profile.get("biography") or ""

    posts_raw = profile.get("latestPosts") or []
    posts = []
    for p in posts_raw[:5]:
        caption = p.get("caption") or p.get("text") or ""
        if caption:
            posts.append(caption.strip())

    return bio, posts


def analisar_com_gemini(texto: str):
    prompt = (
        'Você é um estrategista de experiência do paciente (Patient Experience) '
        'em uma clínica odontológica de estética de altíssimo padrão. '
        'Analise o Instagram deste futuro paciente para que o dentista possa '
        'personalizar o atendimento e criar momentos de encantamento. '
        'Retorne APENAS um JSON estrito no seguinte formato: '
        '{"estilo_atendimento": "Descreva como o dentista deve agir", '
        '"estilo_vestuario_design": "Identifique o estilo de roupas", '
        '"gostos_premium": ["lista", "de", "gostos"], '
        '"sugestao_presente": "Sugira um presente", '
        '"quebra_gelo": "A melhor pergunta para iniciar a conversa"}\n\n'
        f'Texto do perfil:\n{texto}'
    )

    # Chamada atualizada exigindo o formato JSON nativamente
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    return json.loads(response.text)


@app.post("/analisar")
async def analisar(req: AnalisarRequest):
    username = req.username.strip().lstrip("@")
    bio, posts = extrair_instagram(username)
    texto_bruto = f"Bio: {bio}\n\nÚltimos posts:\n" + "\n---\n".join(posts)
    resultado = analisar_com_gemini(texto_bruto)
    return resultado