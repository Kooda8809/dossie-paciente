import os
import json
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not APIFY_TOKEN:
    raise RuntimeError("APIFY_TOKEN não definido")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY não definida")

def extrair_instagram(username: str):
    run_resp = requests.post(
        "https://api.apify.com/v2/acts/apify~instagram-scraper/runs",
        params={"token": APIFY_TOKEN},
        json={
            "directUrls": [f"https://www.instagram.com/{username}/"],
            "resultsType": "details",
            "resultsLimit": 8, 
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
        raise HTTPException(400, "Perfil não encontrado ou bloqueado pelo anti-bot do Instagram.")

    profile = items[0]
    bio = profile.get("biography") or ""
    
    followers = profile.get("followersCount", 0)
    following = profile.get("followsCount", 0)

    posts_raw = profile.get("latestPosts") or []
    posts = []
    for p in posts_raw:
        caption = p.get("caption") or p.get("text") or ""
        if caption:
            posts.append(caption.strip())

    return bio, followers, following, posts

def analisar_com_groq(texto: str):
    prompt = (
        'Você é um estrategista de negócios de elite e analista comportamental '
        'trabalhando para uma clínica odontológica de estética e reabilitação de altíssimo padrão. '
        'Sua missão é fornecer dossiês profundos, sofisticados e altamente detalhados sobre o paciente. '
        'REGRAS CRÚCIAIS DE VALORAÇÃO:\n'
        '1. ESCREVA PARÁGRAFOS RICOS E BEM DESENVOLVIDOS. É estritamente proibido dar respostas curtas. Use um tom de consultoria premium e elegante.\n'
        '2. Analise profundamente a relação de Seguidores vs Seguindo, a biografia e o tom das postagens para deduzir o ego e a personalidade.\n'
        '3. Seja realista e implacável: justifique detalhadamente o PORQUÊ de cada conclusão com base nos sinais lidos.\n\n'
        'Retorne UNICAMENTE um JSON estrito no seguinte formato exato:\n'
        '{\n'
        '  "capacidade_pagamento": "Veredito (Alto/Médio/Baixo) seguido de um parágrafo detalhado justificando a análise financeira e o nível de sofisticação do lifestyle.",\n'
        '  "perfil_disc": "Classificação (Dominante, Influente, Estável ou Conforme) acompanhada de uma análise psicológica profunda de como conduzir a comunicação e a venda com essa mente.",\n'
        '  "objecao_principal": "Escreva um parágrafo robusto identificando o provável maior obstáculo (preço, tempo, medo) e entregando a estratégia argumentativa para quebrá-lo.",\n'
        '  "red_flags": "Análise detalhada de possíveis traços tóxicos, vitimismo ou nível de exigência. Se não houver, explique por que o perfil indica ser um excelente paciente.",\n'
        '  "match_estetica": "Parágrafo detalhado sobre a preferência visual (Naturalidade Absoluta vs Branco Extravagante) e como ancorar o valor do tratamento nisso.",\n'
        '  "radar_concorrentes": "Análise profunda sobre o consumo de luxo, padrão de exigência e se o paciente parece buscar status através de marcas conhecidas.",\n'
        '  "estilo_atendimento": "Um roteiro comercial extenso e estratégico, detalhando a postura clínica, o tom de voz e os gatilhos de autoridade a serem usados.",\n'
        '  "estilo_vestuario_design": "Descrição minuciosa do estilo pessoal, deduzindo grifes e o que esse padrão de exigência visual significa para o trabalho do dentista.",\n'
        '  "gostos_premium": ["Hobby ou Marca 1", "Destino ou Item 2", "Interesse 3", "Característica 4"],\n'
        '  "sugestao_presente": "Três opções descritivas e detalhadas de presentes corporativos sofisticados, com a justificativa de por que cada um encantaria este perfil.",\n'
        '  "quebra_gelo": "Uma ou duas frases magnéticas, personalizadas e altamente elegantes para abrir a conversa gerando rapport imediato."\n'
        '}\n\n'
        f'Dados coletados do perfil:\n{texto}'
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Você é um consultor premium que escreve análises longas, discursivas, detalhadas e sofisticadas. Retorne apenas JSON válido."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.6 # Destravando a eloquência e criatividade do modelo
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    
    resultado_str = response.json()["choices"][0]["message"]["content"]
    
    texto_resposta = resultado_str.strip()
    if texto_resposta.startswith("```json"):
        texto_resposta = texto_resposta.removeprefix("```json").removesuffix("```").strip()
    elif texto_resposta.startswith("```"):
        texto_resposta = texto_resposta.removeprefix("```").removesuffix("```").strip()
        
    return json.loads(texto_resposta)

@app.post("/analisar")
async def analisar(req: AnalisarRequest):
    username = req.username.strip().lstrip("@")
    bio, followers, following, posts = extrair_instagram(username)
    
    texto_bruto = f"MÉTRICAS DE STATUS:\nSeguidores: {followers}\nSeguindo: {following}\n\nBIO:\n{bio}\n\nÚLTIMOS 8 POSTS:\n" + "\n---\n".join(posts)
    
    resultado = analisar_com_groq(texto_bruto)
    return resultado