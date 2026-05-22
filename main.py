import os
import json
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

client = genai.Client(api_key=GEMINI_API_KEY)

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

def analisar_com_gemini(texto: str):
    prompt = (
        'Você é um auditor financeiro implacável, frio e experiente estrategista de vendas '
        'em uma clínica odontológica de estética de altíssimo ticket (High-Ticket). '
        'Analise os dados deste futuro paciente para mapear sua real capacidade financeira e perfil psicológico. '
        'REGRAS CRÚCIAIS DE VALORAÇÃO:\n'
        '1. Considere a relação de Seguidores vs Seguindo para entender o status social e o ego da pessoa.\n'
        '2. NÃO seja polido, otimista ou complacente. Avalie o histórico longo de postagens. Se não houver sinais explícitos de luxo, classifique categoricamente como Baixo ou Médio Padrão.\n'
        '3. Identifique marcas de luxo específicas, destinos de viagens ou hábitos.\n\n'
        'Retorne UNICAMENTE um JSON estrito no seguinte formato exato:\n'
        '{\n'
        '  "capacidade_pagamento": "Veredito direto (Alto, Médio ou Baixo Padrão). Justifique friamente com base na presença/ausência de patrimônio e status social (seguidores).",\n'
        '  "perfil_disc": "Classifique em apenas uma palavra (Dominante, Influente, Estável ou Conforme) seguido de uma frase curta ensinando como falar com esse tipo de mente.",\n'
        '  "objecao_principal": "Qual será o principal entrave na venda? (Preço, Tempo, Medo de Dor, Necessidade de aprovação de terceiros). Defina e dê a contra-argumentação.",\n'
        '  "red_flags": "Identifique traços de vitimismo, inclinação a reclamações ou polêmicas. Se o perfil parecer tranquilo, retorne uma string vazia.",\n'
        '  "match_estetica": "Defina a linha estética predileta do paciente com base no estilo de fotos: Naturalidade Absoluta (Discreto/Elegant) ou Branco Extravagante (Marcante/Alta visibilidade).",\n'
        '  "radar_concorrentes": "Analise se há indícios de que ele busca ou consome outros players do mercado estético de luxo, ou se é altamente blindado. Retorne vazio se não notar nada.",\n'
        '  "estilo_atendimento": "Script de condução comercial focada no fechamento do tratamento sem dar descontos.",\n'
        '  "estilo_vestuario_design": "Relação de grifes, relógios e vestuário identificados ou deduzidos pelo nível de sofisticação.",\n'
        '  "gostos_premium": ["Lista", "de hobbies", "destinos de viagem", "bens de valor"],\n'
        '  "sugestao_presente": "Apresente 3 opções claras de mimos corporativos personalizados: 1) Um item minimalista/artesanal de alto padrão, 2) Uma experiência memorável, 3) Um item de luxo de marca consolidada.",\n'
        '  "quebra_gelo": "A melhor frase de abertura magnética baseada no ego ou grande interesse do paciente."\n'
        '}\n\n'
        f'Dados coletados do perfil:\n{texto}'
    )

    # 🚀 BLINDAGEM ATUALIZADA PARA CHAVES NOVAS (Geração 2.0)
    modelos_para_testar = [
        'gemini-2.0-flash',           # O modelo oficial atual liberado para chaves novas
        'gemini-2.0-flash-exp',       # Versão experimental (sempre com cota liberada)
        'gemini-1.5-flash'            # Backup caso a chave antiga ressuscite
    ]

    ultimo_erro = None

    for modelo in modelos_para_testar:
        try:
            print(f"Tentando motor: {modelo}...")
            response = client.models.generate_content(
                model=modelo,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )

            texto_resposta = response.text.strip()
            if texto_resposta.startswith("```json"):
                texto_resposta = texto_resposta.removeprefix("```json").removesuffix("```").strip()
            elif texto_resposta.startswith("```"):
                texto_resposta = texto_resposta.removeprefix("```").removesuffix("```").strip()

            print(f"Sucesso com o motor: {modelo}!")
            return json.loads(texto_resposta)
            
        except Exception as e:
            print(f"Motor {modelo} falhou. Motivo: {e}. Trocando para o próximo...")
            ultimo_erro = e
            continue 

    raise HTTPException(status_code=500, detail=f"Todos os motores da IA falharam. Último erro: {ultimo_erro}")

@app.post("/analisar")
async def analisar(req: AnalisarRequest):
    username = req.username.strip().lstrip("@")
    bio, followers, following, posts = extrair_instagram(username)
    
    texto_bruto = f"MÉTRICAS DE STATUS:\nSeguidores: {followers}\nSeguindo: {following}\n\nBIO:\n{bio}\n\nÚLTIMOS 8 POSTS:\n" + "\n---\n".join(posts)
    
    resultado = analisar_com_gemini(texto_bruto)
    return resultado