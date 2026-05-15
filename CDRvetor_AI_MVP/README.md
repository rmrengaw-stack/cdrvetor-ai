# CDRvetor AI - MVP

Protótipo web para vetorizar imagens e retornar SVG.

## Como publicar no Render

1. Crie conta em https://render.com
2. Crie um novo Web Service
3. Envie estes arquivos para um repositório GitHub
4. No Render, escolha o repositório
5. Build Command: `pip install -r requirements.txt`
6. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

Depois abra a URL pública gerada pelo Render.

## Rotas

- `/` página web
- `/vectorize` API POST para enviar imagem
- `/docs` documentação automática da API

## Observação

Este MVP separa a imagem por cores e gera paths SVG.
Ainda não é a versão profissional final com OCR, degradês reais e curvas Bézier avançadas.
