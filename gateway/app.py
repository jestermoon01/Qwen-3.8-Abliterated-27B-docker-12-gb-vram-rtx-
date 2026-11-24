"""Local chat UI and transparent OpenAI-compatible proxy to llama.cpp."""
from contextlib import asynccontextmanager
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

BACKEND = os.getenv('BACKEND', 'http://inference:8080')
MODEL = os.getenv('MODEL_ALIAS', 'huihui-qwen3.8-27b')
CONTEXT = int(os.getenv('CONTEXT_PER_SLOT', '32768'))


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(base_url=BACKEND, timeout=httpx.Timeout(1800, connect=10),
                                 trust_env=False) as client:
        app.state.backend = client
        yield


app = FastAPI(title='Local Qwen 27B inference', lifespan=lifespan)


@app.get('/', include_in_schema=False)
async def ui():
    return FileResponse(Path(__file__).with_name('static') / 'index.html',
                        headers={'Cache-Control': 'no-store'})


@app.get('/config')
async def config():
    return {'model': MODEL, 'context_per_request': CONTEXT,
            'default_system': 'You are a helpful assistant.'}


@app.get('/health')
async def health(request: Request):
    try:
        response = await request.app.state.backend.get('/health', timeout=5)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(503, 'Model is not ready') from error
    return {'status': 'ok', 'context_per_request': CONTEXT}


@app.api_route('/v1/{path:path}', methods=['GET', 'POST'])
async def api(path: str, request: Request):
    client = request.app.state.backend
    headers = {'Content-Type': request.headers.get('content-type', 'application/json')}
    try:
        response = await client.send(client.build_request(
            request.method, '/v1/' + path, params=request.query_params,
            content=await request.body(), headers=headers), stream=True)
    except httpx.RequestError:
        return JSONResponse({'error': {'message': 'Inference server unavailable'}}, status_code=503)
    return StreamingResponse(response.aiter_bytes(), status_code=response.status_code,
                             headers={'Content-Type': response.headers.get('content-type', 'application/json')},
                             background=BackgroundTask(response.aclose))
