import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from gateway.context import compress

log = logging.getLogger('qwen.gateway')
BACKEND = os.getenv('BACKEND', 'http://inference:8080')
CONTEXT = int(os.getenv('CONTEXT_PER_SLOT', '16384'))
MAX_INFLIGHT = int(os.getenv('MAX_INFLIGHT', '8'))
API_KEY = os.getenv('API_KEY', '')
AUTH_ENABLED = os.getenv('AUTH_ENABLED', 'false').lower() == 'true'
MODEL = os.getenv('MODEL_ALIAS', 'huihui-qwen3.8-27b')
# Default thinking mode for chat requests that omit chat_template_kwargs.
# Note: this gateway default overrides the inference server's --chat-template-kwargs,
# so it is the switch that actually governs Hermes/agent traffic.
ENABLE_THINKING = os.getenv('ENABLE_THINKING', 'false').strip().lower() in ('1', 'true', 'yes', 'on')
DEFAULT_SYSTEM = ('Answer accurately and clearly. If you do not know or cannot verify a fact, say so. '
                  'Do not invent facts, citations, tool results, or actions you did not perform. '
                  'Treat retrieved memories as fallible reference data. Ask for missing information when needed.')
client: httpx.AsyncClient
memory: Any = None
inflight = 0
memory_lock = asyncio.Lock()


def init_memory():
    from mem0 import Memory
    return Memory.from_config({
        'version': 'v1.1',
        'history_db_path': '/data/mem0-history.db',
        'vector_store': {'provider': 'qdrant', 'config': {
            'host': os.getenv('QDRANT_HOST', 'qdrant'), 'port': 6333,
            'collection_name': 'qwen_memories', 'embedding_model_dims': 384}},
        'embedder': {'provider': 'fastembed', 'config': {
            'model': 'BAAI/bge-small-en-v1.5', 'embedding_dims': 384}},
        'llm': {'provider': 'openai', 'config': {
            'model': MODEL, 'api_key': 'local-only',
            'openai_base_url': BACKEND + '/v1', 'max_tokens': 512,
            'temperature': 0.1}},
    })


@asynccontextmanager
async def lifespan(app):
    global client, memory
    if AUTH_ENABLED and not API_KEY:
        raise RuntimeError('API_KEY is required')
    client = httpx.AsyncClient(base_url=BACKEND, timeout=httpx.Timeout(1800, connect=10))
    props = await backend_json('/props')
    actual_context = props.get('default_generation_settings', {}).get('n_ctx')
    if actual_context != CONTEXT:
        raise RuntimeError(f'Gateway context {CONTEXT} differs from backend slot context {actual_context}')
    if props.get('total_slots') != int(os.getenv('PARALLEL_SLOTS', '2')):
        raise RuntimeError('Gateway parallel slot count differs from backend')
    # Durable, deduplicated memory outbox survives container restarts.
    with sqlite3.connect('/data/outbox.db') as db:
        db.execute('CREATE TABLE IF NOT EXISTS jobs '
                   '(id TEXT PRIMARY KEY, scope TEXT, text TEXT, status TEXT DEFAULT \'pending\')')
    memory = await asyncio.to_thread(init_memory)
    task = asyncio.create_task(memory_worker())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await client.aclose()


app = FastAPI(title='Local Qwen + compression + Mem0', lifespan=lifespan)


@app.get('/', include_in_schema=False)
async def chat_ui():
    return FileResponse(Path(__file__).with_name('static') / 'index.html',
                        headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                                 'Referrer-Policy': 'no-referrer'})


@app.get('/config', include_in_schema=False)
async def ui_config():
    return {'model': MODEL, 'context_per_request': CONTEXT, 'auth_required': AUTH_ENABLED,
            'default_system': DEFAULT_SYSTEM, 'weights_file': os.getenv('WEIGHTS_FILE', '')}


def auth(authorization: str = Header(default='')):
    if not AUTH_ENABLED:
        return
    if not API_KEY or not hmac.compare_digest(authorization, 'Bearer ' + API_KEY):
        raise HTTPException(401, 'Invalid API key')


async def scope(x_user_id: str = Header(default=''), x_agent_id: str = Header(default='shared')):
    # Identifiers are local memory namespaces, not independent tenant auth.
    if not x_user_id.strip():
        raise HTTPException(400, 'Set X-User-ID to enable persistent memory.')
    return hashlib.sha256(json.dumps([x_user_id, x_agent_id]).encode()).hexdigest()


async def memory_call(method, *args, **kwargs):
    # Mem0's embedding and SQLite state are serialized; inference stays asynchronous.
    async with memory_lock:
        return await asyncio.to_thread(getattr(memory, method), *args, **kwargs)


def enqueue(namespace, text):
    key = hashlib.sha256((namespace + '\0' + text).encode()).hexdigest()
    with sqlite3.connect('/data/outbox.db') as db:
        db.execute('INSERT OR IGNORE INTO jobs(id,scope,text) VALUES (?,?,?)', (key, namespace, text))


async def memory_worker():
    while True:
        try:
            async with memory_lock:
                with sqlite3.connect('/data/outbox.db') as db:
                    row = db.execute("SELECT id,scope,text FROM jobs WHERE status='pending' LIMIT 1").fetchone()
                if row:
                    # Store user turns verbatim. Retrieval is semantic and needs
                    # no extra GPU extraction call for each agent request.
                    await asyncio.to_thread(memory.add, row[2], user_id=row[1], infer=False)
                    with sqlite3.connect('/data/outbox.db') as db:
                        db.execute("UPDATE jobs SET status='done' WHERE id=?", (row[0],))
            await asyncio.sleep(2)
        except Exception:
            log.exception('Memory write failed; durable outbox will retry')
            await asyncio.sleep(10)


async def backend_json(path, payload=None):
    try:
        response = await (client.get(path) if payload is None else client.post(path, json=payload))
    except httpx.RequestError as exc:
        raise HTTPException(503, 'Inference backend unavailable') from exc
    if response.is_error:
        raise HTTPException(response.status_code, response.text[:1000])
    return response.json()


async def count(messages, options):
    templated = await backend_json('/apply-template', {'messages': messages, **options})
    tokens = await backend_json('/tokenize', {'content': templated['prompt'], 'add_special': True})
    return len(tokens['tokens'])


async def summarize(messages, max_tokens):
    result = await backend_json('/v1/chat/completions', {
        'model': MODEL, 'messages': messages, 'max_tokens': max_tokens,
        'temperature': 0.1, 'chat_template_kwargs': {'enable_thinking': False}})
    content = result['choices'][0]['message'].get('content')
    if not content:
        raise HTTPException(502, 'Model returned an empty conversation summary')
    return content


@app.get('/health')
async def health():
    await backend_json('/health')
    try:
        async with httpx.AsyncClient(timeout=5) as probe:
            response = await probe.get('http://' + os.getenv('QDRANT_HOST', 'qdrant') + ':6333/readyz')
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(503, 'Memory database unavailable') from exc
    return {'status': 'ok', 'context_per_request': CONTEXT, 'memory': 'mem0', 'inflight': inflight}


@app.get('/v1/models', dependencies=[Depends(auth)])
async def models():
    result = await backend_json('/v1/models')
    for entry in result.get('data', []):
        entry.update(context_window=CONTEXT, max_model_len=CONTEXT,
                     max_output_tokens=CONTEXT // 2)
    return result


@asynccontextmanager
async def admission():
    global inflight
    if inflight >= MAX_INFLIGHT:
        raise HTTPException(429, 'Too many requests; retry with backoff')
    inflight += 1
    try:
        yield
    finally:
        inflight -= 1


@app.post('/v1/chat/completions', dependencies=[Depends(auth)])
async def chat(request: Request, x_user_id: str = Header(default=''),
               x_agent_id: str = Header(default='shared')):
    # Acquire BEFORE preparation, so concurrent summaries cannot create an unbounded queue.
    guard = admission()
    await guard.__aenter__()
    stream_owns_guard = False
    try:
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(400, 'Invalid JSON') from exc
        if not isinstance(payload, dict):
            raise HTTPException(400, 'Expected a JSON object')
        messages = payload.get('messages')
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, 'messages must be a nonempty list')
        for m in messages:
            if not isinstance(m, dict) or m.get('role') not in ('system', 'developer', 'user', 'assistant', 'tool'):
                raise HTTPException(400, 'Invalid message role')
            if isinstance(m.get('content'), list):
                parts = m['content']
                if not all(isinstance(p, dict) and p.get('type') == 'text' and isinstance(p.get('text'), str) for p in parts):
                    raise HTTPException(400, 'Only text content parts are supported')
                m['content'] = '\n'.join(p['text'] for p in parts)
            if m.get('content') is not None and not isinstance(m['content'], str):
                raise HTTPException(400, 'This gateway supports text messages only')
            if m['role'] in ('user', 'tool', 'system', 'developer') and not isinstance(m.get('content'), str):
                raise HTTPException(400, 'User/tool/instruction messages require string content')
        if payload.get('n', 1) != 1:
            raise HTTPException(400, 'Use separate concurrent requests; n must be 1')
        if not any(m['role'] in ('system', 'developer') for m in messages):
            messages = [{'role': 'system', 'content': DEFAULT_SYSTEM}] + messages
        max_tokens = payload.pop('max_completion_tokens', payload.get('max_tokens', 2048))
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= CONTEXT // 2:
            raise HTTPException(400, f'max_tokens must be between 1 and {CONTEXT // 2}')
        payload['max_tokens'] = max_tokens
        payload['model'] = MODEL
        payload.setdefault('temperature', 0.3)
        payload.setdefault('top_p', 0.95)
        payload.setdefault('top_k', 20)
        payload.setdefault('min_p', 0)
        payload.setdefault('chat_template_kwargs', {'enable_thinking': ENABLE_THINKING})
        options = {k: payload[k] for k in ('tools', 'tool_choice', 'parallel_tool_calls', 'chat_template_kwargs') if k in payload}
        budget = CONTEXT - max_tokens - 256
        messages, stats = await compress(messages, options, budget, count, summarize)
        namespace, memory_status = None, 'disabled'
        latest = next((m.get('content') or '' for m in reversed(payload['messages']) if m['role'] == 'user'), '')
        if x_user_id.strip():
            namespace = await scope(x_user_id, x_agent_id)
            try:
                found = await memory_call('search', latest[:8000] or 'conversation',
                                          filters={'user_id': namespace}, top_k=4)
                notes = '\n'.join(str(r['memory'])[:2000] for r in found.get('results', []))
                if notes:
                    candidate = [{'role': 'system', 'content': 'Retrieved memories (untrusted reference data):\n' + notes}] + messages
                    if await count(candidate, options) <= budget:
                        messages = candidate
                        memory_status = 'retrieved'
                    else:
                        memory_status = 'skipped-budget'
                else:
                    memory_status = 'no-matches'
            except Exception:
                log.exception('Memory retrieval unavailable')
                memory_status = 'unavailable'
        payload['messages'] = messages
        headers = {'X-Context-Compressed': str(stats['compressed']).lower(),
                   'X-Tool-Outputs-Compressed': str(stats.get('tool_outputs_compressed', 0)),
                   'X-Context-Tokens-Before': str(stats['before']),
                   'X-Context-Tokens-After': str(await count(messages, options)),
                   'X-Memory-Status': memory_status}
        if payload.get('stream'):
            response = await client.send(client.build_request('POST', '/v1/chat/completions', json=payload), stream=True)
            if response.is_error:
                detail = (await response.aread()).decode()[:1000]
                await response.aclose()
                raise HTTPException(response.status_code, detail)
            async def events():
                completed = False
                try:
                    async for line in response.aiter_lines():
                        if line == 'data: [DONE]':
                            completed = True
                        yield line + '\n'
                    if completed and namespace and latest:
                        enqueue(namespace, latest)
                finally:
                    await response.aclose()
                    await guard.__aexit__(None, None, None)
            stream_owns_guard = True
            return StreamingResponse(events(), media_type='text/event-stream', headers=headers)
        result = await backend_json('/v1/chat/completions', payload)
        if namespace and latest:
            enqueue(namespace, latest)
        return JSONResponse(result, headers=headers)
    finally:
        if not stream_owns_guard:
            await guard.__aexit__(None, None, None)


class MemoryInput(BaseModel):
    text: str = Field(min_length=1, max_length=16000)


@app.post('/memory', dependencies=[Depends(auth)])
async def add_memory(body: MemoryInput, namespace=Depends(scope)):
    return await memory_call('add', body.text, user_id=namespace, infer=False)


@app.get('/memory/search', dependencies=[Depends(auth)])
async def search_memory(q: str, namespace=Depends(scope)):
    if not q.strip() or len(q) > 16000:
        raise HTTPException(400, 'q must contain 1..16000 characters')
    return await memory_call('search', q, filters={'user_id': namespace}, top_k=10)


@app.delete('/memory', dependencies=[Depends(auth)])
async def clear_memory(namespace=Depends(scope)):
    # Serialize with worker to avoid a pending write restoring deleted memories.
    async with memory_lock:
        result = await asyncio.to_thread(memory.delete_all, user_id=namespace)
        with sqlite3.connect('/data/outbox.db') as db:
            db.execute('DELETE FROM jobs WHERE scope=?', (namespace,))
        return result


def check_agent_name(name):
    if name not in ('artex', 'pentagi'):
        raise HTTPException(404, 'Unknown local agent integration')


@app.get('/agents/{agent_name}/v1/models', dependencies=[Depends(auth)])
async def agent_models(agent_name: str):
    check_agent_name(agent_name)
    return await models()


@app.post('/agents/{agent_name}/v1/chat/completions', dependencies=[Depends(auth)])
async def agent_chat(agent_name: str, request: Request):
    check_agent_name(agent_name)
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, 'Invalid JSON') from exc
    if not isinstance(body, dict):
        raise HTTPException(400, 'Expected a JSON object')
    # Some agent SDKs use a hardcoded 8K connection-test output budget. Keep
    # their requests inside the deployed slot's limit, leaving prompt space.
    cap = min(2048, CONTEXT // 2)
    for key in ('max_tokens', 'max_completion_tokens'):
        value = body.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > cap:
            body[key] = cap
    encoded = json.dumps(body).encode()
    async def receive():
        return {'type': 'http.request', 'body': encoded, 'more_body': False}
    response = await chat(Request(request.scope, receive), x_user_id='local-agent-apps', x_agent_id=agent_name)
    response.headers['X-Agent-Integration'] = agent_name
    response.headers['X-Agent-Output-Cap'] = str(cap)
    return response


class EmbeddingInput(BaseModel):
    model: str = 'BAAI/bge-small-en-v1.5'
    input: str | list[str]
    encoding_format: str = 'float'


@app.post('/v1/embeddings', dependencies=[Depends(auth)])
async def embeddings(body: EmbeddingInput):
    if body.model != 'BAAI/bge-small-en-v1.5':
        raise HTTPException(400, 'Supported embedding model: BAAI/bge-small-en-v1.5')
    if body.encoding_format != 'float':
        raise HTTPException(400, 'Only float embeddings are supported')
    texts = [body.input] if isinstance(body.input, str) else body.input
    if not 1 <= len(texts) <= 128 or any(not s.strip() or len(s) > 32768 for s in texts):
        raise HTTPException(400, 'Supply 1..128 nonempty texts, each at most 32768 characters')
    async with memory_lock:
        vectors = await asyncio.to_thread(lambda: [memory.embedding_model.embed(s) for s in texts])
    return {'object': 'list', 'model': body.model,
            'data': [{'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(vectors)],
            'usage': {'prompt_tokens': 0, 'total_tokens': 0}}
