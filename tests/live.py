"""Live inference, concurrency, compression, streaming, and scoped Mem0 checks."""
import asyncio
import os
import uuid

import httpx


async def main():
    headers = {}
    async with httpx.AsyncClient(base_url='http://localhost:8000', headers=headers, timeout=1800) as c:
        response = await c.get('/health')
        response.raise_for_status()
        print('Health:', response.json())
        response = await c.get('/v1/models')
        response.raise_for_status()
        print('Models:', [m['id'] for m in response.json()['data']])
        async def ask(text, **kwargs):
            result = await c.post('/v1/chat/completions', json={
                'model': 'huihui-qwen3.8-27b', 'messages': [{'role': 'user', 'content': text}],
                'max_tokens': 48, **kwargs})
            result.raise_for_status()
            return result
        answers = await asyncio.gather(ask('Reply with the word ONE.'), ask('Reply with the word TWO.'))
        print('Concurrent replies:', [r.json()['choices'][0]['message']['content'] for r in answers])
        async with c.stream('POST', '/v1/chat/completions', json={
            'model': 'huihui-qwen3.8-27b', 'messages': [{'role': 'user', 'content': 'Say hello.'}],
            'max_tokens': 32, 'stream': True}) as r:
            r.raise_for_status()
            stream = ''.join([line async for line in r.aiter_lines()])
            assert '[DONE]' in stream, stream
        print('Streaming: passed')
        namespace = 'live-' + uuid.uuid4().hex
        mem_headers = {'X-User-ID': namespace, 'X-Agent-ID': 'planner'}
        try:
            r = await c.post('/memory', headers=mem_headers,
                             json={'text': 'The project codename is saffron-castle.'})
            r.raise_for_status()
            r = await c.get('/memory/search', headers=mem_headers, params={'q': 'project codename'})
            r.raise_for_status()
            assert any('saffron-castle' in m['memory'] for m in r.json()['results']), r.text
            r = await c.get('/memory/search', headers={'X-User-ID': namespace, 'X-Agent-ID': 'other'},
                            params={'q': 'project codename'})
            r.raise_for_status()
            assert not r.json()['results'], r.text
            print('Memory add/search + agent isolation: passed')
            r = await c.post('/v1/chat/completions', headers=mem_headers, json={
                'model': 'huihui-qwen3.8-27b', 'max_tokens': 32,
                'messages': [{'role': 'user', 'content': 'Remember that my preferred editor is Helix. Reply OK.'}]})
            r.raise_for_status()
            assert r.headers['x-memory-status'] in ('retrieved', 'no-matches'), dict(r.headers)
            for _ in range(30):
                r = await c.get('/memory/search', headers=mem_headers, params={'q': 'preferred editor'})
                r.raise_for_status()
                if any('Helix' in m['memory'] for m in r.json()['results']):
                    break
                await asyncio.sleep(1)
            else:
                raise AssertionError('Automatic memory write did not reach Qdrant')
            print('Automatic memory outbox: passed')
        finally:
            r = await c.delete('/memory', headers=mem_headers)
            r.raise_for_status()
        # Enough history to exceed the default 16K slot; compression keeps latest turn intact.
        r = await c.post('/v1/chat/completions', json={
            'model': 'huihui-qwen3.8-27b', 'max_tokens': 48,
            'messages': [
                {'role': 'system', 'content': 'Answer concisely.'},
                {'role': 'user', 'content': 'The project color is blue. ' * (int(os.getenv('CONTEXT_PER_SLOT', '16384')) // 5 + 100)},
                {'role': 'assistant', 'content': 'Understood.'},
                {'role': 'user', 'content': 'What project color was mentioned?'}]})
        r.raise_for_status()
        assert r.headers['x-context-compressed'] == 'true', dict(r.headers)
        assert 'blue' in r.json()['choices'][0]['message']['content'].lower(), r.text
        print('Compression:', r.headers['x-context-tokens-before'], '->', r.headers['x-context-tokens-after'])
        print('ALL LIVE CHECKS PASSED')


asyncio.run(main())
