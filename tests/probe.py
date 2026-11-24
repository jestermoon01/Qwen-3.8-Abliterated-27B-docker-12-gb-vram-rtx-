"""Quick integration checks, also suitable for the slow long-context profile."""
import asyncio
import os

import httpx


async def main():
    async with httpx.AsyncClient(base_url='http://localhost:8000', timeout=1800) as c:
        r = await c.get('/health')
        r.raise_for_status()
        print('Health:', r.json())
        assert (await c.get('/v1/models')).status_code == 200
        assert (await c.post('/v1/chat/completions', content='broken json')).status_code == 400
        assert (await c.post('/v1/chat/completions', json={
            'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 999999})).status_code == 400
        schema = {'type': 'function', 'function': {
            'name': 'get_weather', 'description': 'Get current weather in a city.',
            'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}
        payload = {'model': 'huihui-qwen3.8-27b', 'max_tokens': 128,
                   'messages': [{'role': 'user', 'content': 'Call get_weather for Casablanca.'}],
                   'tools': [schema], 'tool_choice': {'type': 'function', 'function': {'name': 'get_weather'}}}
        r = await c.post('/v1/chat/completions', json=payload)
        r.raise_for_status()
        result = r.json()
        calls = result['choices'][0]['message'].get('tool_calls')
        assert calls and calls[0]['function']['name'] == 'get_weather', result
        counted = int(r.headers['x-context-tokens-after'])
        used = result['usage']['prompt_tokens']
        assert abs(counted - used) <= 8, (counted, used)
        print('Tool calling + tokenizer/template accounting:', counted, 'counted vs', used, 'used')
        print('Access without API key and invalid inputs: passed')


asyncio.run(main())
