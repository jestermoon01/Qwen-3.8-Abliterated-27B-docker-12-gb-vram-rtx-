"""Run 'store', recreate containers, then run 'verify' to check durable memory."""
import os
import sys

import httpx

headers = {'X-User-ID': 'deployment-persistence-check', 'X-Agent-ID': 'test'}
with httpx.Client(base_url='http://localhost:8000', headers=headers, timeout=120) as c:
    if sys.argv[1] == 'store':
        r = c.post('/memory', json={'text': 'Persistence check marker: copper-moon-731.'})
        r.raise_for_status()
        print('Persistence marker stored')
    elif sys.argv[1] == 'verify':
        r = c.get('/memory/search', params={'q': 'Persistence check marker'})
        r.raise_for_status()
        assert any('copper-moon-731' in m['memory'] for m in r.json()['results']), r.text
        c.delete('/memory').raise_for_status()
        print('Memory survived container recreation; test namespace removed')
    else:
        raise ValueError('Use store or verify')
