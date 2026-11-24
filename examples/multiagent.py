"""python examples/multiagent.py -- stdlib only, no paid API services."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import Request, urlopen

settings = dict(line.split('=', 1) for line in
                (Path(__file__).resolve().parents[1] / '.env').read_text().splitlines()
                if line and not line.startswith('#') and '=' in line)


def ask(agent, prompt):
    request = Request(
        f"http://localhost:{settings.get('API_PORT', '8001')}/v1/chat/completions",
        data=json.dumps({'model': settings.get('MODEL_ALIAS', 'huihui-qwen3.8-27b'), 'max_tokens': 512,
                         'messages': [{'role': 'user', 'content': prompt}]}).encode(),
        headers={'Content-Type': 'application/json',
                 'X-User-ID': 'my-project', 'X-Agent-ID': agent})
    with urlopen(request, timeout=1800) as response:
        return agent, json.load(response)['choices'][0]['message']['content']


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ask, 'planner', 'Make a short plan for a local notes app.'),
                   pool.submit(ask, 'reviewer', 'List three pitfalls in a local notes app.')]
        for future in futures:
            agent, answer = future.result()
            print(f'[{agent}]\n{answer}\n')
