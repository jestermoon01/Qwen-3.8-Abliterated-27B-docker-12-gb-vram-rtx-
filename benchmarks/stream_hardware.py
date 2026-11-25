"""Measure actual streamed input/output lengths, TTFT and observed decode rate."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
p.add_argument('--weights', default='huihui-27b-iq3_xxs.gguf')
p.add_argument('--url', default='http://127.0.0.1:8001')
p.add_argument('--lines', type=int, default=1500)
p.add_argument('--tokens', type=int, default=4096)
p.add_argument('--idle-timeout', type=int, default=180)
p.add_argument('--prefill-deadline', type=int, default=180)
p.add_argument('--deadline', type=int, default=600)
a = p.parse_args()
rows = '\n'.join(f'Record {i}: north warehouse inventory verified; shipment pending review.' for i in range(a.lines))
prompt = ('Reference ledger follows.\nSTART-CODE: cedar-731\n' + rows +
          '\nEND-CODE: maple-492\nFirst state START-CODE and END-CODE exactly. Then write a detailed numbered '
          'manual with 200 sections on durable task queues, retries, idempotency, crash recovery, monitoring, '
          'backpressure, and security. Each section must contain at least 80 words with a concrete example. '
          'Continue until all 200 sections are finished; do not conclude early.')
body = {'model': 'huihui-qwen3.8-27b', 'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': a.tokens, 'temperature': 0, 'seed': 42, 'stream': True,
        'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'enable_thinking': False}}
path = Path(__file__).parent / 'results' / ('stream-' + a.label + '.json')
path.parent.mkdir(parents=True, exist_ok=True)
r = {'label': a.label, 'url': a.url, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
     'weights_file': a.weights, 'request': body, 'status': 'running', 'content': '', 'usage': None}
start = time.perf_counter()
first = last = None
saved = 0
def save():
    r['elapsed_seconds'] = time.perf_counter() - start
    r['ttft_seconds'] = first
    r['last_content_seconds'] = last
    path.write_text(json.dumps(r, indent=2), encoding='utf-8')
save()
try:
    with urlopen(Request(a.url + '/v1/chat/completions', json.dumps(body).encode(),
                         {'Content-Type': 'application/json'}), timeout=a.idle_timeout) as response:
        r['headers'] = dict(response.headers)
        done = False
        for raw in response:
            now = time.perf_counter() - start
            if first is None and now > a.prefill_deadline:
                raise TimeoutError('No first content token within prefill deadline')
            if now > a.deadline:
                raise TimeoutError('Total benchmark deadline exceeded')
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError('SSE error: ' + json.dumps(event['error']))
            if event.get('usage'):
                r['usage'] = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                text = delta.get('content') or delta.get('reasoning_content') or ''
                if text:
                    if first is None:
                        first = now
                        print(f'{a.label}: first token at {first:.2f}s', flush=True)
                    last = now
                    r['content'] += text
                if choice.get('finish_reason'):
                    r['finish_reason'] = choice['finish_reason']
            if now - saved > 10:
                saved = now
                save()
        if not done:
            raise RuntimeError('SSE stream ended without [DONE]')
        if not r['usage']:
            raise RuntimeError('Stream completed without requested token usage')
    r['status'] = 'complete'
except Exception as exc:
    r['status'] = 'failed'
    r['error'] = repr(exc)
finally:
    if r['usage'] and first is not None and last is not None and last > first:
        r['observed_decode_tps'] = (r['usage']['completion_tokens'] - 1) / (last - first)
    r['boundary_recall'] = all(x in r['content'] for x in ['cedar-731', 'maple-492'])
    save()
    print(json.dumps({k: v for k, v in r.items() if k not in ('request', 'content', 'headers')}, indent=2), flush=True)
raise SystemExit(0 if r['status'] == 'complete' else 1)
