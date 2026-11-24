import asyncio
import unittest

from fastapi import HTTPException
from gateway.context import compress


async def count(messages, options):
    return sum(len(m.get('content', '')) + 4 for m in messages) + len(str(options))


async def summarize(messages, limit):
    assert await count(messages, {}) < 4000
    return 'Earlier decisions retained.'


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_history_unchanged(self):
        messages = [{'role': 'user', 'content': 'Hello'}]
        result, stats = await compress(messages, {}, 4000, count, summarize)
        self.assertEqual(result, messages)
        self.assertFalse(stats['compressed'])

    async def test_large_multilingual_history_and_tool_pairs(self):
        latest = [
            {'role': 'user', 'content': 'Continue'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_1'}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'Tool result'},
        ]
        messages = [{'role': 'system', 'content': 'Keep this instruction'}]
        messages += [{'role': 'user', 'content': '你好 مرحبا ' * 3000},
                     {'role': 'assistant', 'content': 'Previous answer'}] + latest
        result, stats = await compress(messages, {}, 4000, count, summarize, summary_limit=100)
        self.assertTrue(stats['compressed'])
        self.assertLessEqual(await count(result, {}), 4000)
        self.assertEqual(result[0], messages[0])
        self.assertEqual(result[-3:], latest)

    async def test_oversized_latest_document_is_compressed(self):
        result, stats = await compress([{'role': 'user', 'content': 'x' * 5000 + ' FINAL QUESTION'}],
                                       {}, 4000, count, summarize, summary_limit=100)
        self.assertTrue(stats['compressed'])
        self.assertTrue(result[-1]['content'].endswith(' FINAL QUESTION'))
        self.assertLessEqual(await count(result, {}), 4000)

    async def test_oversized_tool_output_summarized_with_exchange_intact(self):
        messages = [{'role': 'user', 'content': 'Read this'},
                    {'role': 'assistant', 'tool_calls': [{'id': '1'}]},
                    {'role': 'tool', 'tool_call_id': '1', 'content': 'x' * 5000}]
        result, stats = await compress(messages, {}, 4000, count, summarize)
        self.assertEqual(result[:2], messages[:2])
        self.assertEqual(result[2]['tool_call_id'], '1')
        self.assertIn('Tool output summary', result[2]['content'])
        self.assertEqual(stats['tool_outputs_compressed'], 1)
        self.assertLessEqual(await count(result, {}), 4000)
        self.assertEqual(messages[2]['content'], 'x' * 5000)

    async def test_oversized_user_with_tool_exchange_rejected(self):
        with self.assertRaises(HTTPException) as error:
            await compress([{'role': 'user', 'content': 'x' * 5000},
                            {'role': 'assistant', 'tool_calls': [{'id': '1'}]},
                            {'role': 'tool', 'tool_call_id': '1', 'content': 'short'}],
                           {}, 4000, count, summarize)
        self.assertEqual(error.exception.status_code, 413)

    async def test_tool_schema_consumes_budget(self):
        with self.assertRaises(HTTPException):
            await compress([{'role': 'user', 'content': 'Hi'}], {'tools': 'x' * 5000},
                           4000, count, summarize)


if __name__ == '__main__':
    unittest.main()
