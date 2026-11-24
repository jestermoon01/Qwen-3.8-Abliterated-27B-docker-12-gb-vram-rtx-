import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest


spec = importlib.util.spec_from_file_location(
    'download_model', Path(__file__).resolve().parents[1] / 'scripts' / 'download_model.py')
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = b'test GGUF payload'

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(cls.payload)

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def manifest(self):
        return dict(repository='test/model', revision='pinned', remote_filename='model.gguf',
                    size_bytes=len(self.payload), sha256=hashlib.sha256(self.payload).hexdigest())

    def test_download_then_cached_offline(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / 'model.gguf'
            downloader.download(self.manifest(), dest, attempts=1, base_url=self.url)
            self.assertEqual(dest.read_bytes(), self.payload)
            self.assertFalse(dest.with_suffix('.gguf.part').exists())
            downloader.download(self.manifest(), dest, attempts=1, base_url='http://127.0.0.1:1')

    def test_bad_checksum_cannot_replace_existing_model(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / 'model.gguf'
            dest.write_bytes(b'previous model')
            manifest = self.manifest()
            manifest['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                downloader.download(manifest, dest, attempts=1, base_url=self.url)
            self.assertEqual(dest.read_bytes(), b'previous model')

    def test_corrupt_cache_is_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / 'model.gguf'
            dest.write_bytes(b'x' * len(self.payload))
            downloader.download(self.manifest(), dest, attempts=1, base_url=self.url)
            self.assertEqual(dest.read_bytes(), self.payload)

    def test_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            dest = Path(root) / 'model.gguf'
            manifest = self.manifest()
            manifest['size_bytes'] += 1
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                downloader.download(manifest, dest, attempts=1, base_url=self.url)
            self.assertFalse(dest.exists())
