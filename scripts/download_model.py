"""Download pinned weights, verify them, and atomically publish into a Docker volume."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import quote
from urllib.request import urlopen


def verified(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    print(f'Checking SHA256: {path}', flush=True)
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == digest


def download(manifest, destination, attempts=3, base_url='https://huggingface.co'):
    size = int(manifest['size_bytes'])
    digest = manifest['sha256'].lower()
    if size <= 0 or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('Manifest must include a positive size and a SHA256 digest')
    if verified(destination, size, digest):
        print('Model already verified; using cached weights.', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    url = (f"{base_url}/{quote(manifest['repository'], safe='/')}/resolve/"
           f"{quote(manifest['revision'], safe='')}/{quote(manifest['remote_filename'], safe='')}")
    for attempt in range(attempts):
        try:
            print(f'Downloading {size / 1e9:.2f} GB (attempt {attempt + 1}/{attempts})', flush=True)
            # A failed attempt is overwritten; incomplete bytes are never served.
            downloaded = 0
            next_report = 1024 ** 3
            with urlopen(url, timeout=120) as response, partial.open('wb') as output:
                while block := response.read(8 * 1024 ** 2):
                    output.write(block)
                    downloaded += len(block)
                    if downloaded > size:
                        raise ValueError('Download exceeds the manifest size')
                    if downloaded >= next_report:
                        print(f'{downloaded / 1e9:.1f}/{size / 1e9:.1f} GB', flush=True)
                        next_report += 1024 ** 3
            if not verified(partial, size, digest):
                raise ValueError('Model size or SHA256 mismatch')
            partial.replace(destination)
            print('Model verified and ready.', flush=True)
            return
        except (OSError, ValueError) as error:
            if attempt == attempts - 1:
                raise
            print(f'Download failed: {error}; retrying.', flush=True)
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    download(json.loads(args.manifest.read_text(encoding='utf-8')), args.destination)


if __name__ == '__main__':
    main()
