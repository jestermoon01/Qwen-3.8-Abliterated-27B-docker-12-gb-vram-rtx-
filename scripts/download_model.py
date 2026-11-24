"""Download pinned weights, verify them, and atomically publish into a Docker volume."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import re
import time
import warnings
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import Request, HTTPRedirectHandler, build_opener


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        # Hugging Face redirects downloads to a CDN. Do not send its token there.
        if redirected is not None and redirected.host != req.host:
            redirected.remove_header('Authorization')
        return redirected


def open_model(url, token):
    headers = {'Authorization': 'Bearer ' + token} if token else {}
    return build_opener(SafeRedirect()).open(Request(url, headers=headers), timeout=120)


def verified(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    print(f'Checking SHA256: {path}', flush=True)
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == digest


def download(manifest, destination, attempts=3, base_url='https://huggingface.co', interactive=False):
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
    token = ''
    for attempt in range(attempts):
        try:
            print(f'Downloading {size / 1e9:.2f} GB (attempt {attempt + 1}/{attempts})', flush=True)
            # A failed attempt is overwritten; incomplete bytes are never served.
            downloaded = 0
            next_report = 1024 ** 3
            while True:
                try:
                    response = open_model(url, token)
                    break
                except HTTPError as error:
                    if error.code not in (401, 403):
                        raise
                    if not interactive:
                        raise PermissionError('Hugging Face requires access. Run scripts/setup.ps1 '
                                              'or sh scripts/setup.sh to enter a token securely.') from None
                    print('Hugging Face requires authentication or model access approval.\n'
                          'Accept any model access terms on huggingface.co, then enter a read token.\n'
                          'The token stays in this process only; it is not saved or added to an image.', flush=True)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('error', getpass.GetPassWarning)
                            token = getpass.getpass('Hugging Face token (hidden; blank cancels): ').strip()
                    except getpass.GetPassWarning:
                        raise PermissionError('Run setup in an interactive terminal for hidden token entry.') from None
                    if not token:
                        raise PermissionError('Token entry cancelled') from None
            with response, partial.open('wb') as output:
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
        except PermissionError:
            raise
        except (OSError, ValueError) as error:
            if attempt == attempts - 1:
                raise
            print(f'Download failed: {error}; retrying.', flush=True)
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--interactive', action='store_true', help='Prompt if Hugging Face requires a token')
    args = parser.parse_args()
    download(json.loads(args.manifest.read_text(encoding='utf-8')), args.destination,
             interactive=args.interactive)


if __name__ == '__main__':
    main()
