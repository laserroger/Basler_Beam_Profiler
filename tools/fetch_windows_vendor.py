"""Fetch checksum-pinned Windows build inputs; end users receive them offline."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def fetch():
    root = Path(__file__).resolve().parents[1]
    source = root / 'external_libs/flir/windows-x64'
    manifest = json.loads((source / 'manifest.json').read_text())
    target = root / '.vendor/windows'
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest['files'].items():
        path = target / name
        if not path.is_file():
            temporary = path.with_suffix(path.suffix + '.download')
            with urllib.request.urlopen(manifest['base_url'] + name, timeout=120) as response:
                with temporary.open('wb') as output:
                    shutil.copyfileobj(response, output)
            temporary.replace(path)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise RuntimeError(f'Windows vendor checksum mismatch: {name}')
        print(f'Verified {name}')
    license_dir = target / 'licenses'
    license_dir.mkdir(exist_ok=True)
    for pattern in ('*.txt', '*.pdf'):
        for path in source.glob(pattern):
            shutil.copy2(path, license_dir / path.name)


if __name__ == '__main__':
    fetch()
