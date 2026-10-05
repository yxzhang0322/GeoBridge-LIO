from pathlib import Path
import hashlib, json

def new_dir(path):
    out = Path(path).resolve()
    out.mkdir(parents=True, exist_ok=False)
    return out

def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
