"""Atomic JSON updates shared by the renderer and the QQ plugin."""
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import time


def write_json(path, value, *, allow_nan=True):
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=allow_nan)
    stream = NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp',
                                delete=False, mode='w', encoding='utf-8')
    temp = Path(stream.name)
    try:
        with stream:
            stream.write(payload)
        deadline = time.monotonic() + 3.0
        delay = 0.02
        while True:
            try:
                temp.replace(path)
                return
            except OSError as exc:
                # Windows readers can briefly deny delete sharing, including on rename.
                remaining = deadline - time.monotonic()
                if getattr(exc, 'winerror', None) not in (5, 32, 33) or remaining <= 0:
                    raise
                time.sleep(min(delay, remaining))
                delay = min(2 * delay, 0.2)
    finally:
        temp.unlink(missing_ok=True)
