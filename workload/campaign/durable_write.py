"""Small durable replacement for accounting and progress metadata."""
from pathlib import Path
import argparse
import json
import os
import tempfile


def write_text(path, text):
    """Preserve the last complete value until write, close and fsync all succeed.

    Failed temporary files remain for inspection. A directory-fsync error after
    replace is surfaced as an uncertain publication, never silently counted success.
    """
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    with os.fdopen(descriptor, 'w') as stream:
        mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
        os.fchmod(stream.fileno(), mode)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def read_credit_pair(credit_path, status_path):
    values, existing = [], []
    for path, is_status in ((Path(credit_path), False), (Path(status_path), True)):
        try:
            raw = path.read_text()
        except FileNotFoundError:
            continue
        existing.append(str(path))
        try:
            if is_status:
                value = json.loads(raw).get('credited_seconds')
                if type(value) is not int or value < 0:
                    raise ValueError('invalid credited_seconds')
            else:
                text = raw.strip()
                if not text or not text.isascii() or not text.isdecimal():
                    raise ValueError('invalid credit counter')
                value = int(text)
            values.append(value)
        except (ValueError, TypeError, AttributeError):
            continue
    if values:
        return max(values)
    if existing:
        raise ValueError('No valid persisted credit remains; preserve and recover the accounting files')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('value')
    args = parser.parse_args()
    write_text(args.path, args.value + '\n')


if __name__ == '__main__':
    main()
