"""Lossless text decoding for the two audited legacy-encoded annotation files."""
from pathlib import Path


def read_annotation(path):
    path=Path(path);raw=path.read_bytes()
    try:
        return raw.decode('utf-8'),'utf-8'
    except UnicodeDecodeError:
        # Audited A1 AF encodes a curly apostrophe in these two corpus files.
        # Never silently replace bytes or infer arbitrary labels on decode error.
        if path.name not in ('5834.txt','4642.txt'):
            raise
        return raw.decode('gb18030'),'gb18030'
