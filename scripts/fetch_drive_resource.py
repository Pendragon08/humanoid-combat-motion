"""Download an explicitly selected public Drive resource; never accepts HTML as weights."""
import argparse
import hashlib
import json
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
import requests


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'form' and a.get('id') == 'download-form':
            self.action = a.get('action')
        if tag == 'input' and a.get('name'):
            self.fields[a['name']] = a.get('value', '')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-gib', type=float, default=4)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    session = requests.Session()
    url = 'https://drive.google.com/uc?' + urlencode({'export':'download','id':args.id})
    record = {'id':args.id, 'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'status':'started'}
    try:
        for _ in range(3):
            response = session.get(url, stream=True, timeout=(20,60))
            response.raise_for_status()
            if 'text/html' not in response.headers.get('Content-Type',''):
                break
            form = DownloadForm()
            form.feed(response.text)
            response.close()
            if not form.action or not form.action.startswith('https://drive.usercontent.google.com/'):
                raise RuntimeError('Drive returned HTML without an authorized public download form (quota/access/link issue)')
            url = form.action + '?' + urlencode(form.fields)
        else:
            raise RuntimeError('Download confirmation did not resolve')
        limit = int(args.max_gib * 1024**3)
        if int(response.headers.get('Content-Length',0)) > limit:
            raise RuntimeError('Content exceeds size budget')
        partial = args.output.with_suffix(args.output.suffix+'.partial')
        size, digest = 0, hashlib.sha256()
        with partial.open('wb') as f:
            for chunk in response.iter_content(1024**2):
                size += len(chunk)
                if size > limit:
                    raise RuntimeError('Download exceeded size budget; partial retained')
                digest.update(chunk)
                f.write(chunk)
        partial.rename(args.output)
        record.update(status='downloaded', bytes=size, sha256=digest.hexdigest())
    except Exception as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        args.output.with_suffix(args.output.suffix+'.download.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
