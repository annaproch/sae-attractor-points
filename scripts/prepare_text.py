"""Download a small, reproducible Wikipedia-text sample for activation probes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/natural-text-001'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = 'Salesforce/wikitext'
    config = 'wikitext-2-raw-v1'
    candidates = []
    for offset in range(0, 1000, 100):
        query = urlencode(dict(dataset=dataset, config=config, split='test',
                               offset=offset, length=100))
        request = Request('https://datasets-server.huggingface.co/rows?' + query,
                          headers={'User-Agent': 'sae-natural-text-pilot/1.0'})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=30) as response:
                    rows = json.load(response)['rows']
                break
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise
        for row in rows:
            text = row['row']['text'].strip()
            if len(text.split()) >= 80 and not text.startswith('='):
                candidates.append({'source_row': row['row_idx'], 'text': text})
    if len(candidates) < 100:
        raise ValueError(f'Only {len(candidates)} eligible paragraphs')
    selected = random.Random(42).sample(candidates, 100)
    payload = ''.join(json.dumps({'id': i, **row}, ensure_ascii=False) + '\n'
                      for i, row in enumerate(selected))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'texts.jsonl').write_text(payload, encoding='utf-8')
    manifest = dict(dataset=dataset, config=config, split='test', seed=42,
                    source_rows_scanned=1000, minimum_words=80, paragraphs=100,
                    sha256=hashlib.sha256(payload.encode('utf-8')).hexdigest(),
                    source='https://huggingface.co/datasets/Salesforce/wikitext',
                    note='Wikipedia-derived WikiText paragraphs; no tokenization or model forward pass yet.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved 100 paragraphs to {args.output}/texts.jsonl')


if __name__ == '__main__':
    main()
