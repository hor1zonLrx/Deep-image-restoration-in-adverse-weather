#!/usr/bin/env python3
"""Check curated URL reachability; this does not verify identity or downloads."""
import argparse
import concurrent.futures
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def check(url):
    # Only fetch response headers. Do not download papers or datasets.
    for method in ('HEAD', 'GET'):
        try:
            request = Request(url, method=method, headers={'User-Agent': 'SurveyLinkAudit/1.0'})
            with urlopen(request, timeout=20) as response:
                return {'url': url, 'result': 'reachable', 'http_status': response.status,
                        'final_url': response.url}
        except HTTPError as error:
            if method == 'HEAD':
                continue
            return {'url': url, 'result': 'broken' if error.code in (404, 410) else 'manual_review',
                    'http_status': error.code}
        except (URLError, TimeoutError, OSError) as error:
            return {'url': url, 'result': 'manual_review', 'error': str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Save a JSON reachability report.')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'data/paper_links.json').read_text())
    urls = sorted({p[field] for p in catalog['papers'].values()
                   for field in ('paper', 'code', 'dataset') if p[field]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(check, urls))
    report = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'summary': dict(Counter(r['result'] for r in results)), 'results': results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']))
    for result in results:
        if result['result'] != 'reachable':
            print(json.dumps(result))
    raise SystemExit(1 if report['summary'].get('broken') else 0)


if __name__ == '__main__':
    main()
