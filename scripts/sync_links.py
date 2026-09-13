#!/usr/bin/env python3
"""Synchronize curated paper links without replacing the survey's prose or numbers."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link(url, label):
    return f"[{label}]({url})" if url else "—"


def render(text, catalog, chinese=False):
    rows = {(r['section'], r['label']): r['key'] for r in catalog['rows']}
    section = None
    seen = set()
    output = []
    for line in text.splitlines():
        heading = re.match(r'^## (\d+\.\d+) ', line)
        if heading:
            section = heading[1]
        if line.startswith('# ') or (line.startswith('## ') and not heading):
            section = None
        if section == '2.3' and line.startswith('- '):
            # Each name owns its links, including the shared MIRNet/MIRNetv2 bullet.
            for row in catalog['rows']:
                if row['section'] != section:
                    continue
                token = '**' + row['label'] + '**'
                if token not in line:
                    continue
                paper = catalog['papers'][row['key']]
                pattern = re.escape(token) + r'(?: \[Paper\]\([^)]*\))?(?: \[(?:Code|Project)\]\([^)]*\))?'
                code_label = 'Project' if paper.get('code_status') == 'announced' else 'Code'
                suffix = ''.join(' ' + link(paper[field], label) for field, label in [('paper', 'Paper'), ('code', code_label)] if paper[field])
                line = re.sub(pattern, lambda _: token + suffix, line)
                seen.add((section, row['label']))
        if section and line.startswith('|') and section != '2.3':
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            expected = [r for r in catalog['rows'] if r['section'] == section]
            if expected:
                is_dataset = section.startswith('3.')
                base_count = 6 if is_dataset else (3 if section.startswith('2.') or section == '1.5' else 4)
                if cells[0] in ('Method', '方法', 'Dataset', '数据集'):
                    cells = cells[:base_count] + ['Paper', 'Dataset' if is_dataset else 'Code']
                elif re.fullmatch(r':?-+:?', cells[0]):
                    cells = cells[:base_count] + ['---', '---']
                elif (section, cells[0]) in rows:
                    seen.add((section, cells[0]))
                    paper = catalog['papers'][rows[(section, cells[0])]]
                    resource = 'dataset' if is_dataset else 'code'
                    label = 'Dataset' if is_dataset else 'Code'
                    if resource == 'code' and paper.get('code_status') == 'announced':
                        label = 'Project'
                    cells = cells[:base_count] + [link(paper['paper'], 'Paper'), link(paper[resource], label)]
                else:
                    raise ValueError(f'Unmapped row in section {section}: {cells[0]}')
                line = '| ' + ' | '.join(cells) + ' |'
        output.append(line)
    missing = set(rows) - seen
    if missing:
        raise ValueError(f'Catalog rows missing from README: {sorted(missing)}')
    return '\n'.join(output) + '\n'


def validate(catalog):
    seen = set()
    for row in catalog['rows']:
        identity = row['section'], row['label']
        if identity in seen:
            raise ValueError(f'Duplicate row: {identity}')
        seen.add(identity)
        if row['key'] not in catalog['papers']:
            raise ValueError(f'Missing citation key: {row}')
    for key, paper in catalog['papers'].items():
        for field in ('paper', 'code', 'dataset'):
            url = paper[field]
            if url and not re.fullmatch(r'https?://[^\s<>\[\]()]+', url):
                raise ValueError(f'Invalid {field} URL for {key}: {url}')
        if paper['paper'] and not paper['sources']:
            raise ValueError(f'No provenance for {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if README links are out of sync; do not write files.')
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'data/paper_links.json').read_text())
    validate(catalog)
    changed = []
    rendered = []
    # Validate both languages before writing either file.
    for name in ('README.md', 'README_ZH.md'):
        path = ROOT / name
        original = path.read_text()
        updated = render(original, catalog, name.endswith('_ZH.md'))
        rendered.append((path, original, updated))
        if updated != original:
            changed.append(name)
    if args.check and changed:
        raise SystemExit('Out of sync: ' + ', '.join(changed))
    if not args.check:
        for path, original, updated in rendered:
            if original != updated:
                path.write_text(updated)
    papers = list(catalog['papers'].values())
    print(f"{len(catalog['rows'])} rows; {len(papers)} papers; "
          f"{sum(bool(p['paper']) for p in papers)} paper links; "
          f"{sum(bool(p['code']) for p in papers)} code/project links.")


if __name__ == '__main__':
    main()
