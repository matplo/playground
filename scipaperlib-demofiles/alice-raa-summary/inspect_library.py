"""Inventory original HEPData YAML without changing the SciPaperlib library."""
from pathlib import Path
import hashlib, json, re
import yaml

HERE = Path(__file__).resolve().parent
LIBRARY = Path.home()/'.scipaperlib'/'default_scipaperlib'

def main():
    manifests = json.loads((HERE / 'source_manifest.json').read_text())
    catalog, problems = [], []
    for m in manifests:
        for f in m['files']:
            if not f.get('table') or not f.get('path', '').endswith(('.yaml', '.yml')):
                continue
            path = LIBRARY / f['path']
            if not path.exists():
                problems.append({'record': m['record'], 'table': f['table'], 'problem': 'missing file'})
                continue
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            if actual != f['sha256']:
                raise ValueError(f'Checksum mismatch: {path}')
            data = yaml.load(raw, Loader=yaml.BaseLoader)
            if not isinstance(data, dict) or 'dependent_variables' not in data:
                problems.append({'record': m['record'], 'table': f['table'], 'problem': 'not HEPData table'})
                continue
            xs = data.get('independent_variables', [])
            for j, dep in enumerate(data['dependent_variables']):
                head = dep['header']['name']
                desc = f.get('description', '')
                compact = re.sub(r'[^a-z0-9]', '', head.lower())
                if not ('raa' in compact or 'rpb' in compact or 'qp' in compact or 'roo' in compact or 'rpo' in compact or 'nuclear modification' in desc.lower() or m['record'] in ['68361', '150694', '86210', '73941']):
                    continue
                catalog.append({'id': f"{m['record']}:{f['table']}:{j}", 'record': m['record'], 'dataset_id': m['dataset_id'], 'table': f['table'], 'column': j, 'header': dep['header'], 'qualifiers': dep.get('qualifiers', []), 'xheaders': [x['header'] for x in xs], 'xfirst': [x['values'][:2] for x in xs], 'n': len(dep['values']), 'first': dep['values'][:1], 'description': desc, 'location': f.get('location'), 'file': f, 'paper_ids': m['paper_ids'], 'arxiv_id': m.get('arxiv_id'), 'title': m['title'], 'version': m['version'], 'hepdata_doi': m.get('hepdata_doi')})
    (HERE / 'table_catalog.json').write_text(json.dumps(catalog, indent=2))
    (HERE / 'inventory_issues.json').write_text(json.dumps(problems, indent=2))
    for c in catalog:
        q = '; '.join(f"{v['name']}={v['value']}" for v in c['qualifiers'])
        print(c['id'], c['header']['name'], c['xheaders'], q, f"n={c['n']}")
    print(f'CATALOG: {len(catalog)} columns; ISSUES: {len(problems)}')

if __name__ == '__main__':
    main()
