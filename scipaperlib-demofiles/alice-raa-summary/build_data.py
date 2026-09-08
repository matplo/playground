"""Read-only extraction from the default SciPaperlib cache; verify every SHA256."""
from pathlib import Path
import csv, hashlib, json, math, re
import yaml
from selection import SELECTIONS, PRESETS
HERE=Path(__file__).resolve().parent
LIBRARY=Path.home()/'.scipaperlib'/'default_scipaperlib'

def magnitude(value,y):
    s=str(value).strip()
    return abs(float(s.rstrip('%'))) * (abs(y)/100 if s.endswith('%') else 1)

def error_kind(label):
    s=label.lower()
    if 'stat' in s and 'syst' not in s:return 'stat'
    if any(w in s for w in ['norm','lumi','tpa','taa','t_{','branching']):return 'global'
    if 'sys' in s:return 'sys'
    raise ValueError(f'Unclassified error: {label!r}')

def main():
    catalog={x['id']:x for x in json.loads((HERE/'table_catalog.json').read_text())}
    rawdir=HERE/'data'/'raw';rawdir.mkdir(parents=True,exist_ok=True)
    curves=[];omitted=[];audit=[]
    for s in SELECTIONS:
        ident=f"{s['record']}:{s['table']}:{s['column']}"
        a=catalog[ident]; f=a['file']; payload=(LIBRARY/f['path']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==f['sha256'],ident
        rawname=f"{s['record']}_{f['sha256'][:16]}.yaml"
        (rawdir/rawname).write_bytes(payload)
        raw=yaml.load(payload,Loader=yaml.BaseLoader)
        dep=raw['dependent_variables'][s['column']]
        assert len(raw['independent_variables'])==1,ident
        independent=raw['independent_variables'][0]
        assert len(independent['values'])==len(dep['values']),ident
        points=[];labels=set()
        for row,(xv,yv) in enumerate(zip(independent['values'],dep['values'])):
            try:y=float(yv['value'])
            except (ValueError,TypeError):
                omitted.append(dict(key=s['key'],row=row,reason='Non-numeric deposited value',original=yv));continue
            lo=float(xv['low']) if 'low' in xv else None
            hi=float(xv['high']) if 'high' in xv else None
            x=float(xv['value']) if 'value' in xv else (lo+hi)/2
            assert x>0 and math.isfinite(y),(ident,row)
            if lo is not None:assert lo<hi and lo<=x<=hi,(ident,row)
            errs=[]
            for e in yv.get('errors',[]):
                lab=e.get('label','UNLABELLED');kind=error_kind(lab);labels.add(lab)
                # In these quarkonium tables the correlated terms are pT-global
                # scale errors (paper Figure 6 / Figure 5), not pointwise errors.
                if s['record']=='69212' and lab=='sys,corr':kind='global'
                if s['record']=='100166' and lab in ['sys, correl','sys, common correl']:kind='global'
                if 'symerror' in e:em=ep=magnitude(e['symerror'],y)
                else:
                    em=magnitude(e['asymerror']['minus'],y);ep=magnitude(e['asymerror']['plus'],y)
                assert math.isfinite(em) and math.isfinite(ep)
                errs.append(dict(label=lab,kind=kind,minus=em,plus=ep,original=e))
            # Published total systematic takes precedence over named subcomponents.
            syst=[e for e in errs if e['kind']=='sys']
            totals=[e for e in syst if e['label'].lower().strip() in ['sys','syst','syst.','syst. uncertainty','full syst.'] or 'systematic uncertainty' in e['label'].lower()]
            chosen=totals[:1] if totals else syst
            point=dict(x=x,xlow=lo,xhigh=hi,y=y,source_row=row,errors=errs,
                stat_minus=math.sqrt(sum(e['minus']**2 for e in errs if e['kind']=='stat')),
                stat_plus=math.sqrt(sum(e['plus']**2 for e in errs if e['kind']=='stat')),
                sys_minus=math.sqrt(sum(e['minus']**2 for e in chosen)),
                sys_plus=math.sqrt(sum(e['plus']**2 for e in chosen)),
                sys_display_labels=[e['label'] for e in chosen],original_x=xv,original_y=yv)
            points.append(point)
        assert len(points)>1,ident
        qualifiers=dep.get('qualifiers',[])
        acceptance=s['acceptance'] or '; '.join(f"{q['name']}: {q['value']} {q.get('units','')}" for q in qualifiers if any(w in q['name'].lower() for w in ['yrap','eta','iso','lead','radius']))
        curve={**s,'acceptance':acceptance or 'See deposited qualifiers and source paper.',
            'label':f"{s['name']} · {s['system']} {s['energy_TeV']:g} TeV · {s['centrality']}",
            'points':points,'source':{k:a[k] for k in ['dataset_id','table','column','header','description','location','arxiv_id','title','version','hepdata_doi','paper_ids']},
            'qualifiers':qualifiers,'xheader_original':independent['header'],
            'raw_file':f'data/raw/{rawname}','sha256':f['sha256'],'table_doi':f['doi'],
            'source_url':'https://doi.org/'+f['doi'],'error_labels':sorted(labels)}
        curves.append(curve)
        audit.append(dict(key=s['key'],table=ident,points=len(points),sha256=f['sha256'],error_labels=sorted(labels)))
    result=dict(schema_version=1,library='default',created='2026-09-07',
        uncertainty_policy='Statistical bars; point-dependent systematic boxes. Prefer supplied systematic total; otherwise quadrature of named components, for visualisation only. Separately named normalization, luminosity, nuclear-overlap and branching-ratio errors are retained but not drawn. No covariance matrix inferred.',
        x_policy='Deposited pT value, or arithmetic bin midpoint if only edges are supplied. Original bins and strings retained. No rebinning or interpolation. For centre-only tables, systematic boxes have decorative width.',
        curves=curves,presets=PRESETS,omitted_rows=omitted)
    (HERE/'data'/'curves.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (HERE/'data'/'validation.json').write_text(json.dumps(dict(curves=len(curves),points=sum(len(c['points']) for c in curves),unique_files=len({c['sha256'] for c in curves}),checks='All input checksums, bin boundaries, numeric values, error labels and row lengths passed.',audit=audit,omitted_rows=omitted),indent=2)+'\n')
    with (HERE/'data'/'points.csv').open('w',newline='') as out:
        fields=['key','name','system','energy_TeV','centrality','observable','x','xlow','xhigh','y','stat_minus','stat_plus','sys_minus','sys_plus','source_row','table_doi']
        w=csv.DictWriter(out,fieldnames=fields);w.writeheader()
        for c in curves:
            for p in c['points']:w.writerow({k:(p.get(k) if k in p else c[k]) for k in fields})
    print(f"Built {len(curves)} curves, {sum(len(c['points']) for c in curves)} points; {len(omitted)} nonnumeric rows omitted.")
if __name__=='__main__':main()
