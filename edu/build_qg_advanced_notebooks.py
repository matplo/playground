"""Build the advanced quark/gluon teaching notebooks deterministically."""
import hashlib
import json
from pathlib import Path


def md(text):
    source=text.strip()+"\n"
    return {"cell_type":"markdown","id":hashlib.sha1(("md"+source).encode()).hexdigest()[:8],"metadata":{},"source":source.splitlines(keepends=True)}


def code(text):
    source=text.strip()+"\n"
    return {"cell_type":"code","execution_count":None,
            "id":hashlib.sha1(("code"+source).encode()).hexdigest()[:8],
            "metadata":{},"outputs":[],"source":source.splitlines(keepends=True)}


META={"kernelspec":{"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"},
      "language_info":{"codemirror_mode":{"name":"ipython","version":3},
                       "file_extension":".py","mimetype":"text/x-python",
                       "name":"python","nbconvert_exporter":"python",
                       "pygments_lexer":"ipython3","version":"3.10.12"}}


BOOT = r'''
import importlib.util
required = ['numpy', 'pandas', 'pyarrow', 'matplotlib', 'sklearn', 'torch', 'tqdm']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError(
        f"Missing packages: {missing}. From a terminal in this directory run "
        "./setup_student_env.sh (or use --current inside an existing henv), "
        "restart Jupyter from that henv, and select its registered kernel."
    )

import json, os, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import qg_constituent_ml as qg

DEVICE = qg.choose_device()
RUN_MODE = os.getenv('QG_RUN_MODE', 'quick')
SOURCE = Path(os.getenv('QG_INPUT_PATH', 'data/inclusive_jets.parquet'))
print(f'PyTorch {torch.__version__}; built with CUDA {torch.version.cuda}')
print(f'device={DEVICE}' + (f'; GPU={torch.cuda.get_device_name(0)}' if DEVICE.type == 'cuda' else ''))
print(f'run mode={RUN_MODE}; source={SOURCE}')
'''


def write(name,cells):
    Path(name).write_text(json.dumps({"cells":cells,"metadata":META,"nbformat":4,"nbformat_minor":5},indent=1)+"\n")


def architecture_notebook(arch,title,description):
    return [
      md(f'''# {title}: quark/gluon classification

This notebook trains a **{title}** on every constituent of each truth-matched jet.
It follows the common preparation lesson and saves a self-describing model bundle.

{description}

Set `QG_RUN_MODE=full` before launching Jupyter for the larger configuration. Quick
mode is the default. Both automatically use CUDA when it is available.'''),
      md('## 1. Environment and device\n\nRun `./setup_student_env.sh` once before this lesson. This check confirms that the selected Jupyter kernel is the student-owned henv; CPU remains a supported fallback.'),
      code(BOOT),
      md('## 2. Current data and model\n\nPreparation is fingerprint-aware: changing the generated Parquet file creates a new prepared-data namespace automatically.'),
      code(f'''prepared = qg.prepare_dataset(SOURCE)
manifest = qg.load_manifest(prepared)
MODEL_CONFIG = qg.default_config('{arch}', RUN_MODE)
model = qg.create_model('{arch}', config=MODEL_CONFIG)
print(json.dumps({{k: manifest[k] for k in ('source_sha256','n_jets','n_constituents','split_counts','class_counts')}}, indent=2))
print(model)
print(f'parameters={{sum(p.numel() for p in model.parameters()):,}}; config={{MODEL_CONFIG}}')'''),
      md('### Canonical implementation\n\nThe reusable class lives in `qg_constituent_ml.py` so saved models can be reconstructed later. Its source is displayed here to keep the architecture visible.'),
      code(f'''import inspect
from IPython.display import Code, display
display(Code(inspect.getsource(qg.model_classes()['{arch}']), language='python'))'''),
      md('## 3. Invariance checks\n\nPadding must not affect a prediction. Reordering constituents must not change an unordered-set classifier.'),
      code(f'''loaders = qg.make_loaders(prepared, '{arch}', RUN_MODE)
batch = next(iter(loaders[2]))
model.eval()
with torch.no_grad():
    base = model(batch['features'][:4], batch['coords'][:4], batch['mask'][:4])
    order = torch.randperm(batch['features'].shape[1])
    permuted = model(batch['features'][:4, order], batch['coords'][:4, order], batch['mask'][:4, order])
    padded_f = torch.nn.functional.pad(batch['features'][:4], (0,0,0,3))
    padded_c = torch.nn.functional.pad(batch['coords'][:4], (0,0,0,3))
    padded_m = torch.nn.functional.pad(batch['mask'][:4], (0,3))
    padded = model(padded_f, padded_c, padded_m)
assert torch.allclose(base, permuted, atol=2e-5)
assert torch.allclose(base, padded, atol=2e-5)
print('Passed permutation and padding invariance checks.')'''),
      md('## 4. Train and save\n\nTraining is balanced, validation/test mixtures are natural, and early stopping restores the best validation checkpoint.'),
      code(f'''history, metrics, predictions = qg.train_model(model, loaders, '{arch}', RUN_MODE, DEVICE)
bundle = qg.save_model_bundle(model, '{arch}', RUN_MODE, MODEL_CONFIG, prepared,
                              history, metrics, predictions)
print(json.dumps(metrics, indent=2))
print(f'Saved model bundle: {{bundle}}')'''),
      code('''from sklearn.metrics import roc_curve
fig, axes = plt.subplots(1, 2, figsize=(11,4))
axes[0].plot([x['epoch'] for x in history], [x['train_loss'] for x in history], marker='o')
axes[0].set(xlabel='epoch', ylabel='BCE loss', title='Training history')
fpr,tpr,_=roc_curve(predictions['labels'], predictions['scores'])
axes[1].plot(tpr,1/np.clip(fpr,1e-3,None),label=f"AUC={metrics['roc_auc']:.3f}")
axes[1].set(xlabel='quark efficiency',ylabel='gluon rejection',yscale='log',title='Held-out performance')
axes[1].legend(); plt.tight_layout(); plt.show()'''),
      md('## 5. What this result means\n\nCompare architectures only through the evaluation notebook, which enforces matching dataset and split fingerprints. A larger network on this small sample is not automatically a better physics model.')]


write('demo_quark_gluon_constituent_setup.ipynb',[
 md('''# Preparing variable-length jet constituents

This lesson turns the list-valued Parquet columns into a scalable representation shared by
three advanced classifiers. It uses every constituent, prevents event leakage, and records
the exact source-data fingerprint. Rerunning the generator with ten times more events is
detected automatically.'''),
 md('## 1. Environment'),code(BOOT),
 md('''## 2. Representation

For particle $i$, use $z_i=p_{T,i}/\sum_jp_{T,j}$ and coordinates relative to the jet.
Continuous inputs are `log(z)`, $\Delta\eta$, wrapped $\Delta\phi$, and `log(ΔR)`.
Particle identity is categorical—not an ordinal PDG number. Stable event hashes create
70/15/15 train/validation/test partitions, and normalization sees train particles only.'''),
 code('''prepared = qg.prepare_dataset(SOURCE)
manifest = qg.load_manifest(prepared)
arrays = qg.load_arrays(prepared)
print(f'Prepared directory: {prepared}')
print(json.dumps(manifest, indent=2))
assert arrays['offsets'][-1] == manifest['n_constituents']
assert len(arrays['labels']) == manifest['n_jets']'''),
 md('## 3. Inspect statistics and particle categories'),
 code('''fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].hist(arrays['n_constituents'],bins=range(0,int(arrays['n_constituents'].max())+2),histtype='step')
axes[0].set(xlabel='constituents per jet',ylabel='jets',title='Dynamic batch lengths')
counts=np.asarray(manifest['normalization']['mean']); axes[1].bar(qg.CONTINUOUS_FEATURES,counts)
axes[1].tick_params(axis='x',rotation=25); axes[1].set(title='Training-set raw means')
plt.tight_layout(); plt.show()'''),
 md('## 4. Validate event isolation and dynamic padding'),
 code('''events=np.asarray(arrays['event_ids']); splits=np.asarray(arrays['splits'])
sets=[set(events[splits==i]) for i in range(3)]
assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
JetDataset=qg.make_dataset_classes(); ds=JetDataset(prepared,split='train')
batch=qg.collate_jets([ds[i] for i in range(min(8,len(ds)))])
assert batch['mask'].sum().item() == sum(len(ds[i]['features']) for i in range(min(8,len(ds))))
print({k:tuple(v.shape) for k,v in batch.items() if hasattr(v,'shape')})'''),
 md('''## 5. Scaling the statistics

Change the visible generator setting or launch it with `QG_N_EVENTS=200000` for ten times
the default event count. The resulting Parquet SHA-256 changes; this setup creates a new
memory-mappable prepared directory, while model checkpoints trained on older statistics
remain separately identified.''')])

write('demo_quark_gluon_pfn.ipynb',architecture_notebook('pfn','Particle Flow Network',
 'A shared network embeds each particle; summing those embeddings enforces permutation invariance before jet-level classification.'))
write('demo_quark_gluon_particle_transformer.ipynb',architecture_notebook('transformer','Compact Particle Transformer',
 'Self-attention relates every pair of constituents. Pairwise angular biases inject geometry without imposing a sequence order.'))
write('demo_quark_gluon_particlenet.ipynb',architecture_notebook('particlenet','ParticleNet-style graph network',
 'EdgeConv learns from local constituent neighborhoods, treating a jet as a particle cloud rather than a fixed vector.'))

write('demo_quark_gluon_model_evaluation.ipynb',[
 md('''# Load, apply, and compare saved constituent models

This notebook reconstructs each architecture from its JSON configuration and weights-only
checkpoint. Direct comparisons are allowed only on the same dataset and split fingerprint.'''),
 md('## 1. Environment'),code(BOOT),
 code('''prepared=qg.prepare_dataset(SOURCE); manifest=qg.load_manifest(prepared)
bundles=qg.discover_bundles(dataset_fingerprint=manifest['source_sha256'])
if not bundles: raise FileNotFoundError('Train at least one architecture notebook first.')
print('\\n'.join(map(str,bundles)))'''),
 md('## 2. Reload checkpoints and reproduce their predictions'),
 code('''from sklearn.metrics import roc_curve
rows=[]; curves={}
for bundle in tqdm(bundles, desc='Evaluating saved models', unit='model'):
    model,config=qg.load_model_bundle(bundle,DEVICE)
    loaders=qg.make_loaders(prepared,config['architecture'],config['mode'])
    pred=qg.predict(model,loaders[2],DEVICE,progress=True,
                    description=f"Evaluating {config['architecture']}"); saved=np.load(bundle/'predictions.npz')
    assert np.array_equal(pred['event_ids'],saved['event_ids']) and np.array_equal(pred['jet_ids'],saved['jet_ids'])
    assert np.allclose(pred['scores'],saved['scores'],atol=2e-5)
    metrics=qg.binary_metrics(pred['labels'],pred['scores']); metrics['model']=f"{config['architecture']} ({config['mode']})"
    metrics['parameters']=sum(p.numel() for p in model.parameters()); metrics['bundle']=str(bundle); rows.append(metrics)
    curves[metrics['model']]=(*roc_curve(pred['labels'],pred['scores'])[:2],metrics['roc_auc'])
import pandas as pd
results=pd.DataFrame(rows).sort_values('roc_auc',ascending=False); display(results)'''),
 code('''fig,ax=plt.subplots(figsize=(7,5))
for name,(fpr,tpr,auc) in curves.items(): ax.plot(tpr,1/np.clip(fpr,1e-3,None),label=f'{name}: {auc:.3f}')
ax.set(xlabel='quark efficiency',ylabel='gluon rejection',yscale='log',title='Common held-out comparison')
ax.legend(); plt.tight_layout(); plt.show()'''),
 md('''## 3. Apply a model to another sample

Set `QG_EVAL_PATH` before launching to evaluate a different compatible Parquet file. The
saved training normalization is reused; it must never be refitted on the evaluation sample.'''),
 code('''EVAL_PATH=os.getenv('QG_EVAL_PATH')
if EVAL_PATH:
    evaluation,config=qg.predict_parquet(bundles[0],EVAL_PATH,device=DEVICE)
    labeled=evaluation['labels'] >= 0
    print(f"Scored all {len(evaluation['scores']):,} jets from {EVAL_PATH}; {labeled.sum():,} have quark/gluon labels")
    if labeled.any(): print(json.dumps(qg.binary_metrics(evaluation['labels'][labeled],evaluation['scores'][labeled]),indent=2))
else: print('QG_EVAL_PATH is unset; common-test comparison complete.')''')])

write('demo_quark_gluon_model_interpretation.ipynb',[
 md('''# What do the constituent models rely on?

This is not a search for causal or universal quark/gluon features. It measures how trained
models respond to controlled input removal and which constituents locally influence a score.'''),
 md('## 1. Environment and compatible models'),code(BOOT),
 code('''prepared=qg.prepare_dataset(SOURCE); manifest=qg.load_manifest(prepared)
bundles=qg.discover_bundles(dataset_fingerprint=manifest['source_sha256'])
if not bundles: raise FileNotFoundError('Train at least one architecture notebook first.')'''),
 md('''## 2. Physics-aware group ablations

Zero in normalized space means replacing a continuous feature by its training mean. PID
ablation removes category information. Constituent removals preserve the remaining particles
but answer a model-reliance question—not a causal physics question.'''),
 code('''def predict_with_ablation(model,loader,kind):
    scores=[]; labels=[]; mean=np.array(manifest['normalization']['mean']); std=np.array(manifest['normalization']['std'])
    model.eval()
    with torch.no_grad():
      for b in tqdm(loader, desc=f'Ablation: {kind}', unit='batch', leave=False):
        f=b['features'].to(DEVICE).clone(); c=b['coords'].to(DEVICE).clone(); m=b['mask'].to(DEVICE).clone()
        if kind=='momentum': f[:,:,0]=0
        elif kind=='angular': f[:,:,1:4]=0; c.zero_()
        elif kind=='pid': f[:,:,4:]=0
        else:
          raw_logz=f[:,:,0]*std[0]+mean[0]; z=torch.exp(raw_logz); dr=torch.linalg.vector_norm(c,dim=-1)
          if kind=='soft': m &= z>=0.05
          elif kind=='core': m &= dr>=0.10
          elif kind=='wide': m &= dr<0.20
        scores.append(torch.sigmoid(model(f,c,m)).cpu().numpy()); labels.append(b['labels'].numpy())
    return np.concatenate(labels).astype(int),np.concatenate(scores)

rows=[]
for bundle in tqdm(bundles, desc='Interpreting models', unit='model'):
  model,config=qg.load_model_bundle(bundle,DEVICE); loader=qg.make_loaders(prepared,config['architecture'],config['mode'])[2]
  y,base=predict_with_ablation(model,loader,'none'); base_auc=qg.binary_metrics(y,base)['roc_auc']
  for kind in ('momentum','angular','pid','soft','core','wide'):
    _,score=predict_with_ablation(model,loader,kind); auc=qg.binary_metrics(y,score)['roc_auc']
    rows.append({'model':config['architecture'],'ablation':kind,'AUC':auc,'delta_AUC':auc-base_auc})
import pandas as pd
ablation=pd.DataFrame(rows); display(ablation.pivot(index='ablation',columns='model',values='delta_AUC'))'''),
 code('''pivot=ablation.pivot(index='ablation',columns='model',values='delta_AUC')
pivot.plot.bar(figsize=(10,4)); plt.axhline(0,color='black',lw=.8); plt.ylabel('AUC(ablation) - AUC(original)')
plt.title('Performance reliance on input groups'); plt.tight_layout(); plt.show()'''),
 md('''## 3. Local integrated-gradient maps

For one jet, interpolate continuous inputs and angular coordinates from a mean/zero baseline.
PID stays fixed and should instead be studied by categorical occlusion. Attribution magnitude
is normalized within each model, so colors are not compared as absolute units across models.'''),
 code('''def integrated_gradient_map(model,batch,steps=24):
    full_f=batch['features'][:1].to(DEVICE); full_c=batch['coords'][:1].to(DEVICE); mask=batch['mask'][:1].to(DEVICE)
    base_f=full_f.clone(); base_f[:,:,:4]=0; base_c=torch.zeros_like(full_c); gf=torch.zeros_like(full_f); gc=torch.zeros_like(full_c)
    for alpha in torch.linspace(0,1,steps,device=DEVICE):
      f=(base_f+alpha*(full_f-base_f)).detach().requires_grad_(True); c=(base_c+alpha*(full_c-base_c)).detach().requires_grad_(True)
      score=torch.sigmoid(model(f,c,mask)).sum(); df,dc=torch.autograd.grad(score,(f,c),allow_unused=True)
      gf += torch.zeros_like(f) if df is None else df; gc += torch.zeros_like(c) if dc is None else dc
    attr=((full_f-base_f)*gf/steps).abs().sum(-1)+((full_c-base_c)*gc/steps).abs().sum(-1)
    return full_c[0].detach().cpu().numpy(),attr[0].detach().cpu().numpy(),mask[0].cpu().numpy()

fig,axes=plt.subplots(1,len(bundles),figsize=(5*len(bundles),4),squeeze=False)
for ax,bundle in zip(axes[0],bundles):
  model,config=qg.load_model_bundle(bundle,DEVICE); loader=qg.make_loaders(prepared,config['architecture'],config['mode'])[2]; batch=next(iter(loader))
  coords,attr,mask=integrated_gradient_map(model,batch); values=attr[mask]; values=values/(values.max()+1e-12)
  sc=ax.scatter(coords[mask,0],coords[mask,1],c=values,s=30+100*values,cmap='viridis'); ax.set(title=config['architecture'],xlabel=r'$\Delta\eta$',ylabel=r'$\Delta\phi$')
  fig.colorbar(sc,ax=ax,label='normalized attribution')
plt.tight_layout(); plt.show()'''),
 md('''## 4. Interpretation limits

Large ablation losses identify reliance, not a uniquely important physical variable. Inputs
are correlated; removing them can create out-of-distribution jets. Generator truth labels are
idealized and process dependent. Attention weights and graph edges describe internal routing,
not causal explanations.''')])


# Patch the generator controls and manifest without rewriting its narrative structure.
path=Path('demo_quark_gluon_samples.ipynb'); nb=json.loads(path.read_text())
for cell in nb['cells']:
    src=''.join(cell.get('source',[]))
    if cell.get('cell_type') == 'code' and 'import pandas as pd' in src and 'from tqdm.auto import tqdm' not in src:
        src=src.replace('import pandas as pd', 'import pandas as pd\nfrom tqdm.auto import tqdm')
    elif cell.get('cell_type') != 'code':
        src=src.replace('import pandas as pd\nfrom tqdm.auto import tqdm', 'import pandas as pd')
    # Normalize remnants from early, non-idempotent versions of this builder.
    src=src.replace("import os\nPYTHIA_SEED = int(os.getenv('QG_SEED', '7'))\nimport os\nPYTHIA_SEED = int(os.getenv('QG_SEED', '7'))",
                    "import os\nPYTHIA_SEED = int(os.getenv('QG_SEED', '7'))")
    src=src.replace("pythia.readString('Random:setSeed = on')\npythia.readString(f'Random:seed = {PYTHIA_SEED}')\npythia.readString('Random:setSeed = on')\npythia.readString(f'Random:seed = {PYTHIA_SEED}')",
                    "pythia.readString('Random:setSeed = on')\npythia.readString(f'Random:seed = {PYTHIA_SEED}')")
    if "pythia = pythia8.Pythia()" in src and "PYTHIA_SEED" not in src:
        src=src.replace("pythia = pythia8.Pythia()", "import os\nPYTHIA_SEED = int(os.getenv('QG_SEED', '7'))\npythia = pythia8.Pythia()")
        src=src.replace("pythia.readString('Next:numberShowEvent = 0')", "pythia.readString('Random:setSeed = on')\npythia.readString(f'Random:seed = {PYTHIA_SEED}')\npythia.readString('Next:numberShowEvent = 0')")
    if "N_EVENTS = 20000" in src:
        src=src.replace("N_EVENTS = 20000", "N_EVENTS = int(os.getenv('QG_N_EVENTS', '20000'))")
        src=src.replace("OUT_DIR = 'data'", "OUT_DIR = os.getenv('QG_OUTPUT_DIR', 'data')")
    if src.startswith('records = []') and 'n_generated = 0' not in src:
        src=src.replace('records = []','records = []\nn_generated = 0')
        src=src.replace('    cs, jets, part_pdgid = cluster_event()','    n_generated += 1\n    cs, jets, part_pdgid = cluster_event()')
    if src.startswith('records = []'):
        src=src.replace('for event_id in range(N_EVENTS):',
                        "for event_id in tqdm(range(N_EVENTS), desc='Generating events', unit='event'):")
        src=src.replace("\n    if (event_id + 1) % 5000 == 0:\n        print(f'... {event_id + 1}/{N_EVENTS} events, {len(records)} jets collected so far')\n", '\n')
    if "quark_df.to_parquet" in src and "generation_manifest" not in src:
        src += '''\n\nimport hashlib, json\ndef file_sha256(path):\n    digest = hashlib.sha256()\n    with open(path, 'rb') as stream:\n        for chunk in iter(lambda: stream.read(1 << 20), b''):\n            digest.update(chunk)\n    return digest.hexdigest()\n\nmanifest = {\n    'schema_version': 1, 'n_events_requested': N_EVENTS,\n    'n_events_generated': n_generated, 'pythia_seed': PYTHIA_SEED,\n    'sqrt_s_gev': 13000.0, 'pt_min_gev': pt_min, 'eta_max': ETA_MAX,\n    'jet_R': R, 'match_R': MATCH_R,\n    'jet_counts': {str(k): int(v) for k, v in inclusive_df['flavor'].value_counts().items()},\n    'inclusive_sha256': file_sha256(paths['inclusive']),\n}\nmanifest_path = os.path.join(OUT_DIR, 'quark_gluon_generation_manifest.json')\nwith open(manifest_path, 'w') as stream:\n    json.dump(manifest, stream, indent=2, sort_keys=True)\nprint(f'Wrote generation manifest: {manifest_path}')\n'''
    cell['source']=src.splitlines(keepends=True)
path.write_text(json.dumps(nb,indent=1)+"\n")


# Add progress reporting to the other long, Python-level teaching loops.
for notebook_name in ('demo_pythia_fastjet.ipynb', 'demo_quark_gluon_classification.ipynb'):
    path = Path(notebook_name); nb = json.loads(path.read_text())
    for cell in nb['cells']:
        src = ''.join(cell.get('source', []))
        if cell.get('cell_type') == 'code' and 'import pandas as pd' in src and 'from tqdm.auto import tqdm' not in src:
            src = src.replace('import pandas as pd', 'import pandas as pd\nfrom tqdm.auto import tqdm')
        if notebook_name == 'demo_pythia_fastjet.ipynb':
            if 'import pythia8' in src and 'from tqdm.auto import tqdm' not in src:
                src = src.replace('import pythia8', 'import pythia8\nfrom tqdm.auto import tqdm')
            src = src.replace('for _ in range(n_events):',
                              "for _ in tqdm(range(n_events), desc='Generating events', unit='event'):")
        else:
            src = src.replace('[compute_jet_features(row) for row in jets.itertuples(index=False)]',
                              "[compute_jet_features(row) for row in tqdm(jets.itertuples(index=False), total=len(jets), desc='Computing jet features', unit='jet')]")
            src = src.replace("top_constituent_inputs(row) for row in model_data.itertuples(index=False)",
                              "top_constituent_inputs(row) for row in tqdm(model_data.itertuples(index=False), total=len(model_data), desc='Building constituent inputs', unit='jet')")
        cell['source'] = src.splitlines(keepends=True)
    path.write_text(json.dumps(nb, indent=1) + "\n")
