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


SOURCE_SETTINGS = r'''
# Student control: edit this value, then run the notebook from the top.
SOURCE_PATH = 'data/inclusive_jets.parquet'
'''

TRAINING_SETTINGS = SOURCE_SETTINGS + r'''
RUN_MODE = 'quick'  # 'quick' for a short lesson; 'full' uses all training jets
'''


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

import json, time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import qg_constituent_ml as qg

DEVICE = qg.choose_device()
RUN_MODE_VALUE = globals().get('RUN_MODE')
if RUN_MODE_VALUE is not None and RUN_MODE_VALUE not in {'quick', 'full'}:
    raise ValueError("RUN_MODE must be 'quick' or 'full'")
SOURCE = Path(SOURCE_PATH)
print(f'PyTorch {torch.__version__}; built with CUDA {torch.version.cuda}')
print(f'device={DEVICE}' + (f'; GPU={torch.cuda.get_device_name(0)}' if DEVICE.type == 'cuda' else ''))
print((f'run mode={RUN_MODE_VALUE}; ' if RUN_MODE_VALUE else '') + f'source={SOURCE}')
'''


def write(name,cells):
    Path(name).write_text(json.dumps({"cells":cells,"metadata":META,"nbformat":4,"nbformat_minor":5},indent=1)+"\n")


def architecture_notebook(arch,title,description):
    return [
      md(f'''# {title}: quark/gluon classification

This notebook trains a **{title}** on every constituent of each truth-matched jet.
A **jet** is a narrow spray of particles produced when a high-energy quark or gluon
turns into observable particles. Those particles are the jet's **constituents**.
The classification task is to use their measured patterns to estimate whether the
original particle was a quark or a gluon.

This is **supervised machine learning**: each training example has inputs (constituent
measurements) and a target answer (the simulation-derived quark/gluon label). It follows
the common preparation lesson and saves a self-describing model bundle.

{description}

Set `RUN_MODE = 'full'` in the student-controls cell for the larger configuration.
Quick mode is the default. Both automatically use CUDA when it is available.'''),
      md('''## 0. Student controls

These ordinary Python variables are the main controls for the lesson. Quick mode limits
training statistics for a short interactive run; full mode uses every training jet. Changing
the input file automatically creates a different fingerprinted prepared dataset.'''),
      code(TRAINING_SETTINGS),
      md('''## 1. Environment and computing device

Run `./setup_student_env.sh` once before this lesson. A **Jupyter kernel** is the Python
process that actually executes notebook cells; selecting the henv kernel makes sure it
can see the packages installed for this course.

PyTorch can calculate on a CPU or on a GPU. A **GPU** performs many similar arithmetic
operations at once, which is useful for neural-network training. **CUDA** is the software
interface PyTorch uses to access an NVIDIA GPU. The code chooses CUDA automatically when
available, but gives the same lesson on a CPU.'''),
      code(BOOT),
      md('''## 2. Current data and model

Preparation is fingerprint-aware: changing the generated Parquet file creates a new
prepared-data namespace automatically. A **fingerprint** is a cryptographic summary
(SHA-256 here) of a file's exact bytes. Even a tiny file change produces a different
fingerprint, preventing us from silently pairing a model with the wrong dataset.

The printed **parameter count** is the number of adjustable numerical values in the neural
network. Training changes these values. More parameters can represent more complicated
patterns, but also require more data and can make overfitting easier.'''),
      code(f'''prepared = qg.prepare_dataset(SOURCE)
manifest = qg.load_manifest(prepared)
MODEL_CONFIG = qg.default_config('{arch}', RUN_MODE)
model = qg.create_model('{arch}', config=MODEL_CONFIG)
print(json.dumps({{k: manifest[k] for k in ('source_sha256','n_jets','n_constituents','split_counts','class_counts')}}, indent=2))
print(model)
print(f'parameters={{sum(p.numel() for p in model.parameters()):,}}; config={{MODEL_CONFIG}}')'''),
      md('''### Canonical implementation

A neural network is built from mathematical **layers**. Each layer transforms numbers into
new numbers; during training, the network learns which transformations help predict the
label. The final raw output is a **logit**. Applying the sigmoid function converts it to a
score between 0 and 1, where larger values mean “more quark-like” in this project.

The reusable class lives in `qg_constituent_ml.py` so saved models can be reconstructed
later. Its source is displayed here to keep the architecture visible.'''),
      code(f'''import inspect
from IPython.display import Code, display
display(Code(inspect.getsource(qg.model_classes()['{arch}']), language='python'))'''),
      md('''## 3. Invariance checks

Jets contain different numbers of particles. To place several jets in one rectangular
tensor, shorter lists receive dummy entries called **padding**. A Boolean **mask** tells the
model which entries are real. Adding more dummy entries must not change a prediction.

The constituents form an unordered **set**, not a sentence: exchanging particle 2 and
particle 7 should not change the jet. This property is called **permutation invariance**.
The assertions below are unit tests for both physical requirements.'''),
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
      md('''## 4. Train, validate, test, and save

During **training**, the model predicts labels, a **loss function** measures its errors,
and backpropagation computes how each parameter contributed to those errors. An optimizer
then makes a small parameter update. One pass through the training sample is an **epoch**.

The training batches are balanced so quarks and gluons contribute equally. A separate
**validation set** chooses when to stop and is never used for parameter updates. **Early
stopping** keeps the checkpoint from the epoch with the best validation result, limiting
overfitting. The **test set** is touched only for the final measurement and keeps the
naturally occurring class mixture.'''),
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
      md('''## 5. Reading the plots and result

**Binary cross-entropy (BCE)** is the training loss: smaller values mean that the predicted
scores agree better with the known labels. The **ROC curve** scans every possible score
threshold. Quark efficiency is the fraction of true quark jets retained; gluon rejection
is the inverse of the fraction of gluon jets mistakenly retained. **AUC** summarizes the
ROC curve: 0.5 is random ordering and 1.0 is perfect ordering on this sample.

Compare architectures only through the evaluation notebook, which enforces matching dataset
and split fingerprints. A larger network on a small sample is not automatically a better
physics model: it may learn statistical fluctuations (**overfit**) instead of patterns that
generalize to unseen jets.''')]


write('demo_quark_gluon_constituent_setup.ipynb',[
 md('''# Preparing variable-length jet constituents

This lesson turns the list-valued Parquet columns into a scalable representation shared by
three advanced classifiers. It uses every constituent, prevents event leakage, and records
the exact source-data fingerprint. Rerunning the generator with ten times more events is
detected automatically.'''),
 md('''## 0. Student controls

Edit the source path directly here. In each later training notebook, a separate `RUN_MODE`
variable controls the amount of training data and model size, not the number of particles
per jet. The advanced models always use every constituent.'''),
 code(SOURCE_SETTINGS),
 md('## 1. Environment'),code(BOOT),
 md('''## 2. From particles to numbers a neural network can use

Each jet is described relative to its own axis. **Transverse momentum** $p_T$ is momentum
perpendicular to the proton beams. **Pseudorapidity** $\eta$ describes direction along the
beam, while the **azimuthal angle** $\phi$ describes direction around it. Angular separation
is $\Delta R=\sqrt{(\Delta\eta)^2+(\Delta\phi)^2}$. Because $\phi$ wraps around at $2\pi$,
$\Delta\phi$ is wrapped to the shortest signed angular distance.

For particle $i$, $z_i=p_{T,i}/\sum_jp_{T,j}$ is its fraction of the jet's total constituent
$p_T$. The continuous inputs are `log(z)`, $\Delta\eta$, wrapped $\Delta\phi$, and
`log(ΔR)`. Logarithms compress quantities spanning many orders of magnitude, making
optimization easier.

The **PDG ID** names a particle species, such as a charged pion or photon. It is a category,
not a measured amount: ID 211 is not “larger” than ID 22. We therefore map species to learned
category embeddings instead of feeding the ID as an ordinary number.

Stable hashes of `event_id` create 70/15/15 train/validation/test partitions. All jets from
one collision event stay together, preventing **data leakage**—accidentally giving related
information to both training and evaluation. **Normalization** subtracts a feature's training
mean and divides by its training standard deviation. Only training particles determine these
numbers, so information from validation or test data cannot leak backward.'''),
 code('''prepared = qg.prepare_dataset(SOURCE)
manifest = qg.load_manifest(prepared)
arrays = qg.load_arrays(prepared)
print(f'Prepared directory: {prepared}')
print(json.dumps(manifest, indent=2))
assert arrays['offsets'][-1] == manifest['n_constituents']
assert len(arrays['labels']) == manifest['n_jets']'''),
 md('''## 3. Inspect statistics and particle categories

A jet is **variable-length** because it may contain a few particles or many. The histogram
shows this multiplicity. During batching, each batch is padded only to its longest jet rather
than to one global maximum; this reduces wasted memory when statistics grow.'''),
 code('''fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].hist(arrays['n_constituents'],bins=range(0,int(arrays['n_constituents'].max())+2),histtype='step')
axes[0].set(xlabel='constituents per jet',ylabel='jets',title='Dynamic batch lengths')
counts=np.asarray(manifest['normalization']['mean']); axes[1].bar(qg.CONTINUOUS_FEATURES,counts)
axes[1].tick_params(axis='x',rotation=25); axes[1].set(title='Training-set raw means')
plt.tight_layout(); plt.show()'''),
 md('''## 4. Validate event isolation and dynamic padding

An **assertion** is an executable statement of something that must be true. These checks
verify that no event occurs in two data splits and that the mask counts exactly the real
particles after padding. If either promise is broken, execution stops immediately.'''),
 code('''events=np.asarray(arrays['event_ids']); splits=np.asarray(arrays['splits'])
sets=[set(events[splits==i]) for i in range(3)]
assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])
JetDataset=qg.make_dataset_classes(); ds=JetDataset(prepared,split='train')
batch=qg.collate_jets([ds[i] for i in range(min(8,len(ds)))])
assert batch['mask'].sum().item() == sum(len(ds[i]['features']) for i in range(min(8,len(ds))))
print({k:tuple(v.shape) for k,v in batch.items() if hasattr(v,'shape')})'''),
 md('''## 5. Scaling the statistics

Change `N_EVENTS` in the generator notebook—for example, from 20,000 to 200,000 for ten
times the default event count. The resulting Parquet SHA-256 changes; this setup creates a new
memory-mappable prepared directory, while model checkpoints trained on older statistics
remain separately identified.''')])

write('demo_quark_gluon_pfn.ipynb',architecture_notebook('pfn','Particle Flow Network',
 '''A **Particle Flow Network (PFN)** applies the same small neural network to every particle,
turning it into a learned vector called an **embedding**. It sums all particle embeddings and
uses a second network to classify the whole jet. A sum is unchanged when its terms are
reordered, so this design builds permutation invariance into the architecture.'''))
write('demo_quark_gluon_particle_transformer.ipynb',architecture_notebook('transformer','Compact Particle Transformer',
 '''A **Transformer** uses **self-attention** to let every constituent compare itself with every
other constituent and learn which relationships matter. Pairwise angular biases supply each
pair's geometric separation. There is no positional sequence encoding because a jet is an
unordered particle set, not a sentence.'''))
write('demo_quark_gluon_particlenet.ipynb',architecture_notebook('particlenet','ParticleNet-style graph network',
 '''A **graph** represents constituents as nodes and nearby pairs as edges. **EdgeConv** builds
features from a particle and its nearest neighbors, then combines those local messages. This
treats the jet as a particle cloud and helps the model learn localized radiation patterns.'''))

write('demo_quark_gluon_model_evaluation.ipynb',[
 md('''# Load, apply, and compare all saved classifiers

This notebook compares the logistic regression, BDTs, compact top-particle MLP, PFN,
Particle Transformer, and ParticleNet-style network on exactly the same held-out jets.
It reconstructs each neural architecture from its JSON configuration and weights-only
checkpoint and loads common-test predictions saved by the baseline notebook. A **checkpoint**
stores learned parameter values so inference can be performed without training again.
**Inference** means applying a trained model to obtain scores.

Direct comparisons are allowed only on the same dataset and split fingerprint. Otherwise a
score difference might come from easier test examples rather than a better architecture.'''),
 md('''## 0. Student controls

Use the same input sample as the training notebooks. Set `EVAL_PATH` to another compatible
Parquet file only for the optional final application section.'''),
 code(SOURCE_SETTINGS + "\n# Optional independent sample to score after the common comparison.\nEVAL_PATH = None"),
 md('## 1. Environment'),code(BOOT),
 code('''prepared=qg.prepare_dataset(SOURCE); manifest=qg.load_manifest(prepared)
bundles=qg.discover_bundles(dataset_fingerprint=manifest['source_sha256'])
if not bundles: raise FileNotFoundError('Train at least one architecture notebook first.')
print('\\n'.join(map(str,bundles)))'''),
 md('''## 2. Reload checkpoints and reproduce their predictions

Reproducing the saved scores is a consistency test: the architecture, preprocessing, and
weights were all restored correctly. The table reports parameter count and classification
metrics. A fair comparison uses identical test jets and never selects a winner by repeatedly
looking at the test set.'''),
 code('''from sklearn.metrics import roc_curve
rows=[]; curves={}; reference=None
for bundle in tqdm(bundles, desc='Evaluating saved models', unit='model'):
    model,config=qg.load_model_bundle(bundle,DEVICE)
    loaders=qg.make_loaders(prepared,config['architecture'],config['mode'])
    pred=qg.predict(model,loaders[2],DEVICE,progress=True,
                    description=f"Evaluating {config['architecture']}"); saved=np.load(bundle/'predictions.npz')
    assert np.array_equal(pred['event_ids'],saved['event_ids']) and np.array_equal(pred['jet_ids'],saved['jet_ids'])
    assert np.allclose(pred['scores'],saved['scores'],atol=2e-5)
    metrics=qg.binary_metrics(pred['labels'],pred['scores']); metrics['model']=f"{config['architecture']} ({config['mode']})"
    metrics['family']='all-constituent neural network'
    metrics['parameters']=sum(p.numel() for p in model.parameters()); metrics['bundle']=str(bundle); rows.append(metrics)
    curves[metrics['model']]=(*roc_curve(pred['labels'],pred['scores'])[:2],metrics['roc_auc'])
    current=(pred['event_ids'],pred['jet_ids'],pred['labels'])
    if reference is None: reference=current
    else:
        assert all(np.array_equal(a,b) for a,b in zip(reference,current))

baseline_bundle=Path('artifacts/qg_baselines')/manifest['source_sha256'][:12]
baseline_predictions=baseline_bundle/'predictions.npz'
if baseline_predictions.exists():
    import joblib
    baseline_config=json.loads((baseline_bundle/'config.json').read_text())
    baseline_models=joblib.load(baseline_bundle/'models.joblib')
    assert baseline_config['dataset_fingerprint']==manifest['source_sha256']
    assert baseline_config['split_fingerprint']==manifest['split_fingerprint']
    assert set(baseline_models['models'])==set(baseline_config['model_names'])
    saved=np.load(baseline_predictions)
    current=(saved['event_ids'],saved['jet_ids'],saved['labels'])
    if reference is not None:
        assert all(np.array_equal(a,b) for a,b in zip(reference,current)), 'Baseline and neural test jets differ'
    for name,score in zip(saved['model_names'].astype(str),saved['scores']):
        metrics=qg.binary_metrics(saved['labels'],score); metrics['model']=name
        metrics['family']='classical / compact baseline'; metrics['parameters']=np.nan
        metrics['bundle']=str(baseline_bundle); rows.append(metrics)
        curves[name]=(*roc_curve(saved['labels'],score)[:2],metrics['roc_auc'])
else:
    print(f'Baseline bundle not found at {baseline_bundle}. Run the classification notebook to add logistic, BDT, and MLP results.')

import pandas as pd
results=pd.DataFrame(rows).sort_values('roc_auc',ascending=False); display(results)'''),
 code('''fig,ax=plt.subplots(figsize=(7,5))
for name,(fpr,tpr,auc) in curves.items(): ax.plot(tpr,1/np.clip(fpr,1e-3,None),label=f'{name}: {auc:.3f}')
ax.set(xlabel='quark efficiency',ylabel='gluon rejection',yscale='log',title='All models on the same held-out jets')
ax.legend(); plt.tight_layout(); plt.show()'''),
 md('''## 3. Apply a model to another sample

Set `EVAL_PATH` in the student-controls cell to evaluate a different compatible Parquet file. The
saved training normalization is reused; it must never be refitted on the evaluation sample.
Refitting would allow the new sample to alter preprocessing and would make its scores
inconsistent with the original model.'''),
 code('''if EVAL_PATH:
    evaluation,config=qg.predict_parquet(bundles[0],EVAL_PATH,device=DEVICE)
    labeled=evaluation['labels'] >= 0
    print(f"Scored all {len(evaluation['scores']):,} jets from {EVAL_PATH}; {labeled.sum():,} have quark/gluon labels")
    if labeled.any(): print(json.dumps(qg.binary_metrics(evaluation['labels'][labeled],evaluation['scores'][labeled]),indent=2))
else: print('EVAL_PATH is unset; common-test comparison complete.')''')])

write('demo_quark_gluon_model_interpretation.ipynb',[
 md('''# What do the constituent models rely on?

This is not a search for causal or universal quark/gluon features. It measures how trained
models respond to controlled input removal and which constituents locally influence a score.
This distinction matters: an explanation of a trained model is not automatically a law of
physics or a statement that one variable *causes* a jet to be a quark jet.'''),
 md('''## 0. Student controls

Use the same source path as the saved models you want to interpret.'''),
 code(SOURCE_SETTINGS),
 md('## 1. Environment and compatible models'),code(BOOT),
 code('''prepared=qg.prepare_dataset(SOURCE); manifest=qg.load_manifest(prepared)
bundles=qg.discover_bundles(dataset_fingerprint=manifest['source_sha256'])
if not bundles: raise FileNotFoundError('Train at least one architecture notebook first.')'''),
 md('''## 2. Physics-aware group ablations

An **ablation** deliberately removes one source of information and measures how performance
changes. A large AUC drop says this trained model relied on that information. It does not say
the input is independently responsible, because input features can be correlated.

Zero in normalized space means replacing a continuous feature by its training mean. PID
ablation removes category information. “Soft,” “core,” and “wide” ablations remove selected
constituents while preserving the others. These are model-reliance tests, not causal physics
experiments.'''),
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

**Integrated gradients** compare one jet with a reference, or **baseline**, and accumulate
how the score changes along many small interpolation steps. The resulting **attribution** is
a local sensitivity measure: it highlights constituents that influenced this particular
prediction, with both the model and baseline held fixed.

For one jet, interpolate continuous inputs and angular coordinates from a mean/zero baseline.
PID stays fixed and should instead be studied by categorical occlusion. Attribution magnitude
is normalized within each model, so colors are not compared as absolute units across models.
A bright point is influential for that displayed jet; it is not necessarily important for
all jets.'''),
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
are correlated; removing them can create **out-of-distribution** jets—artificial inputs unlike
anything seen during training. Generator truth labels are
idealized and process dependent. Attention weights and graph edges describe internal routing,
not causal explanations. Reliable scientific conclusions should be checked with several
generators, detector conditions, kinematic regions, and interpretation methods.''')])


# Patch the generator controls and manifest without rewriting its narrative structure.
path=Path('demo_quark_gluon_samples.ipynb'); nb=json.loads(path.read_text())
for cell in nb['cells']:
    src=''.join(cell.get('source',[]))
    src=src.replace('from tqdm.auto import tqdm', 'from tqdm import tqdm')
    if cell.get('cell_type') == 'code' and 'import pandas as pd' in src and 'from tqdm import tqdm' not in src:
        src=src.replace('import pandas as pd', 'import pandas as pd\nfrom tqdm import tqdm')
    elif cell.get('cell_type') != 'code':
        src=src.replace('import pandas as pd\nfrom tqdm import tqdm', 'import pandas as pd')
    # Normalize remnants from early, non-idempotent versions of this builder.
    src=src.replace("import os\nPYTHIA_SEED = int(os.getenv('QG_SEED', '7'))\nimport os\nPYTHIA_SEED = int(os.getenv('QG_SEED', '7'))",
                    "import os\nPYTHIA_SEED = int(os.getenv('QG_SEED', '7'))")
    src=src.replace("pythia.readString('Random:setSeed = on')\npythia.readString(f'Random:seed = {PYTHIA_SEED}')\npythia.readString('Random:setSeed = on')\npythia.readString(f'Random:seed = {PYTHIA_SEED}')",
                    "pythia.readString('Random:setSeed = on')\npythia.readString(f'Random:seed = {PYTHIA_SEED}')")
    src=src.replace("PYTHIA_SEED = int(os.getenv('QG_SEED', '7'))",
                    "PYTHIA_SEED = 7  # edit for another reproducible random sequence")
    src=src.replace("N_EVENTS = int(os.getenv('QG_N_EVENTS', '20000'))",
                    "N_EVENTS = 20_000  # edit this number to change the sample statistics")
    src=src.replace("OUT_DIR = os.getenv('QG_OUTPUT_DIR', 'data')",
                    "OUT_DIR = 'data'  # edit to write the generated files elsewhere")
    if "pythia = pythia8.Pythia()" in src and "PYTHIA_SEED" not in src:
        src=src.replace("pythia = pythia8.Pythia()", "import os\nPYTHIA_SEED = 7  # edit for another reproducible random sequence\npythia = pythia8.Pythia()")
        src=src.replace("pythia.readString('Next:numberShowEvent = 0')", "pythia.readString('Random:setSeed = on')\npythia.readString(f'Random:seed = {PYTHIA_SEED}')\npythia.readString('Next:numberShowEvent = 0')")
    src=src.replace(
        "N_EVENTS = 20_000  # edit this number to change the sample statistics     # bump for more statistics; runtime scales ~linearly",
        "N_EVENTS = 20_000  # edit this number; runtime scales approximately linearly",
    )
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
        src=src.replace('from tqdm.auto import tqdm', 'from tqdm import tqdm')
        if cell.get('cell_type') == 'code' and 'import pandas as pd' in src and 'from tqdm import tqdm' not in src:
            src = src.replace('import pandas as pd', 'import pandas as pd\nfrom tqdm import tqdm')
        if notebook_name == 'demo_pythia_fastjet.ipynb':
            if 'import pythia8' in src and 'from tqdm import tqdm' not in src:
                src = src.replace('import pythia8', 'import pythia8\nfrom tqdm import tqdm')
            src = src.replace('for _ in range(n_events):',
                              "for _ in tqdm(range(n_events), desc='Generating events', unit='event'):")
        else:
            src = src.replace('[compute_jet_features(row) for row in jets.itertuples(index=False)]',
                              "[compute_jet_features(row) for row in tqdm(jets.itertuples(index=False), total=len(jets), desc='Computing jet features', unit='jet')]")
            src = src.replace("top_constituent_inputs(row) for row in model_data.itertuples(index=False)",
                              "top_constituent_inputs(row) for row in tqdm(model_data.itertuples(index=False), total=len(model_data), desc='Building constituent inputs', unit='jet')")
        cell['source'] = src.splitlines(keepends=True)
    path.write_text(json.dumps(nb, indent=1) + "\n")


# Keep the classical baselines on the identical event split used by the constituent
# networks, retain 20 leading particles for the compact MLP, and save reusable artifacts.
path = Path('demo_quark_gluon_classification.ipynb')
nb = json.loads(path.read_text())
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if cell.get('cell_type') == 'markdown':
        if src.startswith('## D. Build leakage-safe ML samples'):
            src = '''## D. Build leakage-safe ML samples

The split is by `event_id`, not by individual jet: every jet from one generated event stays
entirely in train, validation, or test. The training set is downsampled to equal quark/gluon
counts so all models see a balanced fitting sample. The untouched validation and test sets
retain their natural mixtures. The stable event-hash rule is identical to the one used by
the advanced constituent models.

The top-constituent representation sorts constituents by $p_T$, keeps the first 20, and stores
$(z,\Delta\eta,\Delta\phi,\Delta R,\mathrm{mask})$. It is a compact, fixed-size proxy for a raw
constituent model—not permutation invariant and not as expressive as a transformer.
The later PFN, Transformer, and ParticleNet models do **not** use this truncation;
their dynamically padded batches retain every constituent.
'''
        src = src.replace('keeps the first 12', 'keeps the first 20')
        src = src.replace('Top-12 constituent MLP', 'Top-20 constituent MLP')
    elif cell.get('cell_type') == 'code' and 'from sklearn.ensemble import GradientBoostingClassifier' in src:
        if 'import hashlib' not in src:
            src = src.replace('from pathlib import Path\n', 'from pathlib import Path\nimport hashlib\nimport json\n')
        if 'import joblib' not in src:
            src = src.replace('import pandas as pd\n', 'import pandas as pd\nimport joblib\n')
        src = src.replace('from sklearn.model_selection import train_test_split\n', '')
        if 'import qg_constituent_ml as qg' not in src:
            src = src.replace('from IPython.display import display\n',
                              'from IPython.display import display\n\nimport qg_constituent_ml as qg\n')
        if 'MAX_CONSTITUENTS = 20' not in src:
            src = src.replace(
                "RANDOM_STATE = 7\n",
                "RANDOM_STATE = 7\nMAX_CONSTITUENTS = 20  # compact MLP only; advanced models use every particle\n"
                "BASELINE_ROOT = Path('artifacts/qg_baselines')\n",
            )
    elif cell.get('cell_type') == 'code' and 'shape_features = [' in src and (
            'event_ids = model_data' in src or 'split_ids =' in src):
        src = '''shape_features = [
    'n_constituents', 'mass_over_pt', 'ptd', 'lead_pt_fraction',
    'soft_pt_fraction', 'angularity_beta_0p5', 'girth', 'radial_width',
    'core_pt_fraction_0p1', 'core_pt_fraction_0p2', 'eec_beta_1',
]
shape_plus_kinematics = shape_features + ['jet_pt', 'abs_jet_eta']

model_data = jets_with_features[
    jets_with_features['flavor'].isin(['quark', 'gluon'])
].reset_index(drop=True)
model_data['label'] = (model_data['flavor'] == 'quark').astype(int)

# This is exactly the stable 70/15/15 event split used by qg_constituent_ml.py.
split_ids = np.array([
    qg.split_for_event(event_id, RANDOM_STATE)
    for event_id in model_data['event_id']
], dtype=np.uint8)
train_mask = split_ids == 0
validation_mask = split_ids == 1
test_mask = split_ids == 2

rng = np.random.default_rng(RANDOM_STATE)
y_all = model_data['label'].to_numpy()
train_by_class = [np.flatnonzero(train_mask & (y_all == label)) for label in (0, 1)]
n_balanced = min(map(len, train_by_class))
train_indices = np.concatenate([
    rng.choice(indices, size=n_balanced, replace=False) for indices in train_by_class
])
rng.shuffle(train_indices)
validation_indices = np.flatnonzero(validation_mask)
test_indices = np.flatnonzero(test_mask)

assert set(model_data.loc[train_indices, 'event_id']).isdisjoint(
    set(model_data.loc[validation_indices, 'event_id'])
)
assert set(model_data.loc[train_indices, 'event_id']).isdisjoint(
    set(model_data.loc[test_indices, 'event_id'])
)
assert set(model_data.loc[validation_indices, 'event_id']).isdisjoint(
    set(model_data.loc[test_indices, 'event_id'])
)
dataset_fingerprint = qg.sha256_file(INPUT_PATH)
split_fingerprint = hashlib.sha256(split_ids.tobytes()).hexdigest()

print(f'Balanced training sample: {len(train_indices):,} jets '
      f'({n_balanced:,} per class)')
print(f'Validation sample: {len(validation_indices):,} jets')
print(f'Natural test sample: {len(test_indices):,} jets')
display(model_data.loc[test_indices, 'flavor'].value_counts().rename('test jets').to_frame())
'''
    elif cell.get('cell_type') == 'code' and 'def top_constituent_inputs' in src:
        src = src.replace('MAX_CONSTITUENTS = 12\n\n\n', '')
        src = src.replace("print('Top-constituent input shape:', X_constituents.shape)",
                          "print(f'Top-{MAX_CONSTITUENTS} constituent input shape:', X_constituents.shape)")
    elif cell.get('cell_type') == 'code' and 'model_specs = {' in src:
        src = src.replace("'Top-12 constituent MLP':", "f'Top-{MAX_CONSTITUENTS} constituent MLP':")
        if 'input_kinds =' not in src:
            src = src.replace(
                "fitted_models = {}\n",
                "input_kinds = {\n"
                "    'Logistic (shapes)': 'shapes',\n"
                "    'BDT (shapes)': 'shapes',\n"
                "    'BDT (shapes + kinematics)': 'shapes_plus_kinematics',\n"
                "    f'Top-{MAX_CONSTITUENTS} constituent MLP': 'top_constituents',\n"
                "}\n\n"
                "fitted_models = {}\n",
            )
    elif cell.get('cell_type') == 'code' and 'metrics_df = pd.DataFrame' in src:
        marker = '# Save a reusable baseline bundle for the common comparison notebook.'
        if marker not in src:
            src += f'''

{marker}
baseline_bundle = BASELINE_ROOT / dataset_fingerprint[:12]
baseline_bundle.mkdir(parents=True, exist_ok=True)
joblib.dump({{
    'models': fitted_models,
    'input_kinds': input_kinds,
    'shape_features': shape_features,
    'shape_plus_kinematics': shape_plus_kinematics,
    'max_constituents': MAX_CONSTITUENTS,
}}, baseline_bundle / 'models.joblib')

baseline_config = {{
    'dataset_fingerprint': dataset_fingerprint,
    'split_fingerprint': split_fingerprint,
    'split_seed': RANDOM_STATE,
    'max_constituents': MAX_CONSTITUENTS,
    'model_names': list(test_scores),
    'positive_class': 'quark',
}}
with open(baseline_bundle / 'config.json', 'w') as stream:
    json.dump(baseline_config, stream, indent=2, sort_keys=True)
metrics_df.to_json(baseline_bundle / 'metrics.json', orient='records', indent=2)
np.savez(
    baseline_bundle / 'predictions.npz',
    labels=y_test,
    event_ids=model_data.loc[test_indices, 'event_id'].to_numpy(np.int64),
    jet_ids=model_data.loc[test_indices, 'jet_id'].to_numpy(np.int32),
    model_names=np.asarray(list(test_scores)),
    scores=np.vstack([test_scores[name] for name in test_scores]),
)
print(f'Saved baseline models and common-test predictions: {{baseline_bundle}}')
'''
    cell['source'] = src.splitlines(keepends=True)
path.write_text(json.dumps(nb, indent=1) + "\n")


# Insert durable, high-school-level teaching notes in the hand-authored notebooks.
# Markers make this idempotent and let future runs update rather than duplicate a note.
def upsert_teaching_note(notebook, after_heading, marker, text):
    path = Path(notebook)
    nb = json.loads(path.read_text())
    marker_text = f'<!-- teaching-note:{marker} -->'
    note = md(f'{marker_text}\n\n{text}')
    old_index = next(
        (i for i, cell in enumerate(nb['cells'])
         if marker_text in ''.join(cell.get('source', []))),
        None,
    )
    if old_index is not None:
        nb['cells'][old_index] = note
    else:
        heading_index = next(
            i for i, cell in enumerate(nb['cells'])
            if cell.get('cell_type') == 'markdown'
            and ''.join(cell.get('source', [])).lstrip().startswith(after_heading)
        )
        nb['cells'].insert(heading_index + 1, note)
    path.write_text(json.dumps(nb, indent=1) + "\n")


SAMPLE_NOTES = [
 ('# Quark/gluon jet samples', 'physics-vocabulary', '''## Vocabulary: from a collision to a jet

Protons are made of **partons**—quarks and gluons. In a high-energy proton collision, one
parton from each proton can undergo a short-distance **hard scattering**. Quarks and gluons
carry color charge and cannot be observed alone. They radiate more quarks and gluons in a
**parton shower**, then form color-neutral particles through **hadronization**. A jet
algorithm gathers the resulting nearby particles into a jet.

The “quark jet” or “gluon jet” label therefore refers to the simulated hard parton associated
with a jet, not to a directly observed quark or gluon. Real detector data do not contain this
truth label, which is one reason simulation assumptions must be stated clearly.'''),
 ('## Configure Pythia8', 'monte-carlo', '''### What Pythia is simulating

Pythia is a **Monte Carlo event generator**. Monte Carlo means that it uses random sampling
from physics probability distributions to create possible collision events. One simulated
event is not a prediction by itself; distributions over many events are the prediction.
The random seed makes a run reproducible: the same seed and settings produce the same random
sequence. The hard-QCD setting requests strong-interaction scattering processes.'''),
 ('## Jet definition', 'coordinates-and-jets', '''### Coordinates, acceptance, and the jet radius

The beam defines the longitudinal direction. Transverse momentum $p_T$ is perpendicular to
that beam. Pseudorapidity $\eta$ is a direction coordinate: $\eta=0$ is perpendicular to the
beam and large $|\eta|$ points closer to it. The requirement $|\eta|<2$ is an **acceptance
cut** that keeps jets in a central region where a collider detector can usually measure them
well.

FastJet's anti-$k_t$ algorithm repeatedly combines particles according to a distance rule.
Its radius parameter $R$ controls the typical angular reach of a jet. Angular distance is
$\Delta R=\sqrt{(\Delta\eta)^2+(\Delta\phi)^2}$, where $\phi$ is the angle around the beam.
Truth matching chooses a one-to-one jet–parton assignment with small $\Delta R$; it does not
claim that every particle in the jet came only from that parton.'''),
 ('## Sanity checks', 'quality-assurance', '''### Why sanity checks matter

A **sanity check** tests simple consequences that should hold before any machine learning.
Here we verify the acceptance, uniqueness of assignments, and agreement between inclusive
and flavor-filtered tables. Passing these checks does not prove the simulation is physically
perfect, but failing one reveals a definite bookkeeping or selection error.'''),
 ('## Write to Parquet', 'parquet-format', '''### Why use Parquet?

Parquet is a column-oriented binary data format. It stores column names and types, compresses
repeated structure efficiently, and lets later code read selected columns. Each row here is
one jet; the constituent columns contain lists because jets have different particle counts.
The manifest records settings and a SHA-256 fingerprint so later stages can identify the
exact sample they used.'''),
]

CLASSIFICATION_NOTES = [
 ('# Quark/gluon jet features', 'ml-vocabulary', '''## Machine-learning vocabulary

A **sample** is the collection of jets, and one jet is an **example**. A **feature** is a
number supplied to a model; the **label** is the answer used during supervised training.
A **classifier** learns a score that ranks jets from gluon-like to quark-like. The score is
not automatically a calibrated physical probability.

We fit model parameters using a training set and report performance on unseen examples.
This tests **generalization**: whether the model learned a repeatable pattern rather than
memorizing its training sample.'''),
 ('## B. Engineer', 'feature-engineering', '''### What “feature engineering” means

Feature engineering uses physics knowledge to summarize a variable-length particle list as
a small table of meaningful numbers. Multiplicity describes how many particles there are;
fragmentation observables describe how momentum is shared; angularities describe how widely
energy is spread. Quarks and gluons have different color charges, so gluons tend, on average,
to radiate more strongly. These are statistical tendencies, not rules for every individual
jet.'''),
 ('## D. Build', 'splits-and-preprocessing', '''### Why the split comes before model comparison

Related jets from one simulated collision can share event conditions. If one enters training
and another enters testing, performance can look better than it truly is; this is **data
leakage**. Grouping by event prevents that. **Standardization** uses the training mean and
standard deviation to put features on comparable numerical scales. Test information must not
be used to choose features, tune settings, or standardize inputs.'''),
 ('## E. Train', 'model-families', '''### The model families in this comparison

**Logistic regression** learns a weighted sum of features followed by a sigmoid; it is a
useful, nearly linear baseline. A **boosted decision tree (BDT)** adds many small if/then
trees, with each new tree correcting earlier errors. A **multilayer perceptron (MLP)** is a
neural network of learned linear transformations and nonlinear activation functions.

A **hyperparameter**—such as tree depth, learning rate, or hidden-layer width—is chosen by
the researcher rather than learned as an ordinary weight. Fair tuning uses validation data,
not the final test sample.'''),
 ('## F. Which classifier', 'metrics', '''### How to read classification performance

Choosing a score threshold creates two competing rates. **Quark efficiency** is the fraction
of real quark jets accepted. The **false-positive rate** is the fraction of gluon jets
mistakenly accepted, and gluon rejection is its inverse. The ROC curve shows this tradeoff
for every threshold. **ROC AUC** is the probability that a randomly selected quark jet gets
a higher score than a randomly selected gluon jet: 0.5 is random ranking and 1.0 is perfect
ranking on the evaluated sample.'''),
]

PYTHIA_NOTES = [
 ('# Pythia8 + FastJet', 'simulation-overview', '''## What this demonstration represents

Pythia generates possible proton–proton collisions by random sampling from physics models.
FastJet then clusters the observable final-state particles into jets. A jet is a reconstructed
spray, not a fundamental particle. Repeating many events lets us compare distributions such
as jet momentum, mass, and particle multiplicity.'''),
 ('## Event loop', 'statistics', '''### Why generate many events?

Particle collisions are probabilistic. A histogram from a small sample fluctuates simply
because of chance; this is **statistical uncertainty**. More independent events reduce those
fluctuations, roughly in proportion to $1/\sqrt{N}$ for a count $N$, although systematic
modeling uncertainties do not disappear by generating more events.'''),
]

DISPLAY_NOTES = [
 ('# Dijet event display', 'display-vocabulary', '''## Reading a collider event display

A **dijet event** contains two prominent jets, commonly produced by a hard two-parton
scattering. An event display is a visualization of one collision, not an average prediction.
It helps us build geometric intuition and debug reconstruction, but physics conclusions
require distributions over many events.'''),
 ('## $\\eta$-$\\phi$', 'eta-phi-map', '''### Why use the $\eta$–$\phi$ plane?

Collider detectors are approximately cylindrical around the beam. The azimuth $\phi$ wraps
around the cylinder, while pseudorapidity $\eta$ tracks direction toward either beam. Nearby
particles in this plane have small $\Delta R$ and are likely to be clustered into the same
jet. Marker size or height represents $p_T$, so hard particles stand out visually.'''),
]

for args in SAMPLE_NOTES:
    upsert_teaching_note('demo_quark_gluon_samples.ipynb', *args)
for args in CLASSIFICATION_NOTES:
    upsert_teaching_note('demo_quark_gluon_classification.ipynb', *args)
for args in PYTHIA_NOTES:
    upsert_teaching_note('demo_pythia_fastjet.ipynb', *args)
for args in DISPLAY_NOTES:
    upsert_teaching_note('demo_dijet_event_display.ipynb', *args)


# Repository notebooks are distributed as clean teaching sources, without machine-specific
# output, warnings, timestamps, or execution counters.
for path in Path('.').glob('*.ipynb'):
    nb = json.loads(path.read_text())
    nb['cells'] = [
        cell for cell in nb.get('cells', [])
        if ''.join(cell.get('source', [])).strip()
    ]
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    path.write_text(json.dumps(nb, indent=1) + "\n")
