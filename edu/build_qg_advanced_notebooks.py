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
      code(f'''# Preserve the exact starting point so the architecture-visualization lesson
# can measure what training changed. This roughly doubles checkpoint weight storage.
INITIAL_STATE = {{name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}
history, metrics, predictions = qg.train_model(model, loaders, '{arch}', RUN_MODE, DEVICE)
bundle = qg.save_model_bundle(model, '{arch}', RUN_MODE, MODEL_CONFIG, prepared,
                              history, metrics, predictions, initial_state=INITIAL_STATE)
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
      md('''### How to read these figures

**Left — training history.** The horizontal axis counts epochs, or complete passes through
the training sample. The vertical axis is binary cross-entropy (BCE) loss; lower means the
training predictions agree better with their labels. Look for a rapid early decrease followed
by a slower change. A falling training loss alone does **not** prove that the model works on
new jets—the held-out test result on the right is the important check. **Held out** means
these test jets were set aside before training: their labels were not used to adjust weights,
choose hyperparameters, or decide when to stop.

**Right — classifier performance.** Each point corresponds to a different threshold on the
model's quark-like score. Quark efficiency is the fraction of true quark jets kept. Gluon
rejection is $1/(\text{fraction of gluon jets kept})$, so larger is better: rejection 10 means
only one gluon jet in ten passes. The vertical axis is logarithmic, so equal vertical steps
represent multiplication, not addition. A curve closer to the upper-right corner is better,
but compare models at the quark efficiency needed for the physics question. AUC summarizes
the whole ranking; 0.5 is random and 1.0 is perfect on this test sample.'''),
      md('''## 5. Reading the result

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
 md('''### How to read these figures

**Left — constituent multiplicity.** The horizontal axis is the number of reconstructed
particles in one jet; the vertical axis counts jets. The width of this distribution explains
why a fixed-size table is awkward: different jets contain different amounts of information.
Notice both the typical multiplicity and the long tail. Dynamic padding lets a batch grow only
to its longest jet, while the mask prevents padding entries from being treated as particles.

**Right — raw feature means.** Each bar is the arithmetic mean, measured using training
constituents only, before normalization. `log(z)` describes a particle's share of jet momentum;
`deta` and `dphi` locate it relative to the jet axis; `log_dr` describes radial distance.
Negative bars are normal for logarithms of fractions or small distances. The bars are on
different physical scales, so their heights do not measure feature importance. These means
are preprocessing constants that will be subtracted from the corresponding inputs.'''),
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
pair's geometric separation. Here that separation is calculated directly from each pair's
$\\Delta\\eta$ and $\\Delta\\phi$. This keeps a particle's distance from itself exactly zero and
prevents masked padding entries from changing the answer through floating-point round-off.
There is no positional sequence encoding because a jet is an unordered particle set, not a
sentence.'''))
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
 md('''### How to read this figure

Every curve uses the **same held-out test sample**: jets set aside before training whose
labels did not adjust model weights, choose settings, or trigger early stopping. Therefore
differences are attributable to the trained models rather than to easier or harder test
examples. Moving right keeps more true quark jets
(quark efficiency). Moving up rejects more gluon jets; rejection 20 means that only about
$1/20=5\%$ of gluon jets pass. Because the vertical axis is logarithmic, a change from 5 to
10 is as large multiplicatively as a change from 10 to 20.

Look first at the efficiency region relevant to the intended analysis: curves can cross, so
the model with the largest AUC need not be best at every operating point. The AUC in each
legend entry summarizes the full curve (0.5 random, 1.0 perfect). Small separations should not
be over-interpreted without statistical uncertainties or another independent test sample.'''),
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
 md('''### How to read this figure

Each color is one trained architecture, and each group on the horizontal axis is an input
ablation. The vertical value is `AUC after removal − original AUC`. A bar below zero means
performance became worse when that information was removed; a more negative bar therefore
suggests stronger model reliance. A bar near zero means little measured change, while a
positive bar can occur through statistical fluctuation or because the removal suppressed a
pattern that did not generalize.

Compare the *pattern* of bars between models. Do not read a bar as “this variable causes the
jet to be a quark”: correlated inputs may replace one another, and an ablation may create
unrealistic jets the model never saw during training.'''),
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
 md('''### How to read these figures

Each panel shows the same selected jet as seen by one architecture. A point is a constituent;
its horizontal and vertical coordinates are $\Delta\eta$ and wrapped $\Delta\phi$ relative
to the jet axis, so the origin is the jet center. Larger, brighter points have larger
integrated-gradient attribution for that model's score. The color is normalized separately
inside each panel: compare *where* a model concentrates attention, not the absolute color
value between panels.

Ask whether influential constituents lie in the hard core, at wide angle, or in several
clusters. This is one local example and the plot uses attribution magnitude, so it does not
show whether a constituent pushed the score toward quark or toward gluon. A population-level
claim requires repeating the study over many jets.'''),
 md('''## 4. Interpretation limits

Large ablation losses identify reliance, not a uniquely important physical variable. Inputs
are correlated; removing them can create **out-of-distribution** jets—artificial inputs unlike
anything seen during training. Generator truth labels are
idealized and process dependent. Attention weights and graph edges describe internal routing,
not causal explanations. Reliable scientific conclusions should be checked with several
generators, detector conditions, kinematic regions, and interpretation methods.''')])


write('demo_quark_gluon_basic_model_visualization.ipynb',[
 md('''# Seeing how the basic quark/gluon classifiers work

This notebook visualizes the four models trained in the basic classification lesson:
logistic regression, two boosted decision trees (BDTs), and the top-20 constituent
multilayer perceptron (MLP). The first half explains their structure without requiring
trained files. The second half optionally opens the saved baseline bundle and looks inside
the fitted models.

These models do not all “learn weights” in the same way. Logistic regression learns one
coefficient per input feature. A BDT grows a sequence of small decision trees. An MLP learns
matrices of connections between layers. We therefore visualize each model using a
representation that matches its mathematics.'''),
 md('''## 0. Student controls

Use the same Parquet sample as the basic classification notebook. TREE_TO_DRAW = 0 displays
the first tree in each boosted ensemble; try a later number below 250 to see a correction
learned later in training. MATRIX_SIDE limits large MLP matrices to a readable corner.'''),
 code(SOURCE_SETTINGS + r'''
TREE_TO_DRAW = 0
MATRIX_SIDE = 40
'''),
 md('## 1. Environment'),
 code(BOOT + r'''
import joblib
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.tree import plot_tree
'''),
 md('''## 2. Architecture maps before training

An **architecture** specifies the form of a model before it has learned from examples.
Engineered shape features are human-designed summaries such as multiplicity, width, and
momentum sharing. Standardization subtracts each training mean and divides by its standard
deviation, putting differently scaled variables on comparable numerical scales.

The two BDTs have the same learning algorithm. One uses only jet-shape variables; the
diagnostic version also receives jet $p_T$ and $|\eta|$. The MLP receives a flattened,
$p_T$-ordered list of at most 20 constituents, including masks for missing positions.'''),
 code('''def draw_pipeline(ax, title, labels, colors):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.set_title(title, weight='bold')
    width = 0.16
    xs = np.linspace(0.02, 0.98-width, len(labels))
    for i, (x, label, color) in enumerate(zip(xs, labels, colors)):
        box = FancyBboxPatch((x,.34),width,.32,boxstyle='round,pad=0.015',
                             facecolor=color,edgecolor='#263238',linewidth=1.2)
        ax.add_patch(box); ax.text(x+width/2,.50,label,ha='center',va='center',fontsize=9)
        if i:
            ax.annotate('',xy=(x-.006,.50),xytext=(xs[i-1]+width+.006,.50),
                        arrowprops=dict(arrowstyle='->',lw=1.5,color='#455a64'))

flows = {
 'Logistic regression': (
   ['engineered\\nshapes','standardize','weighted\\nsum','sigmoid','quark-like\\nscore'],
   ['#bbdefb','#fff59d','#a5d6a7','#ffccbc','#ffccbc']),
 'Boosted decision tree (two input choices)': (
   ['engineered\\nfeatures','small decision\\ntree','add correcting\\ntrees','sum tree\\noutputs','quark-like\\nscore'],
   ['#bbdefb','#80cbc4','#80cbc4','#a5d6a7','#ffccbc']),
 'Top-20 constituent MLP': (
   ['sorted + padded\\nparticles','standardize','64-neuron\\nlayer','32-neuron\\nlayer','quark-like\\nscore'],
   ['#bbdefb','#fff59d','#ce93d8','#ce93d8','#ffccbc'])}
fig,axes=plt.subplots(3,1,figsize=(13,7))
for ax,(title,(labels,colors)) in zip(axes,flows.items()): draw_pipeline(ax,title,labels,colors)
plt.tight_layout(); plt.show()'''),
 md('''### What to look for

Logistic regression has only one learned step: a weighted sum. Its decision boundary is a
flat surface in standardized feature space. A decision tree follows if/then branches, and
boosting adds many trees so later trees correct errors left by earlier ones. The MLP alternates
weighted sums with nonlinear activation functions, allowing curved decision boundaries.

“Basic” does not mean useless. Simpler models can train well on smaller samples, run quickly,
and be easier to diagnose. The BDT is often an especially strong baseline for engineered
tabular features.'''),
 md('''## 3. Load the trained baseline bundle, if available

The filename is selected using the exact SHA-256 fingerprint of the source sample. This keeps
us from silently inspecting a model trained on different jets. A **pipeline** packages
preprocessing and the classifier together so the same standardization is applied at training
and inference time.'''),
 code('''baseline_bundle = None; baseline_models = {}; baseline_payload = None
if SOURCE.exists():
    fingerprint = qg.sha256_file(SOURCE)
    candidate = Path('artifacts/qg_baselines')/fingerprint[:12]
    if (candidate/'models.joblib').exists():
        baseline_bundle = candidate
        config = json.loads((candidate/'config.json').read_text())
        assert config['dataset_fingerprint'] == fingerprint
        baseline_payload = joblib.load(candidate/'models.joblib')
        baseline_models = baseline_payload['models']
        print(f'Loaded {len(baseline_models)} models from {candidate}')
    else:
        print(f'No baseline bundle at {candidate}. Run the basic classification notebook first.')
else:
    print(f'{SOURCE} does not exist. The architecture-map lesson above is still complete.')'''),
 md('''## 4. Compare the size of the fitted models

Model size has different meanings across families. Logistic regression and the MLP store
numeric parameters. For a BDT, nodes are branch points or leaves across all trees. These
counts describe complexity and memory, not performance; the common evaluation notebook is
the correct place to compare classification quality.'''),
 code('''summary_rows=[]
for name,model in baseline_models.items():
    if 'Logistic' in name:
        estimator=model.named_steps['logisticregression']
        summary_rows.append({'model':name,'learned objects':'coefficients',
                             'count':estimator.coef_.size+estimator.intercept_.size,
                             'structure':f'{estimator.coef_.shape[1]} inputs → 1 output'})
    elif 'BDT' in name:
        trees=[tree[0] for tree in model.estimators_]
        summary_rows.append({'model':name,'learned objects':'tree nodes',
                             'count':sum(tree.tree_.node_count for tree in trees),
                             'structure':f'{len(trees)} trees; maximum depth {model.max_depth}'})
    elif 'MLP' in name:
        estimator=model.named_steps['mlpclassifier']
        count=sum(w.size for w in estimator.coefs_)+sum(b.size for b in estimator.intercepts_)
        layers=[estimator.coefs_[0].shape[0]]+list(estimator.hidden_layer_sizes)+[estimator.n_outputs_]
        summary_rows.append({'model':name,'learned objects':'weights + biases',
                             'count':count,'structure':' → '.join(map(str,layers))})
if summary_rows: display(pd.DataFrame(summary_rows))
else: print('No trained models to summarize.')'''),
 md('''## 5. Logistic regression: one coefficient per shape variable

Because inputs were standardized, coefficient magnitudes are reasonably comparable within
this model. A positive coefficient pushes the logit toward the positive class—quark here—and
a negative coefficient pushes toward gluon, while all other standardized inputs are held
fixed. Correlated inputs can share or exchange influence, so this is a model description,
not proof that a variable physically causes a jet's label.'''),
 code('''logistic_name='Logistic (shapes)'
if logistic_name in baseline_models:
    pipeline=baseline_models[logistic_name]
    estimator=pipeline.named_steps['logisticregression']
    names=baseline_payload['shape_features']; coefficients=estimator.coef_[0]
    order=np.argsort(np.abs(coefficients))
    fig,ax=plt.subplots(figsize=(8,5))
    ax.barh(np.asarray(names)[order],coefficients[order],
            color=np.where(coefficients[order]>=0,'#ef6c00','#1976d2'))
    ax.axvline(0,color='black',lw=.8)
    ax.set(xlabel='coefficient in standardized feature space',
           title='Learned logistic-regression coefficients')
    plt.tight_layout(); plt.show()
else: print('Trained logistic regression not available.')'''),
 md('''### How to read this figure

Longer bars change the model's logit more for a one-standard-deviation input change. Orange
points toward the quark class and blue toward gluon under the model's convention. Do not
compare these coefficient numbers directly with neural-network weights: a neural feature is
repeatedly transformed, whereas each logistic coefficient acts directly on a named input.'''),
 md('''## 6. BDTs: inspect one correcting tree

Each displayed tree is only one member of a 250-tree ensemble. Start at the top node. If its
condition is true, follow the left branch; otherwise follow the right. A leaf contains that
tree's numerical correction to the ensemble score. Early trees usually capture broad
patterns; later trees focus on remaining errors.'''),
 code('''bdt_items=[(name,model) for name,model in baseline_models.items() if 'BDT' in name]
if bdt_items:
    fig,axes=plt.subplots(1,len(bdt_items),figsize=(9*len(bdt_items),5),squeeze=False)
    for ax,(name,model) in zip(axes[0],bdt_items):
        tree_index=min(max(int(TREE_TO_DRAW),0),len(model.estimators_)-1)
        features=(baseline_payload['shape_plus_kinematics'] if 'kinematics' in name
                  else baseline_payload['shape_features'])
        plot_tree(model.estimators_[tree_index,0],feature_names=features,filled=True,
                  rounded=True,precision=2,fontsize=8,ax=ax)
        ax.set_title(f'{name}\\nboosting tree {tree_index}')
    plt.tight_layout(); plt.show()
else: print('Trained BDTs not available.')'''),
 md('''### How to read these trees

The top box is the root decision. The feature-and-threshold test divides the jets reaching
that node. The color and value describe the correction learned there; they are not
probabilities from the complete classifier. Follow one path to a bottom leaf to see a compact
sequence of rules. The two BDTs may choose different splits because one can also use jet
kinematics.

Never interpret a single tree as the full decision. The final BDT score adds the initial
prediction and the small corrections from all 250 trees.'''),
 md('''## 7. How the boosted ensemble grows

Unlike an MLP, a gradient-boosted tree model does not repeatedly update one fixed collection
of connection matrices. Training appends trees. This plot shows the structural complexity of
each successive correction. Tree depth was deliberately capped at two, limiting how many
feature conditions can interact inside one tree.'''),
 code('''if bdt_items:
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,model in bdt_items:
        trees=[tree[0].tree_ for tree in model.estimators_]
        axes[0].plot([tree.node_count for tree in trees],label=name,alpha=.8)
        axes[1].plot([tree.n_leaves for tree in trees],label=name,alpha=.8)
    axes[0].set(xlabel='boosting round',ylabel='nodes in added tree',title='Tree size through training')
    axes[1].set(xlabel='boosting round',ylabel='leaves in added tree',title='Terminal regions through training')
    for ax in axes: ax.legend()
    plt.tight_layout(); plt.show()
else: print('Trained BDTs not available.')'''),
 md('''### How to read these figures

One point is one newly added tree. More nodes and leaves mean that correction divides feature
space into more regions, but not necessarily that it contributes more to final accuracy.
Nearly constant values are expected because maximum depth two strongly restricts every tree.
To see whether later rounds improved validation performance we would need staged predictions,
not just structural counts.'''),
 md('''## 8. MLP: learned connection matrices

An MLP layer connects every input number to every neuron in the next layer. A matrix cell is
one connection weight. The fitted top-20 model has an input-to-64 matrix, a 64-to-32 matrix,
and a 32-to-output matrix. Only a readable corner of a large matrix is drawn; the full matrix
is still used by the classifier.'''),
 code('''mlp_names=[name for name in baseline_models if 'MLP' in name]
if mlp_names:
    pipeline=baseline_models[mlp_names[0]]; estimator=pipeline.named_steps['mlpclassifier']
    matrices=estimator.coefs_
    fig,axes=plt.subplots(1,len(matrices),figsize=(5*len(matrices),4),squeeze=False)
    for i,(ax,weights) in enumerate(zip(axes[0],matrices)):
        view=weights[:MATRIX_SIDE,:MATRIX_SIDE]
        limit=max(abs(view).max(),1e-12)
        image=ax.imshow(view,cmap='coolwarm',vmin=-limit,vmax=limit,aspect='auto')
        ax.set(title=f'layer {i}: {weights.shape[0]} × {weights.shape[1]}',
               xlabel='next-layer neuron',ylabel='previous-layer input')
        fig.colorbar(image,ax=ax,fraction=.046)
    plt.tight_layout(); plt.show()

    magnitudes=pd.DataFrame([
        {'connection':f'{w.shape[0]} → {w.shape[1]}','mean |weight|':np.mean(np.abs(w)),
         'RMS weight':np.sqrt(np.mean(w**2))} for w in matrices])
    display(magnitudes)
else: print('Trained constituent MLP not available.')'''),
 md('''### How to read these matrices

Red and blue show opposite signs; pale cells are closer to zero. A strong individual
connection is not automatically important because its source neuron may rarely activate,
and later layers can cancel or amplify it. The first matrix's rows correspond to standardized
flattened constituent inputs; hidden-layer rows and columns are learned coordinates without
simple physics names.

The current baseline bundle stores the fitted MLP but not its exact random initialization, so
this notebook does not pretend that a newly randomized matrix is its true “before” state.
Exact before/after tracking is provided for the PyTorch constituent networks in the companion
architecture notebook.'''),
 md('''## 9. What these pictures establish

The visualizations answer structural questions: which inputs enter, how decisions are
assembled, how large the fitted objects are, and what numerical patterns were learned. They
do not determine which classifier performs best; use the common held-out evaluation notebook
for that. They also do not turn correlations into causes. Feature importance should be checked
with several methods, independent samples, and physics knowledge.''')])


write('demo_quark_gluon_architecture_visualization.ipynb',[
 md('''# Seeing how the constituent networks are built—and how training changes them

This notebook turns the PFN, Particle Transformer, and ParticleNet-style network into
visual maps. The first half needs no trained model: it shows the route information takes
from jet constituents to one quark-like score. The second half optionally loads saved
checkpoints and compares their parameters and internal activations with their starting
points.

A **parameter** (often called a weight) is an adjustable number inside a neural network.
A **block** is a named group of layers that performs one stage of the calculation. An
**activation** is the intermediate array of numbers produced by a layer for a particular
input. Architecture tells us which calculations are possible; training chooses parameter
values that make those calculations useful.'''),
 md('''## 0. Student controls

Choose the same sample and quick/full mode used to train the checkpoints you want to inspect.
The conceptual diagrams and untrained models work even if the Parquet file or checkpoints do
not exist. The trained sections skip missing models instead of failing.'''),
 code(SOURCE_SETTINGS + "\nMODEL_MODE = 'quick'  # inspect saved 'quick' or 'full' checkpoints"),
 md('## 1. Environment'), code(BOOT + "\nimport pandas as pd\nfrom matplotlib.patches import FancyBboxPatch"),
 md('''## 2. Three information-flow maps

The arrows show the main direction of computation, not every tensor operation. A shared
layer applies the **same learned rule** to every constituent. **Pooling** combines a
variable-size particle set into a fixed-size jet summary. **Self-attention** lets particles
exchange information with all other particles. A **nearest-neighbor graph** instead connects
each particle to a small local neighborhood.

All three networks end in a single **logit**. The sigmoid function turns that unrestricted
number into a score between 0 and 1. These are conceptual maps; the executable PyTorch models
remain the authoritative definitions.'''),
 code('''def draw_pipeline(ax, title, labels, colors):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.set_title(title, weight='bold')
    width = 0.145 if len(labels) >= 6 else 0.17
    xs = np.linspace(0.02, 0.98 - width, len(labels))
    for i, (x, label, color) in enumerate(zip(xs, labels, colors)):
        box = FancyBboxPatch((x, .35), width, .30, boxstyle='round,pad=0.015',
                             facecolor=color, edgecolor='#263238', linewidth=1.2)
        ax.add_patch(box); ax.text(x + width/2, .50, label, ha='center', va='center', fontsize=9)
        if i:
            ax.annotate('', xy=(x-.006, .50), xytext=(xs[i-1]+width+.006, .50),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='#455a64'))

flows = {
 'PFN': (['particle\\nfeatures', 'shared Φ\\nnetwork', 'masked\\nsum', 'ρ\\nnetwork', 'jet\\nlogit'],
         ['#bbdefb','#90caf9','#fff59d','#a5d6a7','#ffccbc']),
 'Particle Transformer': (['particle\\nfeatures', 'embedding', 'self-attention\\nblocks', 'class-query\\nattention', 'output\\nhead', 'jet\\nlogit'],
         ['#bbdefb','#90caf9','#ce93d8','#ffe082','#a5d6a7','#ffccbc']),
 'ParticleNet style': (['particle\\ncloud', 'nearest-\\nneighbor graph', 'EdgeConv\\nblocks', 'mean + max\\npooling', 'output\\nhead', 'jet\\nlogit'],
         ['#bbdefb','#80cbc4','#80cbc4','#fff59d','#a5d6a7','#ffccbc'])}
fig, axes = plt.subplots(3, 1, figsize=(13, 7))
for ax, (title, (labels, colors)) in zip(axes, flows.items()): draw_pipeline(ax, title, labels, colors)
plt.tight_layout(); plt.show()'''),
 md('''### What to look for

The PFN communicates between constituents only when their learned vectors are summed. It is
the simplest and cheapest of the three designs. The Transformer performs all-to-all
comparisons, so it can learn relationships between distant constituents but its work grows
rapidly with multiplicity. ParticleNet builds changing local neighborhoods and learns from
nearby pairs. The yellow pooling stages are information bottlenecks: after them, a
variable-length jet has become one fixed-length vector.

The diagram does not rank the models. A more elaborate route can represent more patterns,
but can also need more data, memory, and regularization.'''),
 md('''## 3. Count parameters inside each untrained block

Here **untrained** means PyTorch has assigned random initial parameter values but no jet label
has adjusted them. Parameter count measures capacity and storage, not intelligence. A block
with more parameters has more adjustable numbers, but parameter-free operations such as a
sum or nearest-neighbor lookup can still be essential.'''),
 code('''if MODEL_MODE not in {'quick', 'full'}: raise ValueError("MODEL_MODE must be 'quick' or 'full'")
architectures = ('pfn', 'transformer', 'particlenet')
fresh_models = {name: qg.create_model(name, config=qg.default_config(name, MODEL_MODE))
                for name in architectures}
parameter_rows = []
for architecture, model in fresh_models.items():
    children = dict(model.named_children())
    for block, module in children.items():
        parameter_rows.append({'architecture': architecture, 'block': block,
                               'parameters': sum(p.numel() for p in module.parameters())})
    direct = sum(p.numel() for p in model.parameters(recurse=False))
    if direct: parameter_rows.append({'architecture': architecture, 'block': 'direct parameter', 'parameters': direct})
parameter_table = pd.DataFrame(parameter_rows)
display(parameter_table)
display(parameter_table.groupby('architecture', as_index=False)['parameters'].sum())

pivot = parameter_table.pivot(index='architecture', columns='block', values='parameters').fillna(0)
pivot.plot.bar(stacked=True, figsize=(10, 5), colormap='tab20')
plt.ylabel('trainable parameters'); plt.title(f'Where the {MODEL_MODE}-mode parameters live')
plt.xticks(rotation=0); plt.tight_layout(); plt.show()'''),
 md('''### How to read this figure

Each full bar is one architecture; its height is the total number of trainable values. Colored
segments locate those values in named blocks. The scales can differ greatly, so compare both
total height and composition. A zero-parameter operation will not appear even when it is
structurally important. Also remember that a parameter is reused every time its layer is
applied—for example, PFN's $\\Phi$ parameters process every particle.'''),
 md('''## 4. Find compatible trained checkpoints

A **checkpoint** is a saved set of parameter values. The sample fingerprint ensures that this
notebook does not silently compare models trained on different generated data. New training
runs also save `initial_weights.pt`, the exact state immediately before optimization. Older
bundles remain usable, but for them we can compare only against a newly randomized reference,
not claim an exact before/after measurement.'''),
 code('''prepared = None; trained_bundles = {}
if SOURCE.exists():
    prepared = qg.prepare_dataset(SOURCE); manifest = qg.load_manifest(prepared)
    for bundle in qg.discover_bundles(dataset_fingerprint=manifest['source_sha256']):
        config = json.loads((bundle/'config.json').read_text())
        if config['mode'] == MODEL_MODE: trained_bundles[config['architecture']] = bundle
else:
    print(f'{SOURCE} does not exist; the untrained architecture lesson is still complete.')

if trained_bundles:
    display(pd.DataFrame([{'architecture': name, 'bundle': str(bundle),
                           'exact_initial_state': (bundle/'initial_weights.pt').exists()}
                          for name, bundle in trained_bundles.items()]))
else:
    print(f'No compatible {MODEL_MODE!r} checkpoints found. Train an architecture notebook, then rerun from here.')'''),
 md('''## 5. How much did each block move?

For every block we compute

$$\\text{relative change}=\\frac{\\|W_{trained}-W_{reference}\\|_2}{\\|W_{reference}\\|_2}.$$

The $L_2$ norm combines all numbers in a block into one geometric length. A value near zero
means little movement relative to the starting scale; a value near one means the update has a
similar overall size to the reference. This is **not feature importance** and does not say
whether a block improved the classifier. Optimizers can rescale connected layers while
preserving similar predictions.'''),
 code('''def reference_state(bundle, config):
    initial_path = bundle/'initial_weights.pt'
    if initial_path.exists():
        return torch.load(initial_path, map_location='cpu', weights_only=True), 'exact initialization'
    # A useful visual baseline for old bundles, but not their actual starting point.
    torch.manual_seed(12345)
    reference = qg.create_model(config['architecture'], config['input_dim'], config['model_config'])
    return reference.state_dict(), 'fresh random reference'

change_rows = []; states = {}
for architecture, bundle in trained_bundles.items():
    config = json.loads((bundle/'config.json').read_text())
    trained = torch.load(bundle/'weights.pt', map_location='cpu', weights_only=True)
    reference, basis = reference_state(bundle, config); states[architecture] = (reference, trained, basis, config)
    parameter_names = {name for name, _ in qg.create_model(
        architecture, config['input_dim'], config['model_config']).named_parameters()}
    groups = {}
    for name in parameter_names:
        group = name.split('.')[0]
        before = reference[name].float(); after = trained[name].float()
        sums = groups.setdefault(group, [0.0, 0.0, 0])
        sums[0] += torch.sum((after-before)**2).item()
        sums[1] += torch.sum(before**2).item(); sums[2] += before.numel()
    for group, (delta2, reference2, count) in groups.items():
        change_rows.append({'architecture': architecture, 'block': group, 'basis': basis,
                            'relative_L2_difference': np.sqrt(delta2/max(reference2, 1e-24)),
                            'parameters': count})

if change_rows:
    changes = pd.DataFrame(change_rows); display(changes)
    labels = changes['architecture'] + ': ' + changes['block']
    color_map = {'pfn':'#1976d2', 'transformer':'#7b1fa2', 'particlenet':'#00796b'}
    fig, ax = plt.subplots(figsize=(10, max(4, .42*len(changes))))
    ax.barh(labels, changes['relative_L2_difference'], color=[color_map[x] for x in changes['architecture']])
    ax.set_xscale('log'); ax.set_xlabel('relative L2 difference (log scale)'); ax.invert_yaxis()
    ax.set_title('Block-level parameter movement'); plt.tight_layout(); plt.show()
else: print('No trained checkpoints to compare.')'''),
 md('''### How to read this figure

Read the table's `basis` column first. “Exact initialization” supports a true statement about
what changed during that run. “Fresh random reference” only shows how the trained checkpoint
differs from a typical untrained model. The logarithmic horizontal scale makes both small and
large differences visible. Compare blocks within a model cautiously; LayerNorm parameters,
biases, and large matrices begin on different scales.'''),
 md('''## 6. Look inside one weight matrix

For checkpoints with an exact saved initialization, the next figure shows a small window from
the first matrix-shaped parameter. Each colored square is one number. The difference panel
uses its own color scale because training updates are often much smaller than initialized
weights. Rows and columns are learned coordinates, not named physics variables, so patterns
here are diagnostic—not a direct explanation of quark/gluon physics.'''),
 code('''exact = [(name, values) for name, values in states.items() if values[2] == 'exact initialization']
if exact:
    fig, axes = plt.subplots(len(exact), 3, figsize=(12, 3.4*len(exact)), squeeze=False)
    for row, (architecture, (reference, trained, basis, config)) in enumerate(exact):
        key = next(k for k, value in trained.items() if value.ndim == 2 and value.is_floating_point())
        before = reference[key].float().numpy()[:32,:32]; after = trained[key].float().numpy()[:32,:32]
        scale = max(abs(before).max(), abs(after).max(), 1e-12)
        for ax, array, title, limit in zip(axes[row], (before, after, after-before),
                                          ('initial', 'trained', 'difference'),
                                          (scale, scale, max(abs(after-before).max(),1e-12))):
            image = ax.imshow(array, cmap='coolwarm', vmin=-limit, vmax=limit, aspect='auto')
            ax.set_title(f'{architecture}: {key} — {title}'); ax.set(xlabel='input index', ylabel='output index')
            fig.colorbar(image, ax=ax, fraction=.046)
    plt.tight_layout(); plt.show()
else: print('No exact initialization files yet. Retrain a model with the updated training notebook to enable this figure.')'''),
 md('''## 7. Compare internal activations on held-out jets

Parameters describe the model, while activations describe what it computes for actual inputs.
Hooks record outputs of the main blocks without changing the forward calculation. We use one
batch from the held-out test split: these jets never updated the weights or controlled early
stopping. RMS (root-mean-square) summarizes activation size; it is not accuracy or importance.'''),
 code('''def logical_blocks(model, architecture):
    if architecture == 'pfn': return [('phi', model.phi), ('rho', model.rho)]
    if architecture == 'transformer':
        return [('embed', model.embed)] + [(f'attention block {i+1}', block) for i, block in enumerate(model.blocks)] + [('class attention', model.cls_attn), ('head', model.head)]
    return [('EdgeConv 1', model.edge1), ('EdgeConv 2', model.edge2), ('head', model.head)]

def activation_rms(model, architecture, batch):
    found = {}; handles = []
    def make_hook(label):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            found[label] = float(value.detach().float().square().mean().sqrt().cpu())
        return hook
    for label, module in logical_blocks(model, architecture): handles.append(module.register_forward_hook(make_hook(label)))
    model.eval()
    with torch.no_grad(): model(batch['features'].to(DEVICE), batch['coords'].to(DEVICE), batch['mask'].to(DEVICE))
    for handle in handles: handle.remove()
    return found

activation_rows = []
if prepared is not None:
  for architecture, (reference, trained, basis, config) in states.items():
    batch = next(iter(qg.make_loaders(prepared, architecture, config['mode'])[2]))
    for stage, state in [('reference', reference), ('trained', trained)]:
        model = qg.create_model(architecture, config['input_dim'], config['model_config'])
        model.load_state_dict(state); model.to(DEVICE)
        for block, rms in activation_rms(model, architecture, batch).items():
            activation_rows.append({'architecture':architecture, 'block':block, 'stage':stage,
                                    'activation_RMS':rms, 'basis':basis})

if activation_rows:
    activations = pd.DataFrame(activation_rows)
    names = list(activations['architecture'].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(5*len(names), 4), squeeze=False)
    for ax, architecture in zip(axes[0], names):
        view = activations[activations.architecture == architecture].pivot(index='block', columns='stage', values='activation_RMS')
        view.plot.bar(ax=ax, color=['#90caf9','#ef6c00']); ax.set_title(architecture)
        ax.set_ylabel('activation RMS'); ax.tick_params(axis='x', rotation=35)
    plt.tight_layout(); plt.show()
else: print('Train a compatible model and provide its source Parquet file to compare activations.')'''),
 md('''### How to read these figures—and what they cannot prove

Within one panel, compare the paired reference and trained bars at each block. A change means
training altered the numerical representation of this held-out batch. Different blocks have
different widths and nonlinearities, so their RMS heights should not be ranked as importance.
An unchanged scale can also hide a major rotation or rearrangement of the representation.

Weight and activation views are excellent checks that learning occurred and useful clues when
debugging vanishing or exploding values. They do **not** by themselves reveal which physical
features drive performance. Use the separate interpretation notebook's ablations and local
attributions for that question, and still treat those as model-reliance evidence rather than
causal laws of nature.''')])


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


def upsert_figure_note(notebook, after_code, marker, text):
    """Place a durable explanation immediately after the code that draws a figure."""
    path = Path(notebook)
    nb = json.loads(path.read_text())
    marker_text = f'<!-- figure-guide:{marker} -->'
    note = md(f'{marker_text}\n\n{text}')
    old_index = next(
        (i for i, cell in enumerate(nb['cells'])
         if marker_text in ''.join(cell.get('source', []))),
        None,
    )
    if old_index is not None:
        nb['cells'].pop(old_index)
    code_index = next(
        i for i, cell in enumerate(nb['cells'])
        if cell.get('cell_type') == 'code'
        and after_code in ''.join(cell.get('source', []))
    )
    nb['cells'].insert(code_index + 1, note)
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
be used to choose features, tune settings, or standardize inputs.

The **training sample** is used to fit model parameters. The **validation sample** is separate
and is used for choices such as hyperparameters or early stopping. The **test sample** is
**held out**—kept untouched until the final evaluation—to imitate genuinely unseen data.
Repeatedly changing a model after looking at its test result leaks test information into the
design process; a fresh test sample would then be needed for an unbiased final claim.'''),
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


FIGURE_NOTES = [
 ('demo_pythia_fastjet.ipynb', "plt.savefig('demo_pythia_fastjet.png'", 'inclusive-jet-summary',
  '''### How to read these figures

**Left — jet transverse momentum.** $p_T$ is momentum perpendicular to the beam. The
leading jet has the largest $p_T$ in an event and the subleading jet has the second largest.
The vertical axis is logarithmic: equal vertical steps represent multiplication, which lets
rare high-$p_T$ jets remain visible. Look for the rapidly falling spectrum and for the leading
curve to extend farther than the subleading curve.

**Middle — jet pseudorapidity.** $\eta=0$ is perpendicular to the beam and larger $|\eta|$
points more nearly along it. A roughly left-right-symmetric distribution is expected because
the two proton beams are equivalent. This plot includes all accepted jets, not one entry per
event.

**Right — jet multiplicity.** The horizontal axis counts reconstructed jets in one collision;
the vertical axis counts events. Two hard jets are common in this selected process, while
extra jets can arise from additional radiation. Do not confuse jet multiplicity with the
number of particles inside a jet.'''),
 ('demo_quark_gluon_samples.ipynb', "plt.savefig('demo_quark_gluon_samples.png'", 'sample-sanity-plots',
  '''### How to read these figures

**Left — $p_T$ by truth label.** Each curve is normalized to unit area, so compare shapes,
not the total number of jets. Ideally the quark and gluon $p_T$ shapes overlap closely:
otherwise a classifier could use kinematics as a shortcut instead of learning internal jet
structure. “Unmatched” means no accepted hard parton passed the one-to-one matching rule; it
does not mean that the jet has no physical origin.

**Right — constituent multiplicity.** A constituent is one final-state particle clustered
into the jet. The horizontal axis counts those particles and the normalized vertical axis
shows how common each count is within a flavor. Gluon jets often contain more constituents
on average because gluons radiate more strongly. The distributions still overlap, so
multiplicity is useful statistical evidence, not a perfect label for an individual jet.'''),
 ('demo_quark_gluon_classification.ipynb', "title='Accepted pseudorapidity'", 'kinematic-shortcuts',
  '''### How to read these figures

Both panels compare **normalized densities**, so the area under each flavor curve is one.
The left panel shows jet $p_T$; separation here is a warning that a classifier might learn a
kinematic shortcut caused by sample selection. The right shows pseudorapidity $\eta$, where
$\eta=0$ is transverse to the beam and the edges at $\pm2$ come from the acceptance cut.

Look for overlap between the blue quark and red gluon curves. These plots diagnose whether
the samples occupy similar kinematic regions; they are not yet measurements of classifier
performance.'''),
 ('demo_quark_gluon_classification.ipynb', "plt.savefig('demo_quark_gluon_feature_distributions.png'", 'engineered-features',
  '''### How to read these figures

Every panel overlays normalized quark and gluon distributions for one engineered observable.
Where the curves separate, that observable may help classification; where they overlap, it
has little power by itself. A model can still gain from combinations and correlations.

`n_constituents` counts particles; jet mass measures invariant mass; $m/p_T$ removes much of
the overall momentum scale. $p_T^D$ and the leading fraction describe whether momentum is
concentrated in a few hard particles or shared among many. Les Houches angularity, girth, and
radial width describe how broad the radiation is around the jet axis. The EEC-like moment
summarizes momentum-weighted particle-pair separations. Gluons tend to be broader and more
populated on average, but the overlap shows why no single variable decides every jet.'''),
 ('demo_quark_gluon_classification.ipynb', "plt.savefig('demo_quark_gluon_classifier_comparison.png'", 'baseline-performance',
  '''### How to read these figures

**Left — operating-point tradeoff.** Moving right accepts a larger fraction of true quark
jets. Moving up rejects more gluons; rejection 10 means one gluon in ten survives. The
vertical axis is logarithmic. Curves may cross, so compare models at the quark efficiency
needed by the analysis rather than declaring a winner from one isolated point. These curves
use a **held-out test sample**: jets kept completely separate while the models learned and
while settings were chosen, so they provide an honest test on unseen examples.

**Right — ROC AUC.** AUC compresses the complete ranking into one number: 0.5 is random and
1.0 is perfect on this test sample. Longer bars are better, but tiny differences may be
statistical fluctuations. The two panels answer related but different questions: overall
ranking quality versus performance at a chosen working point.'''),
 ('demo_quark_gluon_classification.ipynb', "importance.plot.barh", 'bdt-feature-importance',
  '''### How to read this figure

Each bar is the BDT's impurity-based feature importance: how much splits using that variable
reduced classification impurity across the fitted trees. Larger bars mean the trained BDT
used that feature more often or more effectively. The importances are relative and sum to
one; they do not have physical units.

This is **model reliance**, not proof that a feature causes quark/gluon differences. Strongly
correlated variables can share or exchange importance, and impurity importance can favor
features offering many possible split points. Treat the ranking as a question to investigate,
then compare it with permutation, ablation, or physics-motivated studies.'''),
 ('demo_dijet_event_display.ipynb', 'fig.show()', 'eta-phi-event-map',
  '''### How to read this figure

Each point is a final-state particle placed at its pseudorapidity $\eta$ and azimuth $\phi$;
larger markers have larger transverse momentum $p_T$. Colors identify particles assigned to
different jets, gray marks particles outside those jets, an × marks each jet axis, and the
dashed circle shows the radius parameter $R$ around that axis.

Look for dense groups of hard particles around the two jet axes. Remember that $\phi$ wraps:
points near $-\pi$ and $+\pi$ are actually neighbors. The circle illustrates a radius in this
plane, but anti-$k_t$ clustering is defined by its distance algorithm rather than by simply
drawing fixed cones.'''),
 ('demo_dijet_event_display.ipynb', 'fig3d.show()', 'pt-lego-view',
  '''### How to read this figure

The horizontal floor is the same $\eta$–$\phi$ plane as above, while the height of each stem
is particle $p_T$ in GeV. Tall stems therefore expose the hard cores of the jets and short
stems show softer radiation. Rotate the interactive view and hover over a point to read its
$p_T$.

This “lego” height is a visualization choice, not a third spatial coordinate. Compare the
amount of hard, concentrated radiation with the softer activity around and between the jet
cores.'''),
 ('demo_dijet_event_display.ipynb', 'fig_cyl.show()', 'momentum-cylinder-view',
  '''### How to read this figure

All rays begin at the collision point and point in the particles' three-dimensional momentum
directions. The axes are labeled $p_x$, $p_y$, and $p_z$: $z$ follows the beam, while $x$ and
$y$ are transverse. Similar-colored, nearby rays form a jet; the two main sprays give the
event its “dijet” name.

Ray length uses $\log(1+|p|)$, not the true momentum magnitude. This compression keeps soft
particles visible beside hard ones, so compare directions and grouping directly but do not
interpret displayed lengths as a linear momentum scale.'''),
]

for args in FIGURE_NOTES:
    upsert_figure_note(*args)


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
