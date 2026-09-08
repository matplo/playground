"""Generate the offline HTML, executable notebook and provenance documentation."""
from pathlib import Path
import json, os, sys, tempfile, html
import nbformat
from nbclient import NotebookClient
from ipykernel.inprocess.manager import InProcessKernelManager
HERE=Path(__file__).resolve().parent

def main(execute=True):
    d=json.loads((HERE/'data/curves.json').read_text())
    payload=json.dumps(d,ensure_ascii=False).replace('</','<\\/')
    (HERE/'alice_raa_explorer.html').write_text((HERE/'explorer_template.html').read_text().replace('__DATA__',payload))
    nb=nbformat.v4.new_notebook()
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    nb.cells=[
      md('# ALICE nuclear modification across probes\n\n65 curated measurement curves from the **default SciPaperlib** library. This notebook uses the bundled numeric tables with Matplotlib and offers an `ipywidgets` selector. Run all cells, then choose a preset or individual measurements. The companion HTML works offline without Python.\n\nCentral A–A $R_{AA}$, centrality-selected p–Pb $Q_{pPb}$ and minimum-bias $R_{pA}$ are kept distinct. The lowest published percentile bin differs by analysis. The oxygen data are minimum bias. See `README.md` and `SOURCES.md` for definitions and caveats.'),
      md('## Setup\n\nOpen this notebook from the extracted folder, keeping `raa_plot.py` and `data/` beside it. Install dependencies in your selected kernel if needed:\n```python\n%pip install matplotlib numpy ipywidgets\n```\nThe original YAML is only needed for independent validation; plotting reads the bundled JSON.'),
      code('%matplotlib inline\nfrom pathlib import Path\nimport json, hashlib\nimport matplotlib.pyplot as plt\nimport ipywidgets as widgets\nfrom IPython.display import display, HTML, clear_output\nfrom raa_plot import load_data, plot_selected\n\ndata = load_data()\nby_key = {c["key"]: c for c in data["curves"]}\nprint(f"{len(by_key)} curves; {sum(len(c[\"points\"]) for c in by_key.values())} points")'),
      md('## Overview\n\nAll four panels below use 5.02 TeV data. Statistical bars and point-dependent systematic boxes are shown; separately deposited global scale errors are retained in the data but not drawn. **† D⁰-tagged jet statistical errors appear unusually small in the deposited table and are preserved without correction.**'),
      code('from IPython.display import Image\ndisplay(Image(filename="alice_raa_summary.png"))'),
      md('## Select what is plotted\n\nUse Ctrl/Cmd-click to select multiple individual curves. Different observable definitions automatically get separate panels. Sources for the selection appear below the plot. A momentum cut changes the displayed range only; no points are rebinned.'),
      code('preset = widgets.Dropdown(options=list(data["presets"]), description="Preset:", layout=widgets.Layout(width="700px"))\nselected = widgets.SelectMultiple(options=[(c["label"],c["key"]) for c in data["curves"]], value=tuple(next(iter(data["presets"].values()))), description="Curves:", rows=13, layout=widgets.Layout(width="98%"))\nlogx = widgets.Checkbox(value=True, description="Log pT")\nsystematics = widgets.Checkbox(value=True, description="Systematic boxes")\noutput = widgets.Output()\n\ndef update_plot(change=None):\n    with output:\n        clear_output(wait=True)\n        fig = plot_selected(selected.value, logx=logx.value, systematics=systematics.value)\n        display(fig)\n        plt.close(fig)\n        import html\n        items=[]\n        for key in selected.value:\n            c=by_key[key]\n            items.append(f\'<li><b>{html.escape(c["label"])}</b><br>{html.escape(c["acceptance"])}<br><a href="{c["source_url"]}">{c["table_doi"]}</a> — {html.escape(c["source"]["table"])}<br>{html.escape(c["notes"])}</li>\')\n        display(HTML("<ul>"+"".join(items)+"</ul>"))\n\ndef use_preset(change):\n    selected.value=tuple(data["presets"][change["new"]])\n\npreset.observe(use_preset, names="value")\nfor control in [selected,logx,systematics]:\n    control.observe(update_plot,names="value")\ndisplay(widgets.VBox([preset,selected,widgets.HBox([logx,systematics]),output]))\nupdate_plot()'),
      md('## Export the current selection\n\nRun this cell again after changing the controls. PNG and SVG use Matplotlib. The JSON includes the original numeric strings, bin edges, all uncertainty components, qualifiers and source DOIs.'),
      code('fig = plot_selected(selected.value, logx=logx.value, systematics=systematics.value)\nfig.savefig("alice_selected.png", dpi=220, bbox_inches="tight")\nfig.savefig("alice_selected.svg", bbox_inches="tight")\nplt.close(fig)\nPath("alice_selected.json").write_text(json.dumps({**data,"curves":[by_key[k] for k in selected.value]}, ensure_ascii=False, indent=2))\nprint("Saved alice_selected.png, alice_selected.svg, alice_selected.json")'),
      md('## Inspect a measurement and validate the bundled source files\n\nThe following checks every included YAML file against the library checksum, verifies the original central values, and prints one deposited point with its complete error breakdown. Missing cells (`-`) are logged in `data/validation.json`; they are not converted to zero.'),
      code('for c in data["curves"]:\n    assert hashlib.sha256(Path(c["raw_file"]).read_bytes()).hexdigest() == c["sha256"]\n    for p in c["points"]:\n        assert p["y"] == float(p["original_y"]["value"])\nprint("All bundled file checksums and original central values verified.")\nexample=by_key["pb_d0"]\nprint(example["source_url"])\nprint(json.dumps(example["points"][0], ensure_ascii=False, indent=2))'),
      md('## Interpretation limits\n\n- This is a curated comparison, not an ALICE combined result or a complete library census.\n- Energies, rapidities, centralities, jet radii, leading-track cuts and reference constructions differ. A hadron and a jet at equal pT do not select equal parent-parton energies.\n- p–Pb $Q_{pPb}$ depends on the event-activity estimator and the nuclear-scaling procedure; minimum-bias/NSD $R_{pPb}$ is kept separate. [ALICE centrality-method paper](https://arxiv.org/abs/1412.6828).\n- Percentage errors are converted to absolute errors on the ordinate. Supplied total systematics are preferred; otherwise named point-dependent terms are combined in quadrature for display only. Cross-bin and cross-measurement correlations are not reconstructed.\n- Separate scale errors are preserved but not drawn, including the correlated pT-global terms in the selected J/ψ QpPb tables. Additional scale errors appearing only in descriptions/paper text are not numerically reconstructed.\n- Horizontal boxes use source bin edges. If a table supplies only pT centres, narrow decorative box widths are used.\n- The D⁰-jet deposited statistical errors should be checked with the original authors before any significance estimate. No corrections have been invented.')
    ]
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]}}
    if execute:
        # A real IPython kernel in this process needs no network/Unix sockets.
        # Execute precisely the notebook cell source and retain kernel outputs.
        os.chdir(HERE)
        km=InProcessKernelManager();km.start_kernel()
        kc=km.client();kc.start_channels()
        try:
            for cell in nb.cells:
                if cell.cell_type!='code':continue
                kc.execute(cell.source)
                reply=kc.get_shell_msg()['content']
                cell.execution_count=reply['execution_count'];cell.outputs=[]
                while kc.iopub_channel.msg_ready():
                    msg=kc.get_iopub_msg();kind=msg['header']['msg_type'];content=msg['content']
                    if kind in ['stream','display_data','execute_result','error']:
                        cell.outputs.append(nbformat.v4.output_from_msg(msg))
                if reply['status']!='ok':raise RuntimeError('\n'.join(reply.get('traceback',[])))
            from ipywidgets import Widget
            nb.metadata['widgets']={'application/vnd.jupyter.widget-state+json':Widget.get_manager_state()}
            nb.metadata['validation']={'execution':'IPython in-process kernel; all code cells executed without errors','reason':'Sandbox prohibits kernel sockets'}
        finally:
            kc.stop_channels();km.shutdown_kernel()
    nbformat.write(nb,HERE/'alice_raa_explorer.ipynb')
    refs=['# Measurement sources\n','Every curve below comes from an existing dataset in the default SciPaperlib library. DOIs refer to the inspected HEPData table version. `data/curves.json` preserves the full provenance, qualifiers, original strings and error components.\n',
      '| Key | Measurement | Observable | Table source | Acceptance / notes |\n|---|---|---|---|---|']
    for c in d['curves']:
        clean=lambda s:s.replace('|','\\|').replace('\n',' ')
        refs.append(f"| `{c['key']}` | {clean(c['label'])} | {c['observable']} | [{c['table_doi']}]({c['source_url']}), [{c['source']['arxiv_id']}](https://arxiv.org/abs/{c['source']['arxiv_id'].replace('arXiv:','')}) | {clean(c['acceptance']+' '+c['notes'])} |")
    refs += ['\n## Supplemental definition checks\n',
      '- Charged-jet centrality and ML method: [arXiv:2303.00592v2, Figure 4](https://arxiv.org/html/2303.00592v2).',
      '- D⁰-tagged jet centrality and constituent cuts: [arXiv:2409.11939v1, sections 2 and 5](https://arxiv.org/html/2409.11939v1).',
      '- p–Pb centrality methods: [arXiv:1412.6828v2](https://arxiv.org/abs/1412.6828v2); charged-jet ZNA scaling: [arXiv:1603.03402](https://arxiv.org/abs/1603.03402).',
      '- J/ψ QpPb pointwise versus scale uncertainties: [arXiv:2008.04806v2, Figures 5, 7 and 8](https://arxiv.org/html/2008.04806v2). Figures 7 and 8 also show the 5.02 TeV reference with its correlated scale uncertainties.',
      '- Inclusive J/ψ rapidity acceptance: [arXiv:2303.13361v2](https://arxiv.org/abs/2303.13361v2).']
    (HERE/'SOURCES.md').write_text('\n'.join(refs)+'\n')
    print('Generated HTML, executed notebook, and source catalogue.')
if __name__=='__main__':main('--no-execute' not in sys.argv)
