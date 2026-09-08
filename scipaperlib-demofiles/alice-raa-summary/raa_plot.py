"""Reusable Matplotlib plotting; data are bundled and require no library access."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
HERE=Path(__file__).resolve().parent
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#6A51A3','#B58A00','#333333','#56B4E9','#A63603','#E377C2']
MARKERS=['o','s','^','D','v','P','X','>','h','<']
Y_LABELS={'RAA':r'$R_{\mathrm{AA}}$','QpPb':r'$Q_{\mathrm{pPb}}$','RpPb':r'$R_{\mathrm{pPb}}$','ROO':r'$R_{\mathrm{OO}}$','RpO':r'$R_{\mathrm{pO}}$'}
def load_data():return json.loads((HERE/'data'/'curves.json').read_text())

def draw_panel(ax,curves,logx=True,systematics=True,title='',short_labels=False,xlim=None,ylim=None):
    if logx:ax.set_xscale('log')
    ax.axhline(1,color='#555555',lw=1,ls=(0,(4,3)),zorder=0)
    for i,c in enumerate(curves):
        color=COLORS[i%len(COLORS)];marker=MARKERS[i%len(MARKERS)]
        ps=c['points'];x=np.array([p['x'] for p in ps]);y=np.array([p['y'] for p in ps])
        if systematics:
            boxes=[]
            for p in ps:
                # True deposited bins, or visibly narrow decorative widths if no edges exist.
                lo,hi=p['xlow'],p['xhigh']
                if lo is None:lo,hi=p['x']*0.975,p['x']*1.025
                if logx and lo<=0:lo=max(p['x']*.5,1e-4)
                boxes.append(Rectangle((lo,p['y']-p['sys_minus']),hi-lo,p['sys_minus']+p['sys_plus']))
            ax.add_collection(PatchCollection(boxes,facecolor=color,edgecolor=color,alpha=.13,linewidth=.6,zorder=1))
        lab=f"{c['name']} ({c['centrality']})" if short_labels else c['label']
        ax.errorbar(x,y,yerr=[[p['stat_minus'] for p in ps],[p['stat_plus'] for p in ps]],
            color=color,fmt=marker,ms=3.5,mew=.65,mfc='white' if i%2 else color,
            elinewidth=.8,capsize=1.5,label=lab,zorder=2+i*.05)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=11,pad=11)
    ax.set_xlabel(r'$p_{\mathrm{T}}$ of the measured object (GeV/$c$)')
    obs=list(dict.fromkeys(c['observable'] for c in curves))
    ax.set_ylabel(' / '.join(Y_LABELS[o] for o in obs) if obs else 'Nuclear modification factor')
    ax.grid(axis='y',color='#e1e6eb',lw=.6)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(which='both',direction='in',top=False,right=False)
    if xlim:ax.set_xlim(*xlim)
    if ylim:ax.set_ylim(*ylim)
    else:
        top=max([1.1]+[p['y']+max(p['stat_plus'],p['sys_plus'] if systematics else 0) for c in curves for p in c['points']])
        bottom=min([0]+[p['y']-max(p['stat_minus'],p['sys_minus'] if systematics else 0) for c in curves for p in c['points']])
        ax.set_ylim(bottom*1.05,top*1.08)
    if curves:ax.legend(fontsize=8,frameon=False,loc='upper left',bbox_to_anchor=(1.015,1),borderaxespad=0,handletextpad=.4,labelspacing=.7)
    return ax

def plot_selected(keys,logx=True,systematics=True,title='ALICE · selected measurements',xlim=None):
    data=load_data();bykey={c['key']:c for c in data['curves']}
    selected=[bykey[k] for k in keys]
    groups={}
    for c in selected:groups.setdefault(c['observable'],[]).append(c)
    if not groups:groups={'':[]}
    fig,axes=plt.subplots(len(groups),1,figsize=(12,4.9*len(groups)),squeeze=False)
    fig.subplots_adjust(left=.09,right=.56,top=.87,bottom=.18 if len(groups)==1 else .1,hspace=.48)
    for ax,(obs,cs) in zip(axes[:,0],groups.items()):
        draw_panel(ax,cs,logx,systematics,title=obs or 'Select at least one measurement',xlim=xlim)
    fig.suptitle(title,x=.09,ha='left',fontsize=16,fontweight='bold',y=.97)
    fig.text(.09,.035,'Bars: statistical. Boxes: point-dependent systematic. Separate global normalization errors are not drawn.\nSelections, energies, acceptances and references differ; see the source table. No rebinning or interpolation.',fontsize=8,color='#52606d')
    return fig

def make_summary():
    data=load_data();bykey={c['key']:c for c in data['curves']}
    presets=list(data['presets'].items())[:4]
    fig=plt.figure(figsize=(16,12),facecolor='white')
    # Separate legend rows keep data panels unobstructed.
    grid=fig.add_gridspec(2,2,left=.07,right=.98,top=.87,bottom=.23,hspace=.75,wspace=.23)
    titles=['Central Pb–Pb · hadrons and quarkonia','Central Pb–Pb · jets and isolated photons',
            'Centrality-selected p–Pb','Minimum-bias / NSD p–Pb references']
    for i,((_,keys),title) in enumerate(zip(presets,titles)):
        ax=fig.add_subplot(grid[i//2,i%2]);cs=[bykey[k] for k in keys]
        draw_panel(ax,cs,title=title,short_labels=True,xlim=(.45,180) if i!=1 else (4,200),ylim=(0,2.05) if i==0 else ((0,1.65) if i==1 else (0,2.1)))
        ax.get_legend().remove()
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.24),ncol=2,frameon=False,fontsize=8,handletextpad=.3,columnspacing=1.1,labelspacing=.45,borderaxespad=0)
    fig.text(.07,.965,'ALICE  |  Nuclear modification across probes',fontsize=24,fontweight='bold',color='#162a40')
    fig.text(.07,.928,'√sNN = 5.02 TeV  •  Most-central available pT-differential selections  •  Default SciPaperlib datasets',fontsize=12,color='#52606d')
    fig.text(.07,.035,f'Bars: statistical; shaded boxes: point-dependent systematic. Separate global normalization errors are not drawn.\nEach label gives the published centrality selection. Object pT is not a common parent-parton energy. Curated comparison, not an ALICE combined result.\n† D⁰-jet statistical errors are shown as deposited but appear unusually small; see README. Full-jet lead cut is in GeV/c; charged-jet results use ML subtraction.\nSource DOIs, acceptances, raw YAML, all {len(data["curves"])} selectable curves and uncertainty components are included in the companion files.',fontsize=8.5,color='#52606d',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(HERE/f'alice_raa_summary.{ext}',dpi=220,facecolor='white')
    plt.close(fig)
if __name__=='__main__':
    plt.switch_backend('Agg')
    make_summary()
