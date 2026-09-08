"""Manually reviewed measurement columns. No models, RCP, or ratios of RAA."""
SELECTIONS = []
def add(key, record, table, name, family, system='Pb-Pb', energy=5.02,
        centrality='0–10%', observable='RAA', column=0, notes='', acceptance=''):
    SELECTIONS.append(dict(key=key, record=str(record), table=table, column=column,
        name=name, family=family, system=system, energy_TeV=energy,
        centrality=centrality, observable=observable, notes=notes, acceptance=acceptance))

add('pb_h',86210,'Table 8','h±','Light hadrons',centrality='0–5%')
for key,t,name in [('pi',15,'π±'),('k',16,'K±'),('p',17,'p + p̄')]:
    add('pb_'+key,104923,f'Table {t}',name,'Light hadrons',centrality='0–5%')
add('pb_kstar',140098,'Table 1','K*⁰ + K̄*⁰','Light hadrons')
add('pb_phi',140098,'Table 12','φ','Light hadrons')
add('pb_kstarpm',150017,'Table 9','K*±','Light hadrons')
for key,t,name in [('d0',11,'Prompt D⁰'),('dplus',12,'Prompt D⁺'),('dstar',13,'Prompt D*⁺'),('davg',17,'Prompt D average')]:
    add('pb_'+key,127979,f'Table {t}',name,'Heavy flavour',acceptance='|y| < 0.5 (table description)')
add('pb_ds',127980,'Table 3','Prompt Dₛ⁺','Heavy flavour')
add('pb_lc',138404,'Table 2','Prompt Λc⁺','Heavy flavour')
add('pb_b_d0',135987,'Table 1b','Nonprompt D⁰','Heavy flavour')
add('pb_b_e',144336,'Table 3','Beauty → e','Heavy flavour')
add('pb_prompt_jpsi',146723,'Table 7','Prompt J/ψ','Quarkonia')
add('pb_b_jpsi',146723,'Table 11','Nonprompt J/ψ','Quarkonia')
add('pb_jpsi',146644,'Figure6-Panel1_midy_0-10','Inclusive J/ψ (mid-y)','Quarkonia',acceptance='|y| < 0.9; arXiv:2303.13361')
add('pb_jpsi_fwd',146644,'Figure6-Panel2_fwdy_0-20','Inclusive J/ψ (forward)','Quarkonia',centrality='0–20%',acceptance='2.5 < y < 4; arXiv:2303.13361')
for r,l in [(0.2,'a'),(0.4,'b'),(0.6,'c')]:
    add(f'pb_chjet{int(r*10)}',146027,f'Figure 4{l} top ML',f'Charged jets R={r}','Jets',
        notes='ML background subtraction. Centrality from arXiv:2303.00592v2, Figure 4. Only pT centres deposited; box widths are decorative.',
        acceptance=f'|ηjet| < {0.9-r:.1f}; anti-kT; arXiv:2303.00592v2')
for r,t,lead in [(0.2,30,5),(0.4,31,7)]:
    add(f'pb_fulljet{int(r*10)}',93739,f'Table {t}',f'Full jets R={r}, lead>{lead}','Jets',
        notes=f'Leading charged track pT > {lead} GeV/c in both pp and Pb-Pb. Full = charged + neutral. Table 31 x header says R=0.4; pT interpretation follows Figure 6/table description.')
add('pb_djet',168854,r'$\mathrm{D^{0}}$-jet $R_{\mathrm{AA}}$','D⁰-tagged charged jets R=0.3 †','Jets',
    notes='Centrality: arXiv:2409.11939v1 section 2. † Deposited statistical errors decrease to 1.53e-5 at high pT and appear unusually small; retained verbatim, not independently validated against an author correction. Do not use for significance estimates. D0 constituent pT: 3–36 GeV/c.')
for r,t in [(0.2,6),(0.4,7)]:
    add(f'pb_gamma{int(r*10)}',157542,f'table_{t}',f'Isolated γ Riso={r}','Photons')

# Additional AA systems/energies remain distinct in the explorer.
add('xe_h',85727,'Table 3','h±','Light hadrons',system='Xe-Xe',energy=5.44,centrality='0–5%')
add('pb276_h',86210,'Table 7','h±','Light hadrons',energy=2.76,centrality='0–5%')
for j,key,name in [(0,'pi','π±'),(1,'k','K±'),(2,'p','p + p̄')]:
    add('pb276_'+key,71310,'Table 7',name,'Light hadrons',energy=2.76,centrality='0–5%',column=j)
add('pb276_pi0',83964,'Table 5','π⁰','Light hadrons',energy=2.76)
add('pb276_eta',83964,'Table 7','η','Light hadrons',energy=2.76)
add('pb276_b_e',77904,'Table 4','Beauty → e','Heavy flavour',energy=2.76,centrality='0–20%')
add('pb276_fulljet',68483,'Table 6','Full jets R=0.2, lead>5','Jets',energy=2.76,notes='Leading charged track pT > 5 GeV/c; arXiv:1502.01689.')

# Event-activity-selected p-Pb uses QpPb, not minimum-bias RpPb.
add('q_h',68361,'Table 9','h± (ZNA hybrid, Pb-side)','Light hadrons',system='p-Pb',centrality='0–5%',observable='QpPb',
    notes='ZNA selection; Ncoll from Pb-side multiplicity scaling (hybrid method). Table 9 header carries centrality. Repeated common TpA/normalization entries lack percent signs; retained as deposited and excluded from pointwise uncertainty boxes.')
for r,t in [(0.2,13),(0.4,18)]:
    add(f'q_chjet{int(r*10)}',72903,f'Table {t}',f'Charged jets R={r} (ZNA)','Jets',system='p-Pb',centrality='0–20%',observable='QpPb',
        notes='ZNA selection and Pb-side multiplicity scaling for Ncoll; arXiv:1603.03402. Deposited sqrt(sNN)=5023 GeV, displayed as 5.02 TeV.')
for record,energy,tables,suffix in [(69212,5.02,[10,11],''),(100166,8.16,[6,7],'816')]:
    for t,direction in zip(tables,['Pb-going','p-going']):
        add('q_jpsi_'+direction+suffix,record,f'Table {t}',f'J/ψ ({direction})','Quarkonia',system='p-Pb',energy=energy,centrality='2–10%',observable='QpPb')

def pa(key,record,table,name,family,energy=5.02,column=0,centrality='Minimum bias',notes=''):
    add(key,record,table,name,family,system='p-Pb',energy=energy,centrality=centrality,observable='RpPb',column=column,notes=notes)
pa('pa_h',86210,'Table 9','h±','Light hadrons',centrality='NSD')
for j,key,name in [(0,'pi','π±'),(1,'k','K±'),(2,'p','p + p̄')]:
    pa('pa_'+key,73749,'Table 10',name,'Light hadrons',column=j,centrality='NSD')
pa('pa_d0',73941,'Table 20','Prompt D⁰','Heavy flavour')
pa('pa_b_e',77904,'Table 3','Beauty → e','Heavy flavour')
for j,r in enumerate([0.2,0.3,0.4]):
    pa(f'pa_chjet{int(r*10)}',150694,'Table 4.1',f'Charged jets R={r}','Jets',column=j)
pa('pa_bjet',128018,'Table 11','b-tagged charged jets R=0.4','Jets',notes='Combined secondary-vertex and impact-parameter result.')
for energy,record,tables,suffix in [(5.02,83700,[5,6],''),(8.16,128138,[7,8],'816')]:
    for t,key,name in zip(tables,['pi0','eta'],['π⁰','η']):
        pa('pa_'+key+suffix,record,f'Table {t}',name,'Light hadrons',energy=energy)
pa('pa_gamma',165479,'RpPb 5.02 TeV','Isolated γ Riso=0.4','Photons')
pa('pa_gamma816',165479,'RpA 8.16 TeV','Isolated γ Riso=0.4','Photons',energy=8.16)
pa('pa_hfe816',142624,'Table 3','Heavy flavour → e','Heavy flavour',energy=8.16)
for key,table,name in [('jpsi','Figure 5b','Inclusive J/ψ'),('prompt_jpsi','Figure 9a','Prompt J/ψ'),('b_jpsi','Figure 9b','Nonprompt J/ψ')]:
    pa('pa_'+key+'816',138403,table,name,'Quarkonia',energy=8.16)
add('oo_pi0',182535,r'$\pi^{0}$ $R_{\rm OO}$ at 5.36 TeV','π⁰','Light hadrons',system='O-O',energy=5.36,centrality='Minimum bias',observable='ROO',notes='Minimum-bias cross-section ratio, scaled by A²=256; not a central-event measurement.')
add('po_pi0',182535,r'$\pi^{0}$ $R_{\rm pO}$ at 9.62 TeV','π⁰','Light hadrons',system='p-O',energy=9.62,centrality='Minimum bias',observable='RpO',notes='Minimum-bias cross-section ratio, scaled by A=16; not a central-event measurement.')

PRESETS = {
 'Central Pb-Pb · hadrons': ['pb_h','pb_pi','pb_p','pb_d0','pb_ds','pb_lc','pb_b_d0','pb_jpsi'],
 'Central Pb-Pb · jets and photons': ['pb_chjet2','pb_chjet4','pb_fulljet4','pb_djet','pb_gamma4'],
 'Central p-Pb · QpPb': ['q_h','q_chjet4','q_jpsi_Pb-going','q_jpsi_p-going'],
 'Minimum-bias p-Pb · RpPb': ['pa_h','pa_p','pa_d0','pa_chjet4','pa_bjet','pa_gamma'],
 'AA · light hadrons by system': ['pb_h','pb276_h','xe_h'],
 'Pb-Pb · identified light hadrons': ['pb_pi','pb_k','pb_p','pb_kstar','pb_phi','pb_kstarpm'],
 'Pb-Pb · heavy flavour': ['pb_d0','pb_ds','pb_lc','pb_b_d0','pb_b_e','pb_prompt_jpsi','pb_b_jpsi'],
 'p-Pb · quarkonia by energy': ['q_jpsi_Pb-going','q_jpsi_p-going','q_jpsi_Pb-going816','q_jpsi_p-going816'],
 'Oxygen systems · minimum bias': ['oo_pi0','po_pi0'],
}
