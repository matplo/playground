#!/usr/bin/env python3

import ROOT
import os

nbin = [2, 3, 4]
system = ['pp', 'PbPb']
observable = 'mass'
fname = f'/rstorage/alice/AnalysisResults/ang/{system}/AngR02_ptbin{nbin}/{observable}/final_results/fFinalResults.root'

def write_histograms(list_of_files, output_fname):
    fout = ROOT.TFile(output_fname, 'recreate')
    for fname in list_of_files:
        f = ROOT.TFile(fname, 'read')
        for hname in list_of_files[fname]:
            h = f.Get(hname)
            if h:
                print('[i] writing', fname, ':', hname)
                fout.cd()
                h.Write()
            else:
                print('[i] failed', fname, ':', hname)
        f.Close()
    fout.Close()
    print('[i] written', output_fname)
    
# dict of files to be used in the analysis
list_of_files_s = {}
for s in system:
    list_of_files_s[s] = {}
    list_of_files = list_of_files_s[s]
    for n in nbin:
        fname = f'/rstorage/alice/AnalysisResults/ang/{s}/AngR02_ptbin{n}/{observable}/final_results/fFinalResults.root'
        list_of_files[fname] = []
        # check if file exists and add it to the list of files
        if not os.path.isfile(fname):
            print(f'File {fname} does not exist')
            continue
        f = ROOT.TFile(fname, 'read')
        for l in f.GetListOfKeys():
            if 'hmain_mass_' in l.GetName() or 'hResult_mass_systotal_' in l.GetName():
                # if object inherits from TH1 or TGraph add it to the list of objects
                if 'TH1' in l.GetClassName() or 'TGraph' in l.GetClassName():
                    list_of_files[fname].append(l.GetName())
        f.Close()

for s in system:
    write_histograms(list_of_files_s[s], f'alice_data_{s}.root')