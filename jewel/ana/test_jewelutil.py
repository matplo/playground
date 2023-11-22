#!/usr/bin/env python

import yasp
import heppyy.util.jewelutil_cppyy
from cppyy.gbl import HeppyyJewelUtil
from cppyy.gbl import HeppyyHepMCUtil

import heppyy.util.fastjet_cppyy
import heppyy.util.heppyy_cppyy
from cppyy.gbl import fastjet as fj
from cppyy.gbl.std import vector

import tqdm
import sys

import ROOT
import argparse
import glob
import math

def main():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('--input', type=str, help='Input file name')
	parser.add_argument('--output', type=str, help='Output file name')
	parser.add_argument('--list', help='switch to read a list of files', action='store_true')
	parser.add_argument('--dir', type=str, help='Directory name')
	parser.add_argument('--pattern', type=str, help='pattern to match in the directory', default='*.hepmc') 

	args = parser.parse_args()
	
	# If a directory is provided, find all .hepmc files in the directory
	if args.dir:
		flist = glob.glob(f'{args.dir}/**/{args.pattern}', recursive=True)
	elif args.list:
		flist = [line.rstrip('\n') for line in open(args.input)]
	else:
		flist = [args.input]
		
	print('[i] found', len(flist), 'files')
	_ = [print(f) for f in flist]

	fout = ROOT.TFile(args.output, 'recreate')
	fout.cd()
	tn = ROOT.TNtuple('tn', 'tn', 'pt:eta:phi:R:m')
	tn_m0 = ROOT.TNtuple('tn_m0', 'tn_m0', 'pt:eta:phi:R:m')
	tn_mx = ROOT.TNtuple('tn_mx', 'tn_mx', 'pt:eta:phi:R:m')
	tn_mx0 = ROOT.TNtuple('tn_mx0', 'tn_mx0', 'pt:eta:phi:R:m')
		
	for fname in flist:
		print('[i] processing', fname)
		fin = HeppyyJewelUtil.ReadJewelHepMC2File(fname)
		nev = HeppyyHepMCUtil.get_n_events(fname)
		pbar = tqdm.tqdm(total=nev)
		dmax = 0.5
		charged_only = True

		hadron_etamax = 0.9

		jet_R0 = 0.4
		jet_etamax = hadron_etamax - jet_R0 * 1.05
		jet_ptmin = 40.
		jet_ptmax = 200.

		jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0)
		jet_selector = fj.SelectorPtMin(jet_ptmin) * fj.SelectorPtMax(jet_ptmax) * fj.SelectorAbsEtaMax(jet_etamax)

		while fin.NextEvent():
			# fjsubtr_all = fin.fjFinalParticlesSubtractedThermalRivet(dmax)
			fjsubtr_charged = fin.fjFinalParticlesSubtractedThermalRivet(dmax, charged_only)
			for j in fj.sorted_by_pt(jet_selector(jet_def(fjsubtr_charged))):
				tn.Fill(j.perp(), j.eta(), j.phi(), jet_R0, j.m())
				mx = math.sqrt(j.e()**2 - j.px()**2 - j.py()**2 - j.pz()**2)
				tn_mx.Fill(j.perp(), j.eta(), j.phi(), jet_R0, mx)
			#print(f'n all:{len(fjsubtr_all)} charged:{len(fjsubtr_charged)}')
			_ = [p.reset_PtYPhiM(p.perp(), p.rapidity(), p.phi(), 0.1396) for p in fjsubtr_charged]
			for j in fj.sorted_by_pt(jet_selector(jet_def(fjsubtr_charged))):
				tn_m0.Fill(j.perp(), j.eta(), j.phi(), jet_R0, j.m())
				mx = math.sqrt(j.e()**2 - j.px()**2 - j.py()**2 - j.pz()**2)
				tn_mx0.Fill(j.perp(), j.eta(), j.phi(), jet_R0, mx)
			pbar.update(1)
		pbar.close()
	fout.Write()
	fout.Close()
	print('[i] written', fout.GetName())

if __name__ == "__main__":
	main()
