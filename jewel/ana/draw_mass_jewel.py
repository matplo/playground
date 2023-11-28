#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import sys
import math
import glob
import tqdm

import ROOT
ROOT.gROOT.IgnoreCommandLineOptions = True


def fill(t, h, args):
	# print(h.GetName(), t.GetName())
	for j in t:
		# select on the range of the pt
		if j.pt < args.jetptmin or j.pt > args.jetptmax:
			continue
		# select on the range of the eta
		if j.eta < -0.9 + args.jetR or j.eta > +0.9 - args.jetR:
			continue
		h.Fill(j.m)
	

def main():
	parser = argparse.ArgumentParser(description="Plot the results from the analysis")
	parser.add_argument("dir", help="Input directory")
	parser.add_argument("output", help="Output file")
	parser.add_argument('type', help="Type of the calculation - med or vac")
	parser.add_argument("--jetptmin", type=float, default=60, help="Minimum jet pt")
	parser.add_argument("--jetptmax", type=float, default=80, help="Maximum jet pt")
	parser.add_argument("--jetR", type=float, default=0.2, help="Jet R")
	parser.add_argument("--bindata", type=str, default="", help="Bin data file filename:histogram_name")

	args = parser.parse_args()
		
	f = ROOT.TFile(args.output, "RECREATE")
	f.cd()
 
	hbin = None
	h = []
	if args.bindata:
		fname = args.bindata.split(":")[0]
		hname = args.bindata.split(":")[1]
		if not os.path.isfile(fname):
			_hbin = ROOT.TH1F(hname, hname, 100, 0, 50)
			_hbin.SetDirectory(0)
		else:
			fbin = ROOT.TFile(fname)
			_hbin = fbin.Get(hname)
		if not _hbin:
			_hbin = ROOT.TH1F(hname, hname, 100, 0, 50)
			_hbin.SetDirectory(0)      
		f.cd()
		for _hname in ["hm", "hm0", "hmx", "hmx0"]:
			h.append(_hbin.Clone(_hname))
			h[-1].Reset()
			h[-1].Sumw2()   
		fbin.Close()
	else: 
		for _hname in ["hm", "hm0", "hmx", "hmx0"]:
			h.append(ROOT.TH1F(_hname, _hname, 100, 0, 50))
			h[-1].Sumw2()
	
	files = glob.glob(os.path.join(args.dir, f"**/{args.type}*_{args.jetR}.root"), recursive=True)
	for fn in files:
		print(fn)
	
	for fn in tqdm.tqdm(files):
		_f = ROOT.TFile(fn)
		for i, t in enumerate([_f.Get("tn"), _f.Get("tn_m0"), _f.Get("tn_mx"), _f.Get("tn_mx0")]):
			fill(t, h[i], args)
		_f.Close()

	for _h in h:
		_h.Scale(1./_h.Integral(), 'width')

	f.Write()
	f.Close()
	print('[i] written', f.GetName())

if __name__ == "__main__":
	main()