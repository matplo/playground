#!/usr/bin/env python3


from __future__ import print_function
import tqdm
import argparse
import os
import numpy as np
import sys
import yasp
import cppyy

import sys
#_heppyy_dir = os.path.join(os.path.dirname(__file__), '..')
#sys.path.append(_heppyy_dir)

headers = [
    "fastjet/PseudoJet.hh",
    "fastjet/JetDefinition.hh",
    "fastjet/ClusterSequence.hh",
    "fastjet/Selector.hh",

    "Pythia8/Pythia.h",
    "Pythia8/Event.h"]

packs = ['fastjet', 'pythia8']
libs = ['fastjet', 'pythia8']

from yasp.cppyyhelper import YaspCppyyHelper
ycppyy = YaspCppyyHelper().load(packs, libs, headers)
print(ycppyy)

from cppyy.gbl import fastjet as fj
from cppyy.gbl import Pythia8
from cppyy.gbl.std import vector

print(sys.path)
from heppyy.pythia_util import configuration as pyconf

print(cppyy.gbl.__dict__)

import ROOT
import math
import array

from heppyy.util.mputils import pwarning, perror, pinfo

from PIL import Image
def jet_to_image(j, R0, w, h, ch=3, filebasename=None, draw_voronoi=False, draw_points=False):
	# img_numpy = np.zeros((h, w, ch), dtype=np.uint8)
	img_numpy = np.full((h, w, ch), 255, dtype=np.uint8)
	points = []
	for c in j.constituents():
		dphi = c.delta_phi_to(j) # -pi to pi
		x = int(dphi / R0 * w / 2) + int(w / 2)
		deta = c.eta() - j.eta()
		y = int(deta / R0 * h / 2) + int(h / 2)
		z = c.perp() / j.perp()
		col = int(z * 255)
		# pinfo('setting', x, y, 'to', col)
		points.append((x, y, col))

	if draw_voronoi:
		for ix in range(0, w):
			for iy in range(0, h):
				pmin = None
				dmin = 1.e7
				for p in points:
					x, y, col = p
					d = math.sqrt(math.pow(x-ix, 2) + math.pow(y-iy, 2))
					if d < dmin:
						pmin = p
						dmin = d
				if pmin:
					x, y, col = pmin
					img_numpy[ix][iy] = [255-col, 255-col, 255-col]

	if draw_points:
		for p in points:
			x, y, col = p
			for ix in range(-2,3):
				for iy in range(-2,3):
					try:
						# img_numpy[x+ix][y+iy] = [255,255,255] # [col, 0, 0]
						img_numpy[x+ix][y+iy] = [col, 0, 0]
					except:
						pass
	img = Image.fromarray(img_numpy, "RGB")
	n = 1
	if filebasename is None:
		filebasename = ''
	fname = f'{filebasename}{n}.png'
	while os.path.exists(fname):
		n += 1
		fname = f'{filebasename}{n}.png'
	img.save(fname)
	# pinfo('image saved as', fname)
	return fname


def mergeAB(A, B, outputdir):
	imgA = Image.open(A)
	npA = np.array(imgA)
	imgB = Image.open(B)
	npB = np.array(imgB)
	npAB = np.concatenate([npA, npB], 1)
	imgAB = Image.fromarray(npAB)
	outputfname = os.path.join(outputdir, os.path.basename(A))
	imgAB.save(outputfname)
	# pinfo('merged image saved as', outputfname)


def config_for_LEP(mycfg):
	# Initialize PYTHIA for hadronic events LEP1 - main06.cc
	mZ = 91.188
	mycfg = [	"PDF:lepton = off", # Allow no substructure in e+- beams: normal for corrected LEP data.
				"WeakSingleBoson:ffbar2gmZ = on", # Process selection.
				"23:onMode = off", # Switch off all Z0 decays and then switch back on those to quarks.
				"23:onIfAny = 1 2 3 4 5",
				"Beams:idA =  11",
				"Beams:idB = -11",
				f"Beams:eCM={mZ}", # LEP1 initialization at Z0 mass.
				"HadronLevel:all=off", # parton level first
				"PhaseSpace:bias2Selection=off", # this is ON by default in pyconf - not OK for these settings
				"Random:setSeed=on"]
	#				"Random:seed={}".format(random.randint(100000, 900000))]
	return mycfg


def main():
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly', prog=os.path.basename(__file__))
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('--ignore-mycfg', help="ignore some settings hardcoded here", default=False, action='store_true')
	parser.add_argument('-v', '--verbose', help="be verbose", default=False, action='store_true')
	parser.add_argument('--output', help='output directory', default='./pix2pixjets', type=str)
	parser.add_argument('--min-jet-pt', help='minimum pt of jets', default=100., type=float)
	parser.add_argument('--max-jet-pt', help='maximum pt of jets', default=120., type=float)
	parser.add_argument('--voronoi', help='draw voronoi areas', action='store_true', default=False)
	parser.add_argument('--points',  help='make points in the figures', action='store_true', default=False)
	parser.add_argument('--LEP',  help='init for LEP', action='store_true', default=False)

	args = parser.parse_args()
	for a in args.__dict__:
		pinfo(a, args.__dict__[a])

	if args.voronoi is False and args.points is False:
		perror('at least one of the --voronoi or --points should be specified')
		return

	outputdirparton = args.output+'/parton/'
	outputdirhadron = args.output+'/hadron/'
	os.makedirs(outputdirparton, exist_ok=True)
	os.makedirs(outputdirhadron, exist_ok=True)
	outputdir_parton_hadron = args.output+'/parton_hadron/'
	os.makedirs(outputdir_parton_hadron, exist_ok=True)

	pythia = Pythia8.Pythia()

	# jet finder
	# print the banner first
	fj.ClusterSequence.print_banner()
	print()
	# jet_R0 = 0.4
	jet_R0 = math.pi
	jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0)

	max_eta_hadron = jet_R0 * 2.5
	pwarning('max eta for particles after hadronization set to', max_eta_hadron)
	parts_selector_h = fj.SelectorAbsEtaMax(max_eta_hadron)
	jet_selector_h = fj.SelectorPtMin(args.min_jet_pt) * fj.SelectorPtMax(args.max_jet_pt) * fj.SelectorAbsEtaMax(max_eta_hadron - 1.05 * jet_R0)
	max_eta_parton = max_eta_hadron + 2. * jet_R0
	pwarning('max eta for partons set to', max_eta_parton)
	parts_selector_p = fj.SelectorAbsEtaMax(max_eta_parton)
	jet_selector_p = fj.SelectorPtMin(args.min_jet_pt) * fj.SelectorPtMax(args.max_jet_pt) * fj.SelectorAbsEtaMax(max_eta_hadron - 1.05 * jet_R0)

	mycfg = ['PhaseSpace:pThatMin = {}'.format(args.min_jet_pt)]
	mycfg.append('HadronLevel:all=off')
	if args.ignore_mycfg:
		mycfg = []
	if args.LEP:
		mycfg = config_for_LEP(mycfg)
		jet_R0 = 1.0
		if args.min_jet_pt > 45:
			args.min_jet_pt = 10
		if args.max_jet_pt < 45:
			args.max_jet_pt = 50
		max_eta_hadron = 3.
		pwarning('max eta for particles after hadronization set to', max_eta_hadron)
		parts_selector_h = fj.SelectorAbsEtaMax(max_eta_hadron)
		jet_selector_h = fj.SelectorPtMin(args.min_jet_pt) * fj.SelectorPtMax(args.max_jet_pt) * fj.SelectorAbsEtaMax(max_eta_hadron - 1.05 * jet_R0)
		max_eta_parton = max_eta_hadron + 2. * jet_R0
		pwarning('max eta for partons set to', max_eta_parton)
		parts_selector_p = fj.SelectorAbsEtaMax(max_eta_parton)
		jet_selector_p = fj.SelectorPtMin(args.min_jet_pt) * fj.SelectorPtMax(args.max_jet_pt) * fj.SelectorAbsEtaMax(max_eta_hadron - 1.05 * jet_R0)

	match_dR = 0.2
	if match_dR >= jet_R0 / 2.:
		match_dR = jet_R0 / 2.

	pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)
	if not pythia:
		perror("pythia initialization failed.")
		return

	if args.nev < 1:
		args.nev = 1

	pbar = tqdm.tqdm(total=args.nev)
	while pbar.n < args.nev:
		if not pythia.next():
			continue
		parts_p = vector[fj.PseudoJet]([fj.PseudoJet(p.px(), p.py(), p.pz(), p.e()) for p in pythia.event if p.isFinal()])
		jets_p = fj.sorted_by_pt(jet_selector_p(jet_def(parts_p)))
		if len(jets_p) < 1:
			# pwarning('no jets on parton level - event skipped')
			continue

		print('preH:', pythia.event.size())

		savedEvent = Pythia8.Event(pythia.event)
		_hadron_level = pythia.forceHadronLevel()
		if _hadron_level is False:
			# pwarning('hadron level failed - event skipped')
			continue

		print('postH:', savedEvent.size(), pythia.event.size())

		while pbar.n < args.nev:
			pythia.event = savedEvent
			print('preH:', savedEvent.size(), pythia.event.size())
			_hadron_level = pythia.forceHadronLevel()
			print('postH:', savedEvent.size(), pythia.event.size())
			parts_h = vector[fj.PseudoJet]([fj.PseudoJet(p.px(), p.py(), p.pz(), p.e()) for p in pythia.event if p.isFinal()])
			jets_h = fj.sorted_by_pt(jet_selector_h(jet_def(parts_h)))
			matched = []
			if len(jets_h) < 1:
				# pwarning('no jets on hadron level - event skipped')
				continue
			# take only leading jet...
			for jh in [jets_h[0]]:
				for jp in jets_p:
					dR = jh.delta_R(jp)
					if dR < match_dR:
						matched.append([jp, jh])
			# pinfo('number of matched jets', len(matched))

			_accepted = False
			for m in matched:
				if len(m[0].constituents()) < 2:
					continue
				# pinfo('parton level jet:', m[0].perp(), 'hadron level', m[1].perp(), 'dR', m[0].delta_R(m[1]))
				A = jet_to_image(m[0], jet_R0, 600, 600, 3, outputdirparton, draw_voronoi=args.voronoi, draw_points=args.points)
				B = jet_to_image(m[1], jet_R0, 600, 600, 3, outputdirhadron, draw_voronoi=args.voronoi, draw_points=args.points)
				mergeAB(A, B, outputdir_parton_hadron)
				_accepted = True

			if _accepted:
				pbar.update(1)

	pbar.close()
	pythia.stat()

	print(type(pythia))


if __name__ == '__main__':
	main()
