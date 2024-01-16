import os
import boosted_fits as bsvj
import quick_plot as tools
import ROOT as r
import numpy as np


rootfiles = bsvj.pull_arg('rootfiles', type=str, nargs='+').rootfiles
clean  = bsvj.pull_arg('--clean', action='store_true').clean
mz     = bsvj.pull_arg('--mz',type=float).mz
rinv   = bsvj.pull_arg('--rinv',type=float).rinv


#make a new ttree (limit) in a new root file, write cls into it 
file = r.TFile.Open("Limits/Limit_mz{}_rinv{}.root".format(mz,rinv),"RECREATE")
tree = r.TTree("limit","limit")

base_qtys = ["quantileExpected"]
keys = ["limit"]
qtys = base_qtys + keys + ["trackedParam_{}".format(q) for q in ["mZprime","mDark","rinv","xsec"]]
r.gROOT.ProcessLine("struct quantile_t { "+" ".join(["Double_t {};".format(qty) for qty in qtys])+" };")

qobj = r.quantile_t()

for qty in qtys:
  tree.Branch(qty, r.addressof(qobj,qty), '{}/D'.format(qty))



for observed, asimov in zip(*tools.organize_rootfiles(rootfiles)):
  obs, asimov = tools.extract_scans([observed, asimov], correct_minimum=True)
  if clean:
    obs    = tools.clean_scan(obs)
    asimov = tools.clean_scan(asimov)
  cls = tools.get_cls(obs, asimov)
  limit = tools.interpolate_95cl_limit(cls)
  lim = {limit.expected: 0.5, limit.observed:-1, limit.twosigma_down:0.975, limit.onesigma_down:0.84, limit.onesigma_up: 0.16, limit.twosigma_up:0.025}
  for key, value in lim.items():
    qobj.limit = key
    qobj.quantileExpected = value
    qobj.trackedParam_mZprime = mz 
    qobj.trackedParam_mDark = 10 
    qobj.trackedParam_rinv = rinv 
    qobj.trackedParam_xsec = bsvj.get_xs(mz)
    tree.Fill()
file.Write()
file.Close()
