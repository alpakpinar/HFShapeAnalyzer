#!/usr/bin/env python

import os
import sys
import uproot
import numpy as np
import coffea.processor as processor

from candidates import setup_candidates
from coffea.processor.dataframe import LazyDataFrame
from coffea import hist
from coffea.util import save
from tqdm import tqdm

pjoin = os.path.join

Hist = hist.Hist
Bin = hist.Bin
Cat = hist.Cat

def accumulator():
    dataset_ax = Cat("dataset", "Primary dataset")
    region_ax = Cat("region", "Selection region")

    met_pt_ax = Bin("met", r"$p_{T}^{miss}$ (GeV)", 40, 0, 2000)
    met_phi_ax = Bin("metphi", r"$\phi^{miss}$ (GeV)", 30, -np.pi, np.pi)
    jet_eta_ax = Bin("jeteta", r"Leading Jet $|\eta|$", np.arange(2.9,5.0,0.1))
    jet_eta_ax_coarse = Bin("jeteta", r"Leading Jet $|\eta|$", [2.9, 3.25, 4, 5])
    jet_phi_ax = Bin("jetphi", r"Leading Jet $\phi$", 30, -np.pi, np.pi)
    sieie_ax = Bin("sieie", r"Leading Jet $\sigma_{i\eta i \eta}$", 50, 0, 0.2)
    sipip_ax = Bin("sipip", r"Leading Jet $\sigma_{i\phi i \phi}$", 50, 0, 0.2)

    cssize_ax = Bin("cssize", r"Jet Central Strip Size", [-0.5,0.5,1.5,2.5,3.5,4.5])
    adssize_ax = Bin("adssize", r"Jet Adjacent Strip Size", [-0.5,0.5,1.5,2.5,3.5,4.5])

    items = {}
    items['ak4_abseta0'] = Hist("Counts", region_ax, jet_eta_ax)
    items['sieie_sipip_ak4_abseta0'] = Hist("Counts", region_ax, sieie_ax, sipip_ax, jet_eta_ax_coarse)
    items['cssize_adssize_ak4_abseta0'] = Hist("Counts", region_ax, cssize_ax, adssize_ax, jet_eta_ax_coarse)
    
    items['met_phi_ak4_eta0'] = Hist("Counts", region_ax, met_phi_ax, jet_eta_ax_coarse)
    items['ak4_phi0_eta0'] = Hist("Counts", region_ax, jet_phi_ax, jet_eta_ax_coarse)

    return processor.dict_accumulator(items)

class Analyzer():
    def __init__(self) -> None:
        self.regions = {}
        self.accumulator = accumulator()

    def add_region(self, region_name: str, cuts: list):
        self.regions[region_name] = cuts
    
    def analyze(self, df):
        jets, muons, met_pt, met_phi = setup_candidates(df)

        leading_jet = jets[:,:1]
        other_jets = jets[:,1:]

        selection = processor.PackedSelection()
        
        selection.add("at_least_one_jet", leading_jet.counts==1)
        
        # Requirements on leading jet: pt>100 & |eta|>2.9
        selection.add("leading_jet_pt", (leading_jet.pt>100).any())
        selection.add("leading_jet_eta", (leading_jet.abseta>2.9).any())
        
        # No other jet with high pt (e.g. pt>30 GeV)
        selection.add("no_other_high_jet_pt", (other_jets.pt<30).all())
        selection.add("met_pt", met_pt>100)

        corner_cut = ((leading_jet.sieie<0.02) & (leading_jet.sipip<0.02)).any()
        selection.add("corner_cut", corner_cut)

        output = self.accumulator.identity()
        
        for region, cuts in self.regions.items():
            mask = selection.all(*cuts)
            output["ak4_abseta0"].fill(
                jeteta=leading_jet.abseta[mask].flatten(),
                region=region
            )

            if "corner" in region:
                output["ak4_phi0_eta0"].fill(
                    jetphi=leading_jet.phi[mask].flatten(),
                    jeteta=leading_jet.abseta[mask].flatten(),
                    region=region
                )
    
                output["met_phi_ak4_eta0"].fill(
                    metphi=met_phi[mask].flatten(),
                    jeteta=leading_jet.abseta[mask].flatten(),
                    region=region
                )
            
            output["sieie_sipip_ak4_abseta0"].fill(
                sieie=leading_jet.sieie[mask].flatten(),
                sipip=leading_jet.sipip[mask].flatten(),
                jeteta=leading_jet.abseta[mask].flatten(),
                region=region,
            )

            output["cssize_adssize_ak4_abseta0"].fill(
                cssize=leading_jet.cssize[mask].flatten(),
                adssize=leading_jet.adssize[mask].flatten(),
                jeteta=leading_jet.abseta[mask].flatten(),
                region=region,
            )

        return output

def main():
    # Input directory with .root files
    indir = sys.argv[1]
    tag = os.path.basename(indir)

    files = [pjoin(indir, f) for f in os.listdir(indir) if f.endswith('.root')]
    for file_idx, inpath in enumerate(tqdm(files)):
        f = uproot.open(inpath)
        tree = f['Events']

        df = LazyDataFrame(tree, flatten=True)
        
        ana = Analyzer()

        noise_enriched_region_cuts = [
            "at_least_one_jet",
            "leading_jet_pt",
            "leading_jet_eta",
            "no_other_high_jet_pt",
            "met_pt",
        ]

        ana.add_region('noise_enriched', noise_enriched_region_cuts)
        ana.add_region(
            'noise_enriched_corner', noise_enriched_region_cuts + ['corner_cut']
            )

        out = ana.analyze(df)

        # Save the output accumulator
        outdir = f"./output/coffea/{tag}"
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outpath = pjoin(outdir, f"output_{file_idx}.coffea")
        save(out, outpath)

if __name__ == '__main__':
    main()

