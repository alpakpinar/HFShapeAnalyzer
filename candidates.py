#!/usr/bin/env python
import numpy as np
import coffea.processor as processor

from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray


def setup_candidates(df):
    """Set up Physics candidates.

    Args:
        df ([type]): The dataframe containing the TTree data (must be flattened!)
    """
    jets = JaggedCandidateArray.candidatesfromcounts(
        df["nHLTJets"],
        pt=df["HLTJets_pt"],
        eta=df["HLTJets_eta"],
        phi=df["HLTJets_phi"],
        mass=df["HLTJets_phi"] * 0.,
        abseta=np.abs(df["HLTJets_eta"]),
        sieie=df["HLTJets_sigmaEtaEta"],
        sipip=df["HLTJets_sigmaPhiPhi"],
        cssize=df["HLTJets_centralEtaStripSize"],
        adssize=df["HLTJets_adjacentEtaStripsSize"]
    )

    muons = JaggedCandidateArray.candidatesfromcounts(
        df["nHLTMuon"],
        pt=df["HLTMuon_pt"],
        eta=df["HLTMuon_eta"],
        phi=df["HLTMuon_phi"],
        mass=df["HLTMuon_phi"]*0.,
        abseta=np.abs(df["HLTMuon_eta"]),
    )

    met_pt = df["HLTMET_pt"]
    met_phi = df["HLTMET_phi"]

    return jets, muons, met_pt, met_phi