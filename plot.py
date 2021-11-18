#!/usr/bin/env python

import os
import sys
import coffea.processor as processor

from coffea import hist
from coffea.util import load
from matplotlib import pyplot as plt
from klepto.archives import dir_archive

pjoin = os.path.join

class Saver():
    def __init__(self, outdir: str) -> None:
        self.outdir = outdir
        self._prepare_outdir()

    def _prepare_outdir(self):
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def save_plot(self, filename: str, fig: plt.figure):
        outpath = pjoin(self.outdir, filename)
        fig.savefig(outpath)
        plt.close(fig)
        print(f'Figure saved: {outpath}')

class Plotter():
    def __init__(self, acc: processor.dict_accumulator, dataset: str, run: int) -> None:
        self.acc = acc
        self.dataset = dataset
        self.run = run
        self.saver = Saver(outdir="output")
    
    def _decorate_plot(self, ax: plt.axis, region: str, plot_diag: bool=True, eta_range: tuple=None):
        ax.text(0,1,f'HLT (Run: {self.run})',
            fontsize=14,
            ha='left',
            va='bottom',
            transform=ax.transAxes
        )
        ax.text(1,1,self.dataset.replace('_', ' '),
            fontsize=14,
            ha='right',
            va='bottom',
            transform=ax.transAxes
        )
        # Diagonal line
        if plot_diag:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.plot(xlim, ylim, color='red', lw=2)

        # Text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        texts = {
            "noise_enriched" : "Noise enriched region",
            "noise_enriched_corner" : "Noise enriched region (corner)",
            "physics_enriched" : "Physics enriched region",
        }
        figtext = texts[region]
        if eta_range is not None:
            figtext +=  f"\n${eta_range[0]:.2f} < |\\eta| < {eta_range[1]:.2f}$"
        
        ax.text(0.05,0.95,figtext,
            fontsize=14,
            va="top",
            transform=ax.transAxes,
            bbox=props
        )
            

        return ax

    def plot2d(self, histoname: str, xaxis: str, regionname: str, etaslice: slice=None, plot_diag: bool=False):
        """Plot the given 2D histogram 

        Args:
            histoname (str): The name of the histogram.
            xaxis (str): Label of the x-axis to plot.
            regionname (str): The name of the region to be integrated over.
            etaslice (slice): The absolute eta slice for the tagged jet.
            plot_diag (bool): Plot a diagonal line in the figure (typically for sieie vs sipip plots)
        """
        self.acc.load(histoname)
        h = self.acc[histoname]
        h = h.integrate('region', regionname)
        if etaslice is None:
            h = h.integrate('jeteta')
        else:
            h = h.integrate('jeteta', etaslice)
        
        assert h.dim() == 2, f"The histogram passed to plot2d() must be 2 dimensional!"
        
        fig, ax = plt.subplots()
        opts = {
            "text_opts" : {},
            "density" : True
        }
        if "cssize" in histoname:
            hist.plot2d(h, xaxis=xaxis, ax=ax, density=True, patch_opts={}, text_opts={"format": "%.2f"})
        else:
            hist.plot2d(h, xaxis=xaxis, ax=ax, density=True)

        ax = self._decorate_plot(ax, regionname, plot_diag="sieie" in histoname, eta_range=(etaslice.start, etaslice.stop))

        if etaslice is None:
            etatag=''
        else:
            etatag=f'_{str(etaslice.start).replace(".","_")}_{str(etaslice.stop).replace(".","_")}'
        self.saver.save_plot(filename=f"{self.dataset}_{histoname}{etatag}_{regionname}.pdf", fig=fig)

    def plot1d(self, histoname: str, regionname: str, etaslice: slice=None):
        """Plot the given 1D histogram

        Args:
            histoname (str): The name of the histogram.
            regionname (str): The name of the region to be integrated over.
            etaslice (slice): The absolute eta slice for the tagged jet.
        """
        self.acc.load(histoname)
        h = self.acc[histoname]
        h = h.integrate('region', regionname)
        if etaslice is None:
            h = h.integrate('jeteta')
        else:
            h = h.integrate('jeteta', etaslice)
        
        assert h.dim() == 1, f"The histogram passed to plot1d() must be 1 dimensional!"
        
        fig, ax = plt.subplots()
        hist.plot1d(h, ax=ax)
        ax.get_legend().remove()
        
        if etaslice:
            ax = self._decorate_plot(ax, regionname, plot_diag=False, eta_range=(etaslice.start, etaslice.stop))
        else:
            ax = self._decorate_plot(ax, regionname, plot_diag=False)

        if etaslice is None:
            etatag=''
        else:
            etatag=f'_{str(etaslice.start).replace(".","_")}_{str(etaslice.stop).replace(".","_")}'
        self.saver.save_plot(filename=f"{self.dataset}_{histoname}{etatag}_{regionname}.pdf", fig=fig)


def main():
    # Path to the merged output directory
    inpath = sys.argv[1]
    acc = dir_archive(inpath)

    p = Plotter(acc, dataset="MET_2018", run=323790)
    slices = [
        slice(2.9,3.25),
        slice(3.25,4.0),
        slice(4.0,5.0),
    ]
    
    for etaslice in slices:
        p.plot2d(
            histoname="sieie_sipip_ak4_abseta0", 
            xaxis='sieie',
            regionname="noise_enriched",
            etaslice=etaslice
            )

        p.plot2d(
            histoname="cssize_adssize_ak4_abseta0", 
            xaxis='cssize',
            regionname="noise_enriched",
            etaslice=etaslice
            )

        p.plot1d(
            histoname="met_phi_ak4_eta0",
            regionname="noise_enriched_corner",
            etaslice=etaslice
        )
    
        p.plot1d(
            histoname="ak4_phi0_eta0",
            regionname="noise_enriched_corner",
            etaslice=etaslice
        )
    
    p.plot1d(
        histoname="met_phi_ak4_eta0",
        regionname="noise_enriched_corner"
    )

if __name__ == '__main__':
    main()