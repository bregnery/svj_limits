"""
Scripts using building blocks in boosted_fits.py to create datacards
"""

import argparse, inspect, os, os.path as osp, re, json, itertools
from pprint import pprint
from time import strftime, sleep
from copy import copy

import numpy as np

import boosted_fits as bsvj

import ROOT # type: ignore
ROOT.RooMsgService.instance().setSilentMode(True)

_scripts = {}
def is_script(fn):
    _scripts[fn.__name__] = fn
    return fn

# _______________________________________________________________________




@is_script
def plot_scipy_fits():
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('rootfile', type=str)
    parser.add_argument('-o', '--plotdir', type=str, default='plots_bkgfits_%b%d')
    parser.add_argument('-b', '--bdtcut', type=float, default=None)
    parser.add_argument('-n', '--npars', type=int, nargs='*')
    parser.add_argument('-p', '--pdftype', type=str, default=None, choices=['main', 'alt'])

    args = parser.parse_args()
    plotdir = strftime(args.plotdir)
    if not osp.isdir(plotdir): os.makedirs(plotdir)

    import matplotlib.pyplot as plt # type: ignore
    bsvj.mpl_fontsizes()

    with bsvj.open_root(args.rootfile) as tf:

        def do_plot(tdir_name):
            tdir = tf.Get(tdir_name)
            bkg_hist = tdir.Get('Bkg')

            for pdf_type in ['main', 'alt']:
                if args.pdftype and pdf_type != args.pdftype: continue
                bsvj.logger.info('Fitting pdf_type=%s, tdir_name=%s', pdf_type, tdir_name)
                fig = plt.figure(figsize=(8,8))
                ax = fig.gca()
                binning, counts = bsvj.th1_binning_and_values(tdir.Get('Bkg'))
                bin_centers = np.array([.5*(l+r) for l, r in zip(binning[:-1], binning[1:])])
                # Bkg histogram
                ax.step(binning[:-1], counts, where='post', label='bkg {}'.format(tdir_name))
                # Fits
                if args.npars is not None and len(args.npars):
                    npars_iter = list(args.npars)
                else:
                    npars_iter = list(range(1,5) if pdf_type == 'alt' else range(2,6))
                for npars in npars_iter:
                    bsvj.logger.info('Fitting pdf_type=%s, tdir_name=%s, npars=%s', pdf_type, tdir_name, npars)
                    res = bsvj.fit_scipy(pdf_type, npars, bkg_hist)
                    y_pdf = bsvj.eval_expression(bsvj.pdf_expression(pdf_type, npars), [bin_centers] + list(res.x))
                    y_pdf = y_pdf/y_pdf.sum() * counts.sum()
                    chi2 = ((y_pdf-counts)**2 / y_pdf).sum() / (len(bin_centers) - npars)
                    label = '{}_npars{}, chi2={:.3f}, {}'.format(
                        pdf_type, npars, chi2,
                        ', '.join(['p{}={:.3f}'.format(i, v) for i, v in enumerate(res.x)])
                        )
                    ax.plot(bin_centers, y_pdf, label=label)
                ax.legend()
                ax.set_xlabel(r'$m_{T}$ (GeV)')
                ax.set_ylabel(r'$N_{events}$')
                ax.set_yscale('log')
                plt.savefig(osp.join(plotdir, tdir_name + '_' + pdf_type + '.png'), bbox_inches='tight')        

        bdtcut = None
        if args.bdtcut is not None:
            tdir_name = 'bsvj_{:.1f}'.format(args.bdtcut).replace('.', 'p')
            do_plot(tdir_name)
        else:
            for tdir_name in [k.GetName() for k in tf.GetListOfKeys()]:
                do_plot(tdir_name)


@is_script
def plot_roofit_fits():
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('rootfile', type=str)
    parser.add_argument('-o', '--plotdir', type=str, default='plots_bkgfits_%b%d')
    parser.add_argument('-b', '--bdtcut', type=float, default=None)
    parser.add_argument('-n', '--npars', type=int, nargs='*')
    parser.add_argument('-p', '--pdftype', type=str, default=None, choices=['main', 'alt'])
    args = parser.parse_args()
    plotdir = strftime(args.plotdir)
    if not osp.isdir(plotdir): os.makedirs(plotdir)

    with bsvj.open_root(args.rootfile) as tf:

        def do_plot(tdir_name):
            tdir = tf.Get(tdir_name)
            bkg_hist = tdir.Get('Bkg')
            mt = bsvj.get_mt_from_th1(bkg_hist)
            for pdf_type in ['main', 'alt']:
                if args.pdftype and pdf_type != args.pdftype: continue
                bsvj.logger.info('Fitting pdf_type=%s, tdir_name=%s', pdf_type, tdir_name)
                if args.npars is not None and len(args.npars):
                    npars_iter = list(args.npars)
                else:
                    npars_iter = list(range(1,5) if pdf_type == 'alt' else range(2,6))
                for npars in npars_iter:
                    bsvj.logger.info('Fitting pdf_type=%s, tdir_name=%s, npars=%s', pdf_type, tdir_name, npars)
                    res_scipy = bsvj.fit_scipy(pdf_type, npars, bkg_hist)
                    if len(res_scipy.x) != npars:
                        raise Exception(
                            'Wrong number of fitted parameters.'
                            ' Found {} parameters in scipy fit result, but npars is {}.'
                            ' Scipy fit result:\n{}'
                            .format(len(res_scipy.x), npars, res_scipy)
                            )
                    res_roofit_only = bsvj.fit_roofit(pdf_type, npars, bkg_hist)
                    res_roofit_wscipy = bsvj.fit_roofit(pdf_type, npars, bkg_hist, init_vals=res_scipy.x)
                    bsvj.plot_pdf_for_various_fitresults(
                        bsvj.make_pdf(pdf_type, npars, bkg_hist, mt=mt, name=bsvj.uid()),
                        [res_scipy, res_roofit_only, res_roofit_wscipy],
                        bsvj.th1_to_datahist(bkg_hist, mt=mt),
                        osp.join(plotdir, '{0}_{1}_npar{2}.png'.format(tdir.GetName(), pdf_type, npars)),
                        labels=['Scipy only', 'RooFit only', 'RooFit init w/ scipy']
                        )
                    print('-'*60)
                    print('Summary of varous fit strategies')
                    print('\nScipy only:')
                    print(res_scipy)
                    print('\nRooFit with initial parameters from Scipy:')
                    res_roofit_wscipy.Print()
                    print('\nRooFit with initial parameters set to 1.:')
                    res_roofit_only.Print()

        if args.bdtcut is not None:
            tdir_name = 'bsvj_{:.1f}'.format(args.bdtcut).replace('.', 'p')
            do_plot(tdir_name)
        else:
            for tdir_name in [k.GetName() for k in tf.GetListOfKeys()]:
                do_plot(tdir_name)



def this_fn_name():
    """
    Returns the name of whatever function this function was called from.
    (inspect.stack()[0][3] would be "this_fn_name"; [3] is just the index for names)
    """
    return inspect.stack()[1][3]


def gen_datacard_worker(args):
    kwargs = args.pop()
    bsvj.gen_datacard(*args, **kwargs)


@is_script
def gen_datacards_mp():
    parser = argparse.ArgumentParser(this_fn_name())
    parser.add_argument('jsonfile', type=str)
    parser.add_argument('--bdtcut', type=float, nargs='*')
    parser.add_argument('--mz', type=int, nargs='*')
    parser.add_argument('--rinv', type=float, nargs='*')
    parser.add_argument('-i', '--injectsignal', action='store_true')
    parser.add_argument('--nthreads', type=int, default=10)
    parser.add_argument('--tag', type=str, help='string suffix to outdir')
    parser.add_argument('--minmt', type=float, default=180.)
    parser.add_argument('--maxmt', type=float, default=720.)
    args = parser.parse_args()

    input = bsvj.InputData(args.jsonfile)
    bdtcuts = input.d['histograms'].keys()
    signals = [h.metadata for h in input.d['histograms']['0.100'].values() if 'mz' in h.metadata]

    # Filter for selected bdtcuts
    if args.bdtcut:
        use_bdtcuts = set('{:.3f}'.format(b) for b in args.bdtcut)
        bdtcuts = [b for b in bdtcuts if b in use_bdtcuts]
    # Filter for selected mzs
    if args.mz:
        use_mzs = set(args.mz)
        signals = [s for s in signals if s['mz'] in use_mzs]
    # Filter for selected rinvs
    if args.rinv:
        use_rinvs = set(args.rinv)
        signals = [s for s in signals if s['rinv'] in use_rinvs]
        
    combinations = list(itertools.product(bdtcuts, signals))
    bsvj.logger.info('Running %s combinations', len(combinations))

    if len(combinations) == 1:
        bsvj.gen_datacard(args.jsonfile, *combinations[0])
    else:
        import multiprocessing as mp
        with mp.Manager() as manager:
            pool = mp.Pool(args.nthreads)
            lock = manager.Lock()
            mp_args = []
            for bdtcut, signal in combinations:
                mp_args.append([
                    args.jsonfile, bdtcut, signal,
                    dict(
                        lock = lock,
                        injectsignal = args.injectsignal,
                        mt_min = args.minmt,
                        mt_max = args.maxmt,
                        tag = args.tag
                        )
                    ])
            pool.map(gen_datacard_worker, mp_args)
            pool.close()


@is_script
def simple_test_fit():
    """
    Runs a simple AsymptoticLimits fit on a datacard, without many options
    """
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('datacard', type=str)
    parser.add_argument('-c', '--chdir', type=str, default=None)
    args = parser.parse_args()

    cmd = bsvj.CombineCommand(args.datacard, 'AsymptoticLimits')
    cmd.track_parameters.extend(['r'])
    cmd.args.add('--saveWorkspace')
    cmd.set_parameter('pdf_index', 1)
    cmd.freeze_parameters.extend([
        'pdf_index',
        'bsvj_bkgfitmain_npars4_p1', 'bsvj_bkgfitmain_npars4_p2', 'bsvj_bkgfitmain_npars4_p3',
        'bsvj_bkgfitmain_npars4_p4',
        # 'bsvj_bkgfitalt_npars3_p1', 'bsvj_bkgfitalt_npars3_p2', 'bsvj_bkgfitalt_npars3_p3'
        ])
    bsvj.run_combine_command(cmd, args.chdir)


@is_script
def multidimfit():
    """
    Runs a single MultiDimFit on a datacard
    """
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('datacard', type=str)
    parser.add_argument('-c', '--chdir', type=str, default=None)
    parser.add_argument('-a', '--asimov', action='store_true')
    args = parser.parse_args()

    dc = bsvj.read_dc(args.datacard)

    cmd = bsvj.CombineCommand(args.datacard, 'MultiDimFit')
    cmd.args.add('--saveWorkspace')
    cmd.args.add('--saveNLL')
    if args.asimov:
        cmd.kwargs['-t'] = '-1'
        cmd.args.add('--toysFreq')
    cmd.set_parameter('pdf_index', 1)
    
    cmd.freeze_parameters.append('pdf_index')
    cmd.freeze_parameters.extend(dc.syst_rgx('bsvj_bkgfitmain_npars*'))

    cmd.redefine_signal_pois.append('r')
    cmd.kwargs['--X-rtd'] = 'REMOVE_CONSTANT_ZERO_POINT=1'
    cmd.track_parameters.extend(['r'])

    bsvj.run_combine_command(cmd, args.chdir)


@is_script
def likelihood_scan():
    """
    Runs a likelihood scan on a datacard
    """
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('datacard', type=str)
    parser.add_argument('-c', '--chdir', type=str, default=None)
    parser.add_argument('-a', '--asimov', action='store_true')
    parser.add_argument('--injectsignal', action='store_true')
    parser.add_argument('-n', '--npoints', type=int, default=51)
    parser.add_argument('-r', '--range', type=float, default=[-.7, .7], nargs=2)
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('--pdf', type=str, default='main', choices=['main', 'alt'])
    args = parser.parse_args()
    cmd = bsvj.likelihood_scan_factory(
        args.datacard, args.range[0], args.range[1], args.npoints,
        args.verbosity, args.asimov, pdf_type=args.pdf
        )
    if args.injectsignal: cmd.kwargs['-n'] += 'InjectedSig'
    bsvj.run_combine_command(cmd, args.chdir)


def likelihood_scan_multiple_worker(input):
    """
    Worker function for likelihood_scan_multiple multiprocessing
    """
    datacard, args = input
    cmd = bsvj.likelihood_scan_factory(
        datacard, args.minmu, args.maxmu, args.npoints,
        args.verbosity, args.asimov
        )
    cmd.name = cmd.name + '_' + osp.basename(datacard).replace('.txt', '')
    output = bsvj.run_combine_command(cmd)
    # Stageout
    output_file = osp.join(args.outdir, cmd.outfile.replace('.root','') + '.out')
    with open(output_file, 'w') as f:
        f.write(''.join(output))
    if osp.isfile(cmd.outfile): os.rename(cmd.outfile, osp.join(args.outdir, cmd.outfile))
    bsvj.logger.info('Finished scan for %s', datacard)


@is_script
def likelihood_scan_multiple():
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('datacards', type=str, nargs='+')
    parser.add_argument('-c', '--chdir', type=str, default=None)
    parser.add_argument('-a', '--asimov', action='store_true')
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('-n', '--npoints', type=int, default=201)
    parser.add_argument('--minmu', type=float, default=-.5)
    parser.add_argument('--maxmu', type=float, default=-.5)
    parser.add_argument('-o', '--outdir', type=str, default=strftime('scans_%b%d'))
    args = parser.parse_args()

    if not osp.isdir(args.outdir): os.makedirs(args.outdir)

    data = [ (d, args) for d in args.datacards ]

    import multiprocessing
    p = multiprocessing.Pool(16)
    p.map(likelihood_scan_multiple_worker, data)
    p.close()
    p.join()
    bsvj.logger.info('Finished pool')


@is_script
def printws():
    """
    Prints a workspace contents
    """
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('rootfile', type=str)
    parser.add_argument('-w', '--workspace', type=str)
    args = parser.parse_args()
    with bsvj.open_root(args.rootfile) as f:
        ws = bsvj.get_ws(f, args.workspace)
    ws.Print()
    return ws



if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str, choices=list(_scripts.keys()))
    parser.add_argument('-v', '--verbose', action='store_true')
    global_args, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args
    if global_args.verbose: bsvj.debug()
    r = _scripts[global_args.script]()