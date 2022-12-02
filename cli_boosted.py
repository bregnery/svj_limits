"""
Scripts using building blocks in boosted_fits.py to create datacards
"""

import argparse, inspect, os, os.path as osp, re, json
from pprint import pprint
from boosted_fits import *
from time import strftime, sleep
from copy import copy

import ROOT
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

    import matplotlib.pyplot as plt
    mpl_fontsizes()

    with open_root(args.rootfile) as tf:

        def do_plot(tdir_name):
            tdir = tf.Get(tdir_name)
            bkg_hist = tdir.Get('Bkg')

            for pdf_type in ['main', 'alt']:
                if args.pdftype and pdf_type != args.pdftype: continue
                logger.info('Fitting pdf_type=%s, tdir_name=%s', pdf_type, tdir_name)
                fig = plt.figure(figsize=(8,8))
                ax = fig.gca()
                binning, counts = th1_binning_and_values(tdir.Get('Bkg'))
                bin_centers = np.array([.5*(l+r) for l, r in zip(binning[:-1], binning[1:])])
                # Bkg histogram
                ax.step(binning[:-1], counts, where='post', label='bkg {}'.format(tdir_name))
                # Fits
                if args.npars is not None and len(args.npars):
                    npars_iter = list(args.npars)
                else:
                    npars_iter = list(range(1,5) if pdf_type == 'alt' else range(2,6))
                for npars in npars_iter:
                    logger.info('Fitting pdf_type=%s, tdir_name=%s, npars=%s', pdf_type, tdir_name, npars)
                    res = fit_scipy(pdf_type, npars, bkg_hist)
                    y_pdf = eval_expression(pdf_expression(pdf_type, npars), [bin_centers] + list(res.x))
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

    with open_root(args.rootfile) as tf:

        def do_plot(tdir_name):
            tdir = tf.Get(tdir_name)
            bkg_hist = tdir.Get('Bkg')
            mt = get_mt_from_th1(bkg_hist)
            for pdf_type in ['main', 'alt']:
                if args.pdftype and pdf_type != args.pdftype: continue
                logger.info('Fitting pdf_type=%s, tdir_name=%s', pdf_type, tdir_name)
                if args.npars is not None and len(args.npars):
                    npars_iter = list(args.npars)
                else:
                    npars_iter = list(range(1,5) if pdf_type == 'alt' else range(2,6))
                for npars in npars_iter:
                    logger.info('Fitting pdf_type=%s, tdir_name=%s, npars=%s', pdf_type, tdir_name, npars)
                    res_scipy = fit_scipy(pdf_type, npars, bkg_hist)
                    if len(res_scipy.x) != npars:
                        raise Exception(
                            'Wrong number of fitted parameters.'
                            ' Found {} parameters in scipy fit result, but npars is {}.'
                            ' Scipy fit result:\n{}'
                            .format(len(res_scipy.x), npars, res_scipy)
                            )
                    res_roofit_only = fit_roofit(pdf_type, npars, bkg_hist)
                    res_roofit_wscipy = fit_roofit(pdf_type, npars, bkg_hist, init_vals=res_scipy.x)
                    plot_pdf_for_various_fitresults(
                        make_pdf(pdf_type, npars, bkg_hist, mt=mt, name=uid()),
                        [res_scipy, res_roofit_only, res_roofit_wscipy],
                        th1_to_datahist(bkg_hist, mt=mt),
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


@is_script
def gen_datacard_ul(args=None):
    """
    Generate datacards for UL
    """
    if args is None:
        parser = argparse.ArgumentParser(this_fn_name())
        parser.add_argument('jsonfile', type=str)
        parser.add_argument('-b', '--bdtcut', type=float, default=.1)
        parser.add_argument('--mz', type=int, default=450, choices=[150, 250, 350, 450, 550])
        parser.add_argument('--rinv', type=float, default=0.3, choices=[.1, .3])
        parser.add_argument('-i', '--injectsignal', action='store_true')
        args = parser.parse_args()

    input = InputData(args.jsonfile)
    input = input.cut_mt(180., 720.)

    bdt_str = '{:.1f}'.format(args.bdtcut).replace('.', 'p')
    mt = get_mt(input.mt[0], input.mt[-1], input.n_bins, name='mt')
    bkg_th1 = input.bkg_th1('bkg', args.bdtcut)

    data_datahist = ROOT.RooDataHist("data_obs", "Data", ROOT.RooArgList(mt), bkg_th1, 1.)

    pdfs_dict = {
        'main' : pdfs_factory('main', mt, bkg_th1, name='bsvj_bkgfitmain'),
        'alt' : pdfs_factory('alt', mt, bkg_th1, name='bsvj_bkgfitalt'),
        }
    winner_pdfs = []
    for pdf_type in ['main', 'alt']:
        pdfs = pdfs_dict[pdf_type]
        ress = [ fit(pdf) for pdf in pdfs ]
        i_winner = do_fisher_test(mt, data_datahist, pdfs)
        winner_pdfs.append(pdfs[i_winner])
        plot_fits(pdfs, ress, data_datahist, pdf_type + '.pdf')

    systs = [
        ['lumi', 'lnN', 1.016, '-'],
        # Place holders
        ['trigger', 'lnN', 1.02, '-'],
        ['pdf', 'lnN', 1.05, '-'],
        ['mcstat', 'lnN', 1.07, '-'],
        ]

    sig_name = 'mz{:.0f}_rinv{:.1f}'.format(args.mz, args.rinv)
    sig_th1 = input.sig_th1(sig_name, args.bdtcut, args.mz, args.rinv)
    sig_datahist = ROOT.RooDataHist(sig_name, sig_name, ROOT.RooArgList(mt), sig_th1, 1.)

    # assert bkg_th1.GetNbinsX() == sig_th1.GetNbinsX()
    # assert bkg_th1.GetBinLowEdge(1) == sig_th1.GetBinLowEdge(1)
    # n = bkg_th1.GetNbinsX()
    # assert bkg_th1.GetBinLowEdge(n+1) == sig_th1.GetBinLowEdge(n+1)
    # assert sig_th1.GetBinLowEdge(n+1) == mt.getMax()
    # x_sig_datahist, y_sig_datahist = roodataset_values(sig_datahist)
    # np.testing.assert_almost_equal(x_sig_datahist, input.mt_centers)
    # np.testing.assert_almost_equal(y_sig_datahist, input.sighist(args.bdtcut, args.mz).vals, decimal=3)
    # print('All passed')
    # return

    if args.injectsignal:
        logger.info('Injecting signal in data_obs')
        data_datahist = ROOT.RooDataHist("data_obs", "Data", ROOT.RooArgList(mt), bkg_th1+sig_th1, 1.)

    compile_datacard_macro(
        winner_pdfs, data_datahist, sig_datahist,
        strftime('dc_%b%d/dc_mz{}_rinv{:.1f}_bdt{}.txt'.format(args.mz, args.rinv, bdt_str)),
        systs=systs
        )


@is_script
def simple_test_fit():
    """
    Runs a simple AsymptoticLimits fit on a datacard, without many options
    """
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('datacard', type=str)
    parser.add_argument('-c', '--chdir', type=str, default=None)
    args = parser.parse_args()

    cmd = CombineCommand(args.datacard, 'AsymptoticLimits')
    cmd.track_parameters.extend(['r'])
    cmd.args.add('--saveWorkspace')
    cmd.set_parameter('pdf_index', 1)
    cmd.freeze_parameters.extend([
        'pdf_index',
        'bsvj_bkgfitmain_npars4_p1', 'bsvj_bkgfitmain_npars4_p2', 'bsvj_bkgfitmain_npars4_p3',
        'bsvj_bkgfitmain_npars4_p4',
        # 'bsvj_bkgfitalt_npars3_p1', 'bsvj_bkgfitalt_npars3_p2', 'bsvj_bkgfitalt_npars3_p3'
        ])
    run_combine_command(cmd, args.chdir)


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

    dc = read_dc(args.datacard)

    cmd = CombineCommand(args.datacard, 'MultiDimFit')
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

    run_combine_command(cmd, args.chdir)


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
    cmd = likelihood_scan_factory(
        args.datacard, args.range[0], args.range[1], args.npoints,
        args.verbosity, args.asimov, pdf_type=args.pdf
        )
    if args.injectsignal: cmd.kwargs['-n'] += 'InjectedSig'
    run_combine_command(cmd, args.chdir)


def likelihood_scan_multiple_worker(input):
    """
    Worker function for likelihood_scan_multiple multiprocessing
    """
    datacard, args = input
    cmd = likelihood_scan_factory(
        datacard, args.range[0], args.range[1], args.npoints,
        args.verbosity, args.asimov
        )
    cmd.name = cmd.name + '_' + osp.basename(datacard).replace('.txt', '')
    output = run_combine_command(cmd)
    # Stageout
    output_file = osp.join(args.outdir, cmd.outfile.replace('.root','') + '.out')
    with open(output_file, 'w') as f:
        f.write(''.join(output))
    if osp.isfile(cmd.outfile): os.rename(cmd.outfile, osp.join(args.outdir, cmd.outfile))
    logger.info('Finished scan for %s', datacard)


@is_script
def likelihood_scan_multiple():
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('datacards', type=str, nargs='+')
    parser.add_argument('-c', '--chdir', type=str, default=None)
    parser.add_argument('-a', '--asimov', action='store_true')
    parser.add_argument('-v', '--verbosity', type=int, default=0)
    parser.add_argument('-n', '--npoints', type=int, default=401)
    parser.add_argument('-r', '--range', type=float, default=[-1., 10.], nargs=2)
    parser.add_argument('-o', '--outdir', type=str, default=strftime('scans_%b%d'))
    args = parser.parse_args()

    if not osp.isdir(args.outdir): os.makedirs(args.outdir)

    data = [ (d, args) for d in args.datacards ]

    import multiprocessing
    p = multiprocessing.Pool(16)
    p.map(likelihood_scan_multiple_worker, data)
    p.close()
    p.join()
    logger.info('Finished pool')


@is_script
def printws():
    """
    Prints a workspace contents
    """
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('rootfile', type=str)
    parser.add_argument('-w', '--workspace', type=str)
    args = parser.parse_args()
    with open_root(args.rootfile) as f:
        ws = get_ws(f, args.workspace)
    ws.Print()
    return ws



if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str, choices=list(_scripts.keys()))
    parser.add_argument('-v', '--verbose', action='store_true')
    global_args, other_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + other_args
    if global_args.verbose: debug()
    r = _scripts[global_args.script]()