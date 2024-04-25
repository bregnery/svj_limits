"""
Scripts using building blocks in boosted_fits.py to create datacards
"""

import argparse, inspect, os, os.path as osp, re, json, itertools, sys, shutil
from pprint import pprint
from time import strftime, sleep
from copy import copy

import numpy as np

import boosted_fits as bsvj

import ROOT # type: ignore
ROOT.RooMsgService.instance().setSilentMode(True)

scripter = bsvj.Scripter()


@scripter
def plot_scipy_fits():
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('rootfile', type=str)
    parser.add_argument('-o', '--plotdir', type=str, default='plots_bkgfits_%b%d')
    parser.add_argument('-b', '--bdtcut', type=float, default=None)
    parser.add_argument('-n', '--npars', type=int, nargs='*')
    parser.add_argument('-p', '--pdftype', type=str, default=None, choices=['main', 'ua2'])

    args = parser.parse_args()
    plotdir = strftime(args.plotdir)
    if not osp.isdir(plotdir): os.makedirs(plotdir)

    import matplotlib.pyplot as plt # type: ignore
    bsvj.mpl_fontsizes()

    with bsvj.open_root(args.rootfile) as tf:

        def do_plot(tdir_name):
            tdir = tf.Get(tdir_name)
            bkg_hist = tdir.Get('Bkg')

            for pdf_type in ['main', 'alt', 'ua2']:
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


@scripter
def plot_roofit_fits():
    parser = argparse.ArgumentParser(inspect.stack()[0][3])
    parser.add_argument('rootfile', type=str)
    parser.add_argument('-o', '--plotdir', type=str, default='plots_bkgfits_%b%d')
    parser.add_argument('-b', '--bdtcut', type=float, default=None)
    parser.add_argument('-n', '--npars', type=int, nargs='*')
    parser.add_argument('-p', '--pdftype', type=str, default=None, choices=['main', 'alt', 'ua2'])
    args = parser.parse_args()
    plotdir = strftime(args.plotdir)
    if not osp.isdir(plotdir): os.makedirs(plotdir)

    with bsvj.open_root(args.rootfile) as tf:

        def do_plot(tdir_name):
            tdir = tf.Get(tdir_name)
            bkg_hist = tdir.Get('Bkg')
            mt = bsvj.get_mt_from_th1(bkg_hist)
            for pdf_type in ['main', 'alt', 'ua2']:
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


@scripter
def gen_datacards_mp():
    parser = argparse.ArgumentParser(this_fn_name())
    parser.add_argument('jsonfile', type=str)
    parser.add_argument('--bdtcut', type=float, nargs='*')
    parser.add_argument('--mz', type=int, nargs='*')
    parser.add_argument('--rinv', type=float, nargs='*')
    parser.add_argument('--mdark', type=int, nargs='*')
    parser.add_argument('-i', '--injectsignal', action='store_true')
    parser.add_argument('--nthreads', type=int, default=10)
    parser.add_argument('--tag', type=str, help='string suffix to outdir')
    parser.add_argument('--minmt', type=float, default=180.)
    parser.add_argument('--maxmt', type=float, default=650.)
    parser.add_argument('--trigeff', type=int, default=None, choices=[2016, 2017, 2018])
    args = parser.parse_args()

    input = bsvj.InputData(args.jsonfile)
    bdtcuts = input.d['histograms'].keys()

    signals = []
    for key, hist in input.d['histograms']['0.000'].items():
    #for key, hist in input.d['histograms']['0.0'].items():
        if 'mz' in hist.metadata and not key.startswith('SYST_'):
            signals.append(hist.metadata)

    # Filter for selected bdtcuts
    if args.bdtcut:
        use_bdtcuts = set('{:.3f}'.format(b) for b in args.bdtcut)
        bdtcuts = [b for b in bdtcuts if b in use_bdtcuts]
    # Filter for selected mzs
    if args.mz:
        use_mzs = set(args.mz)
        signals = [s for s in signals if int(s['mz']) in use_mzs]
    # Filter for selected rinvs
    if args.rinv:
        use_rinvs = set(args.rinv)
        signals = [s for s in signals if float(s['rinv']) in use_rinvs]
    # Filter for selected mdarks
    if args.mdark:
        use_mdarks = set(args.mdark)
        signals = [s for s in signals if int(s['mdark']) in use_mdarks]

    combinations = list(itertools.product(bdtcuts, signals))
    bsvj.logger.info('Running %s combinations', len(combinations))

    if len(combinations) == 1:
        bsvj.gen_datacard(args.jsonfile, *combinations[0], trigeff=args.trigeff)
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
                        tag = args.tag,
                        trigeff=args.trigeff
                        )
                    ])
            pool.map(gen_datacard_worker, mp_args)
            pool.close()

@scripter
def gen_datacards_v2(args=None):
    if args is None:
        lock = None
        json_files = bsvj.pull_arg('jsonfiles', nargs='+', type=str).jsonfiles
    else:
        json_files, lock = args

    if len(json_files) == 1:
        bsvj.InputDataV2(json_files[0]).gen_datacard(fit_cache_lock=lock)
    else:
        import multiprocessing as mp
        with mp.Manager() as manager:
            pool = mp.Pool(8)
            lock = manager.Lock()
            mp_args = [ ([f], lock) for f in json_files ]
            pool.map(gen_datacards_v2, mp_args)
            pool.close()


@scripter
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


# @scripter
# def multidimfit():
#     """
#     Runs a single MultiDimFit on a datacard
#     """
#     parser = argparse.ArgumentParser(inspect.stack()[0][3])
#     parser.add_argument('datacard', type=str)
#     parser.add_argument('pdf', type=str, choices=['main', 'alt'])
#     parser.add_argument('-c', '--chdir', type=str, default=None)
#     parser.add_argument('-a', '--asimov', action='store_true')
#     args, other_args = parser.parse_known_args()

#     dc = bsvj.read_dc(args.datacard)

#     cmd = bsvj.CombineCommand(args.datacard, 'MultiDimFit', raw=' '.join(other_args))
#     cmd.args.add('--saveWorkspace')
#     cmd.args.add('--saveNLL')
#     if args.asimov:
#         cmd.kwargs['-t'] = '-1'
#         cmd.args.add('--toysFrequentist')
#     cmd.set_parameter('pdf_index', 1 if args.pdf=='alt' else 0)
    
#     pdf_pars = dc.syst_rgx('bsvj_bkgfit%s_npars*' % args.pdf)
#     other_pdf = 'main' if args.pdf == 'alt' else 'alt'
#     other_pdf_pars = dc.syst_rgx('bsvj_bkgfit%s_npars*' % other_pdf)

#     cmd.freeze_parameters.append('pdf_index')
#     cmd.freeze_parameters.extend(pdf_pars)

#     cmd.redefine_signal_pois.append('r')
#     cmd.kwargs['--X-rtd'] = 'REMOVE_CONSTANT_ZERO_POINT=1'
#     cmd.track_parameters.extend(['r'] + other_pdf_pars)

#     bsvj.run_combine_command(cmd, args.chdir)




def make_bestfit_and_scan_commands(txtfile, args=None):
    if args is None: args = sys.argv[1:]
    with bsvj.set_args(sys.argv[:1] + args):
        dc = bsvj.Datacard.from_txt(txtfile)
        cmd = bsvj.CombineCommand(dc)
        cmd.configure_from_command_line()
        cmd.name += osp.basename(dc.filename).replace('.txt','')
        scan = bsvj.scan(cmd)
        bestfit = bsvj.bestfit(cmd)
        scan.name += 'Scan'
        bestfit.name += 'Bestfit'
    return bestfit, scan


@scripter
def bestfit(txtfile=None):
    if txtfile is None:
        # Allow multiprocessing if multiple datacards are passed on the command line
        txtfiles = bsvj.pull_arg('datacards', type=str, nargs='+').datacards
        if len(txtfiles) > 1:
            # Call this function in a pool instead            
            import multiprocessing as mp
            with mp.Manager() as manager:
                pool = mp.Pool(8)
                pool.map(bestfit, txtfiles)
                pool.close()
            return
        else:
            txtfile = osp.abspath(txtfiles[0])

    dc = bsvj.Datacard.from_txt(txtfile)
    cmd = bsvj.CombineCommand(dc)
    cmd.configure_from_command_line()
    cmd = bsvj.bestfit(cmd)
    cmd.raw = ' '.join(sys.argv[1:])
    cmd.name += 'Bestfit_' + osp.basename(txtfile).replace('.txt','')
    bsvj.run_combine_command(cmd, logfile=cmd.logfile)

    outdir = bsvj.pull_arg('-o', '--outdir', type=str, default=strftime('bestfits_%Y%m%d')).outdir
    outdir = osp.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    bsvj.logger.info(f'{cmd.outfile} -> {osp.join(outdir, osp.basename(cmd.outfile))}')
    shutil.move(cmd.outfile, osp.join(outdir, osp.basename(cmd.outfile)))
    shutil.move(cmd.logfile, osp.join(outdir, osp.basename(cmd.logfile)))


@scripter
def gentoys():
    """
    Generate toys from datacards
    """
    datacards = bsvj.pull_arg('datacards', type=str, nargs='+').datacards
    outdir = bsvj.pull_arg('-o', '--outdir', type=str, default=strftime('toys_%Y%m%d')).outdir
    if not osp.isdir(outdir): os.makedirs(outdir)
    bsvj.logger.info(f'Output will be moved to {outdir}')

    for dc_file in datacards:
        dc = bsvj.Datacard.from_txt(dc_file)
        cmd = bsvj.CombineCommand(dc)
        cmd.configure_from_command_line()
        cmd.name += osp.basename(dc.filename).replace('.txt','')

        # Some specific settings for toy generation
        cmd.method = 'GenerateOnly'
        cmd.args.add('--saveToys')
        cmd.args.add('--bypassFrequentistFit')
        cmd.args.add('--saveWorkspace')
        # Possibly delete some settings too
        cmd.kwargs.pop('--algo', None)
        cmd.track_parameters = set()

        assert '-t' in cmd.kwargs
        assert '--expectSignal' in cmd.kwargs

        bsvj.run_combine_command(cmd)
        bsvj.logger.info(f'Moving {cmd.outfile} -> {osp.join(outdir, osp.basename(cmd.outfile))}')
        shutil.move(cmd.outfile, osp.join(outdir, osp.basename(cmd.outfile)))


@scripter
def fittoys2():
    infiles = bsvj.pull_arg('infiles', type=str, nargs='+').infiles
    outdir = bsvj.pull_arg('-o', '--outdir', type=str, default=strftime('fittoys_%Y%m%d')).outdir
    if not osp.isdir(outdir): os.makedirs(outdir)
    bsvj.logger.info(f'Output will be moved to {outdir}')

    # Sort datacards and toysfiles
    datacards = []
    toysfiles = []
    for f in infiles:
        if 'GenerateOnly' in f:
            toysfiles.append(f)
        else:
            datacards.append(f)

    # Submit fit per datacard
    for dc_file in datacards:
        name = osp.basename(dc_file).replace('.txt','')
        dc = bsvj.Datacard.from_txt(dc_file)
        cmd = bsvj.CombineCommand(dc)
        cmd.configure_from_command_line()
        cmd = bsvj.bestfit(cmd)
        cmd.name += name

        for tf in toysfiles:
            if name in tf:
                cmd.kwargs['--toysFile'] = tf
                break
        else:
            raise Exception(
                f'Could not find a toy file for datacard {dc_file}; available toy files:\n'
                + "\n".join(toysfiles)
                )

        bsvj.run_combine_command(cmd)
        bsvj.logger.info(f'{cmd.outfile} -> {osp.join(outdir, osp.basename(cmd.outfile))}')
        shutil.move(cmd.outfile, osp.join(outdir, osp.basename(cmd.outfile)))


@scripter
def fittoys():
    # cmdFit="combine ${DC_NAME_ALL}
    #    -M FitDiagnostics
    #    -n ${fitName}
    #    --toysFile higgsCombine${genName}.GenerateOnly.mH120.123456.root
    #    -t ${nTOYS}
    #    -v
    #    -1
    #    --toysFrequentist
    #    --saveToys
    #    --expectSignal ${expSig}
    #    --rMin ${rMin}
    #    --rMax ${rMax}
    #    --savePredictionsPerToy
    #    --bypassFrequentistFit
    #    --X-rtd MINIMIZER_MaxCalls=100000
    #    --setParameters $SetArgFitAll
    #    --freezeParameters $FrzArgFitAll
    #    --trackParameters $TrkArgFitAll"

    datacards = bsvj.pull_arg('datacards', type=str, nargs='+').datacards
    outdir = bsvj.pull_arg('-o', '--outdir', type=str, default=strftime('toyfits_%b%d')).outdir
    if not osp.isdir(outdir): os.makedirs(outdir)

    for dc_file in datacards:
        dc = bsvj.Datacard.from_txt(dc_file)
        cmd = bsvj.CombineCommand(dc)
        cmd.configure_from_command_line()
        cmd.name += osp.basename(dc.filename).replace('.txt','')

        cmd.method = 'FitDiagnostics'
        cmd.kwargs.pop('--algo', None)
        cmd.args.add('--toysFrequentist')
        cmd.args.add('--saveToys')
        cmd.args.add('--savePredictionsPerToy')
        cmd.args.add('--bypassFrequentistFit')
        cmd.kwargs['--X-rtd'] = 'MINIMIZER_MaxCalls=100000'

        toysFile = bsvj.pull_arg('--toysFile', required=True, type=str).toysFile
        cmd.kwargs['--toysFile'] = toysFile

        if not '-t' in cmd.kwargs:
            with bsvj.open_root(toysFile) as f:
                cmd.kwargs['-t'] = f.Get('limit').GetEntries()

        assert '-t' in cmd.kwargs
        assert '--expectSignal' in cmd.kwargs

        bsvj.run_combine_command(cmd)
        os.rename(cmd.outfile, osp.join(outdir, osp.basename(cmd.outfile)))

        fit_diag_file = 'fitDiagnostics{}.root'.format(cmd.name)
        os.rename(fit_diag_file, osp.join(outdir, fit_diag_file))



@scripter
def impacts(dc_file=None):
    if dc_file is None:
        # Allow multiprocessing if multiple datacards are passed on the command line
        dc_files = bsvj.pull_arg('datacards', type=str, nargs='+').datacards
        if len(dc_files) > 1:
            # Call this function in a pool instead
            import multiprocessing as mp
            with mp.Manager() as manager:
                pool = mp.Pool(8)
                pool.map(impacts, dc_files)
                pool.close()
            return
        else:
            dc_file = osp.abspath(dc_files[0])

    dc = bsvj.Datacard.from_txt(dc_file)
    base_cmd = bsvj.CombineCommand(dc)
    base_cmd.configure_from_command_line()
    # HIER VERDER

    workdir = strftime(f'impacts_cli_%Y%m%d_{osp.basename(dc_file).replace(".txt","")}')
    bsvj.logger.info(f'Executing from {workdir}')
    if not bsvj.DRYMODE:
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)

    # Initial fit
    cmd = base_cmd.copy()
    cmd.kwargs['--algo'] = 'singles'
    cmd.kwargs['--redefineSignalPOIs'] = 'r'
    cmd.kwargs['--floatOtherPOIs'] = 1
    cmd.kwargs['--saveInactivePOI'] = 1
    cmd.kwargs['--robustFit'] = 1
    cmd.kwargs['--rMin'] = -2.4
    # cmd.kwargs['--rMax'] = 10.
    cmd.kwargs['-t'] = -1
    cmd.kwargs['--expectSignal'] = 0.2
    cmd.args.add('--saveWorkspace')
    cmd.add_range('shapeBkg_roomultipdf_bsvj__norm', 0.1, 2.0)
    cmd.name = '_initialFit_Test'
    if osp.isfile(cmd.outfile):
        bsvj.logger.warning(
            f'Initial fit output already exists, not running initial fit command: {cmd}'
            )
    else:
        bsvj.run_combine_command(cmd, logfile=cmd.logfile)
    initial_fit_outfile = cmd.outfile

    systs = []
    for syst in dc.syst_names:
        if 'mcstat' in syst: continue
        if syst in base_cmd.freeze_parameters: continue
        systs.append(syst)
    bsvj.logger.info(f'Doing systematics: {" ".join(systs)}')

    # Individual systs
    cmd = base_cmd.copy()
    cmd.kwargs['--algo'] = 'impact'
    cmd.kwargs['--redefineSignalPOIs'] = 'r'
    cmd.kwargs['--floatOtherPOIs'] = 1
    cmd.kwargs['--saveInactivePOI'] = 1
    cmd.kwargs['--robustFit'] = 1
    cmd.kwargs['--rMin'] = -10.
    cmd.kwargs['--rMax'] = 10.
    cmd.kwargs['-t'] = -1
    cmd.kwargs['--expectSignal'] = 0.2
    cmd.add_range('shapeBkg_roomultipdf_bsvj__norm', 0.1, 2.0)

    name = '_paramFit_Test_{0}'
    for syst in systs:
        if 'bkg_' in syst: continue # Ignore the bkg parameters
        cmd.name = name.format(syst)
        cmd.kwargs['-P'] = syst
        # bsvj.run_combine_command(cmd, logfile=cmd.logfile)

    # Create the impacts.json file
    combinetool_json_cmd = (
        f'combineTool.py -M Impacts'
        f' -d {initial_fit_outfile} -m 120 -o impacts.json'
        f' --named {",".join(systs)} --redefineSignalPOIs r'
        )
    # bsvj.run_command(combinetool_json_cmd)

    # Create the pdf
    plot_impacts_cmd = (
        f'plotImpacts.py'
        f' -i impacts.json --label-size 0.047'
        f' -o impacts'
        )
    # bsvj.run_command(plot_impacts_cmd)

    # Also a version without the bkg parameters
    systs = [s for s in systs if 'bkgfit' not in s]
    combinetool_json_cmd = (
        f'combineTool.py -M Impacts'
        f' -d {initial_fit_outfile} -m 120 -o impacts_nobkg.json'
        f' --named {",".join(systs)} --redefineSignalPOIs r'
        )
    bsvj.run_command(combinetool_json_cmd)

    plot_impacts_cmd = (
        f'plotImpacts.py'
        f' -i impacts_nobkg.json --label-size 0.047'
        f' -o impacts_nobkg'
        )
    bsvj.run_command(plot_impacts_cmd)



@scripter
def likelihood_scan(args=None):
    """
    Runs a likelihood scan on a datacard
    """
    if args is None: args = sys.argv
    with bsvj.set_args(args):

        print(sys.argv)

        outdir = bsvj.pull_arg('-o', '--outdir', type=str).outdir
        txtfile = bsvj.pull_arg('datacard', type=str).datacard
        bestfit, scan = make_bestfit_and_scan_commands(txtfile)

        if outdir and not osp.isdir(outdir): os.makedirs(outdir)

        for cmd in [bestfit, scan]:
            bsvj.run_combine_command(cmd, logfile=cmd.logfile)
            if outdir is not None:
                if osp.isfile(cmd.logfile):
                    os.rename(cmd.logfile, osp.join(outdir, osp.basename(cmd.logfile)))
                else:
                    bsvj.logger.error('No logfile %s', cmd.logfile)
                if osp.isfile(cmd.outfile):
                    os.rename(cmd.outfile, osp.join(outdir, osp.basename(cmd.outfile)))
                else:
                    bsvj.logger.error('No outfile %s', cmd.outfile)
            else:
                bsvj.logger.error('No outdir specified')


@scripter
def likelihood_scan_mp():
    """
    Like likelihood_scan, but accepts multiple datacards. 
    """
    datacards = bsvj.pull_arg('datacards', type=str, nargs='+').datacards
    outdir = bsvj.pull_arg('-o', '--outdir', type=str, default=strftime('scans_%Y%m%d')).outdir
    if not osp.isdir(outdir): os.makedirs(outdir)

    # Copy sys.argv per job, setting first argument to the datacard
    args = sys.argv[:]
    args.insert(1, datacards[0])
    args.extend(['--outdir', outdir])
    jobs = []
    for txtfile in datacards:
        args[1] = txtfile
        jobs.append(args[:])

    import multiprocessing
    p = multiprocessing.Pool(16)
    p.map(likelihood_scan, jobs)
    p.close()
    p.join()
    bsvj.logger.info('Finished pool')



# def likelihood_scan_multiple_worker(input):
#     """
#     Worker function for likelihood_scan_multiple multiprocessing
#     """
#     datacard, args, other_args = input
#     cmd = bsvj.likelihood_scan_factory(
#         datacard, args.minmu, args.maxmu, args.npoints,
#         args.verbosity, args.asimov,
#         raw = ' '.join(other_args)
#         )
#     cmd.name = cmd.name + '_' + osp.basename(datacard).replace('.txt', '')
#     output = bsvj.run_combine_command(cmd)
#     # Stageout
#     output_file = osp.join(args.outdir, cmd.outfile.replace('.root','') + '.out')
#     with open(output_file, 'w') as f:
#         f.write(''.join(output))
#     if osp.isfile(cmd.outfile): os.rename(cmd.outfile, osp.join(args.outdir, cmd.outfile))
#     bsvj.logger.info('Finished scan for %s', datacard)


# @scripter
# def likelihood_scan_multiple():
#     parser = argparse.ArgumentParser(inspect.stack()[0][3])
#     parser.add_argument('datacards', type=str, nargs='+')
#     parser.add_argument('-c', '--chdir', type=str, default=None)
#     parser.add_argument('-a', '--asimov', action='store_true')
#     parser.add_argument('-v', '--verbosity', type=int, default=0)
#     parser.add_argument('-n', '--npoints', type=int, default=201)
#     parser.add_argument('--minmu', type=float, default=-.5)
#     parser.add_argument('--maxmu', type=float, default=.5)
#     parser.add_argument('-o', '--outdir', type=str, default=strftime('scans_%b%d'))
#     args, other_args = parser.parse_known_args()

#     if not osp.isdir(args.outdir): os.makedirs(args.outdir)
#     data = [ (d, args, other_args) for d in args.datacards ]

#     import multiprocessing
#     p = multiprocessing.Pool(16)
#     p.map(likelihood_scan_multiple_worker, data)
#     p.close()
#     p.join()
#     bsvj.logger.info('Finished pool')


@scripter
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


@scripter
def remove_fsr():
    dc_files = bsvj.pull_arg('dcfiles', type=str, nargs='+').dcfiles
    for dc_file in dc_files:
        dc = bsvj.Datacard.from_txt(dc_file)
        dc.systs = [s for s in dc.systs if s[0] != 'fsr']
        bsvj.logger.info(f'Overwriting {dc_file}')
        with open(dc_file, 'w') as f:
            f.write(bsvj.parse_dc(dc))


if __name__ == '__main__':
    bsvj.debug(bsvj.pull_arg('-d', '--debug', action='store_true').debug)
    bsvj.drymode(bsvj.pull_arg('--dry', action='store_true').dry)
    scripter.run()
