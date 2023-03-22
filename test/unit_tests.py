from __future__ import print_function
import os, os.path as osp, sys, argparse, fnmatch
from pprint import pprint
from collections import OrderedDict
from contextlib import contextmanager
import inspect

import ROOT
import numpy as np

TEST_DIR = osp.dirname(osp.abspath(__file__))
TEST_ROOTFILE = osp.join(TEST_DIR, 'example_fit_result.root')

sys.path.append(osp.dirname(TEST_DIR))
import boosted_fits as bsvj


def get_ws(rootfile=None):
    if rootfile is None: rootfile = TEST_ROOTFILE
    with bsvj.open_root(rootfile) as f:
        return bsvj.get_ws(f)


_known_tests = OrderedDict()

def is_test(fn):
    def wrapper(*args, **kwargs):
        bsvj.logger.info('-'*80)
        bsvj.logger.info('\033[32mRunning test {}\033[0m'.format(fn.__name__))
        try:
            fn(*args, **kwargs)
        except:
            bsvj.logger.error('\033[31mTest {} failed\033[0m'.format(fn.__name__))
            raise
        else:
            bsvj.logger.info('\033[32mTest {} finished successfully\033[0m'.format(fn.__name__))
    _known_tests[fn.__name__] = wrapper
    return wrapper


@is_test
def test_roofit_get_y_values():
    # Values from text datacard used to create the example ws
    n_expected_data = 825105
    n_expected_sig = 24666

    ws = get_ws()
    ws.loadSnapshot('MultiDimFit')

    mt = ws.var('mt')
    mt_binning = bsvj.binning_from_roorealvar(mt)
    mt_bin_centers = .5*(mt_binning[1:]+mt_binning[:-1])

    # Test getting from RooDataSet
    data = ws.data('data_obs')
    x_data, y_data = bsvj.roodataset_values(data)

    assert abs(y_data.sum() - n_expected_data) <= 1.
    np.testing.assert_array_equal(mt_bin_centers, x_data)
    assert len(mt_binning)-1 == len(y_data)

    # Compare with values from createHistogram
    data_th1 = ROOT.RooDataHist(bsvj.uid(), '', ROOT.RooArgSet(mt), data).createHistogram(bsvj.uid(), mt)
    data_th1_binning, y_data_th1, errs_data_th1 = bsvj.th1_binning_and_values(data_th1, True)

    np.testing.assert_array_equal(mt_binning, data_th1_binning)
    np.testing.assert_array_equal(y_data, y_data_th1)
    np.testing.assert_array_equal(y_data, errs_data_th1)

    # Test getting from RooMultiPdf/RooAbsPdf
    bkg = ws.pdf('shapeBkg_roomultipdf_bsvj')
    y_bkg = bsvj.pdf_values(bkg, x_data)
    bkg_norm = ws.function('n_exp_final_binbsvj_proc_roomultipdf').getVal()

    # Compare with values from createHistogram
    bkg_th1 = bkg.createHistogram(bsvj.uid(), mt)
    bkg_th1_binning, y_bkg_th1 = bsvj.th1_binning_and_values(bkg_th1)

    assert len(y_bkg) == len(y_data)
    np.testing.assert_almost_equal(bkg_th1_binning, mt_binning)
    np.testing.assert_almost_equal(y_bkg, y_bkg_th1)

    # Test getting from RooDataHist
    sig = ws.embeddedData('shapeSig_sig_bsvj')
    x_sig, y_sig = bsvj.roodataset_values(sig)

    np.testing.assert_array_equal(mt_bin_centers, x_sig)
    assert abs(y_sig.sum() - n_expected_sig) <= 1.


@is_test
def test_eval_expression():
    assert bsvj.eval_expression('pow(@0, 2)', [2.]) == 4
    import numpy as np
    np.testing.assert_array_equal(
        bsvj.eval_expression('pow(@0, 2)', [np.array([2., 4.])]),
        np.array([4., 16.])
        )
    assert bsvj.add_normalization('@0*@1') == '@2*(@0*@1)'


@is_test
def test_chi2():
    import time
    from scipy import stats

    ws = get_ws()

    mt = ws.var('mt')
    pdf = ws.pdf('bsvj_bkgfitalt_npars1_rpsbp')
    data = ws.data('data_obs')

    res = pdf.fitTo(data, ROOT.RooFit.SumW2Error(True), ROOT.RooFit.Save(True))
    bsvj.logger.info('Fit result: %s', res)

    n_fit_parameters = res.floatParsFinal().getSize()
    bsvj.logger.info('n_fit_parameters: %s', n_fit_parameters)

    t0 = time.time()
    chi2_viaframe = bsvj.get_chi2_viaframe(mt, pdf, data, n_fit_parameters)
    t1 = time.time()
    rss_viaframe = bsvj.get_rss_viaframe(mt, pdf, data)
    t2 = time.time()

    mt_binning = bsvj.binning_from_roorealvar(mt)
    mt_bin_centers = .5*(mt_binning[:-1] + mt_binning[1:])
    _, y_data = bsvj.roodataset_values(data)
    y_pdf = bsvj.pdf_values(pdf, mt_bin_centers) * y_data.sum()
    raw_chi2 = ((y_pdf-y_data)**2 / y_pdf).sum()
    ndf = len(mt_bin_centers) - n_fit_parameters
    chi2 = raw_chi2 / ndf
    prob = stats.distributions.chi2.sf(raw_chi2, ndf) # or cdf?
    rss = np.sqrt(((y_pdf-y_data)**2).sum())
    t3 = time.time()

    bsvj.logger.info('chi2_viaframe: %s, took %s sec', chi2_viaframe, t1-t0)
    bsvj.logger.info(
        'chi2_manual: %s, took %s sec',
        (chi2, raw_chi2, prob, ndf),
        t3-t2
        )

    # ________________________________________________________
    # Extra test: Fit normalization manually to data again

    def fom(norm):
        y = y_pdf * norm
        return ((y-y_data)**2 / y).sum() / ndf
    from scipy.optimize import minimize
    res = minimize(fom, 1.)
    
    y_pdf = res.x * y_pdf
    raw_chi2 = ((y_pdf-y_data)**2 / y_pdf).sum()
    ndf = len(mt_bin_centers) - n_fit_parameters
    chi2 = raw_chi2 / ndf
    prob = stats.distributions.chi2.sf(raw_chi2, ndf) # or cdf?
    bsvj.logger.info('chi2_manual after fitting norm: %s',(chi2, raw_chi2, prob, ndf))

    # ________________________________________________________
    # RSS

    bsvj.logger.info('rss_viaframe: %s, took %s sec', rss_viaframe, t2-t1)
    bsvj.logger.info('rss_manual: %s', rss)



@is_test
def test_combine_command():
    cmd = bsvj.CombineCommand('bla.txt')
    cmd.kwargs['--test'] = .1
    cmd.kwargs['--test2'] = 'NO'
    cmd.args.add('--saveNLL')
    cmd.freeze_parameters.extend(['x', 'y'])
    assert cmd.parameters == {}
    assert cmd.freeze_parameters == ['x', 'y']
    assert cmd.dc == 'bla.txt'
    assert cmd.str == 'combine -M MultiDimFit bla.txt --saveNLL --test 0.1 --test2 NO --freezeParameters x,y'
    cmd.add_range('x', .4, 1.6)
    cmd.add_range('y', 100, 101)
    assert cmd.str == (
        'combine -M MultiDimFit bla.txt --saveNLL --test 0.1 --test2 NO --freezeParameters x,y'
        ' --setParameterRanges x=0.4,1.6:y=100,101'
        )

    assert cmd.name == ''
    cmd.kwargs['-n'] = 'test'
    assert cmd.get_name_key() == '-n'
    assert cmd.name == 'test'
    cmd.name += 'bla'
    assert cmd.name == 'testbla'
    cmd.kwargs['--name'] = cmd.kwargs['-n']
    del cmd.kwargs['-n']
    assert cmd.get_name_key() == '--name'


@is_test
def test_datacard():
    dc = bsvj.Datacard()
    dc.shapes.append(['roomultipdf', 'bsvj', 'my_ws.root', 'SVJ:$PROCESS'])
    dc.shapes.append(['sig', 'bsvj', 'my_ws.root', 'SVJ:$PROCESS', 'SVJ:$PROCESS_$SYSTEMATIC'])
    dc.shapes.append(['data_obs', 'bsvj', 'my_ws.root', 'SVJ:$PROCESS'])
    dc.channels.append(('bsvj', 100000))
    dc.rates['bsvj'] = OrderedDict()
    dc.rates['bsvj']['sig'] = 15000
    dc.rates['bsvj']['roomultipdf'] = 120000
    dc.systs.extend([
        ['bsvj_bkgfitmain_npars2_p1', 'flatParam'],
        ['bsvj_bkgfitmain_npars2_p2', 'flatParam'],
        ['bsvj_bkgfitalt_npars1_p1',  'flatParam'],
        ['pdf_index',                 'discrete'],
        ['lumi', 'lnN', 1.02, '-']
        ])
    txt = bsvj.parse_dc(dc)

    print(txt)
    print('\n')

    dc2 = bsvj.read_dc_txt(txt)
    txt2 = bsvj.parse_dc(dc2)
    
    print(txt2)

    assert dc.shapes == dc2.shapes
    assert dc.channels == dc2.channels
    assert dc.rates == dc2.rates
    assert dc.systs == dc2.systs    
    assert dc == dc2

    assert dc2.syst_rgx('bsvj_bkgfitmain_*') == ['bsvj_bkgfitmain_npars2_p1', 'bsvj_bkgfitmain_npars2_p2']
    


def test_fit_cache_mp_worker(tup):
    i, cache_file, lock = tup
    from fit_cache import FitCache, logger
    logger.handlers[0].formatter._fmt = str(i) + ':' + bsvj.logger.handlers[0].formatter._fmt
    cache = FitCache(cache_file, lock)
    cache.write(i, i) # Use i as both the hash and the result

@is_test
def test_fit_cache_mp():
    from fit_cache import FitCache
    import multiprocessing as mp

    cache_file = 'test_fit_cache.pickle'
    if osp.isfile(cache_file): os.remove(cache_file)

    with mp.Manager() as manager:
        lock = manager.Lock()
        mp_args = [ (i, cache_file, lock) for i in range(100)]
        pool = mp.Pool(4)
        pool.map(test_fit_cache_mp_worker, mp_args)
        pool.close()
        pool.join()

    cache = FitCache(cache_file)
    cache.read()
    available_keys = set(sorted(cache.cache.keys()))
    print(available_keys)
    assert available_keys == set(range(100))


@is_test
def test_poly1d():
    import requests
    parameters = np.array(
        requests.get(
            'https://raw.githubusercontent.com/boostedsvj/triggerstudy/main/bkg/bkg_trigeff_fit_2018.txt'
            ).json()
        )
    x = np.array([0., 200., 500., 800., 2000.])
    # Evaluate with numpy
    poly = np.poly1d(parameters)
    fit = lambda x: np.where(x<1000., 1./(1.+np.exp(-poly(x))), 1.)
    y_np = fit(x)
    # Evaluate the RooFit formula
    expr = bsvj.sigmoid(bsvj.poly1d(parameters))
    expr = '({0})*(@0<1000.) + (@0>=1000.)'.format(expr)
    y_eval = bsvj.eval_expression(expr, [x])
    # Should be mostly the same
    bsvj.logger.info('Evaluation with numpy: %s', y_np)
    bsvj.logger.info('Evaluation with formula: %s', y_eval)
    np.testing.assert_array_almost_equal(y_np, y_eval, decimal=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('patterns', nargs='*', type=str, help='Select only test containing any of these substrings')
    args = parser.parse_args()

    do_tests = set(_known_tests.keys())

    if args.patterns:
        do_tests_filtered = set()
        for pat in args.patterns:
            do_tests_filtered.update(fnmatch.filter(do_tests, '*'+pat+'*'))
        do_tests = do_tests_filtered

    for test_name, test_fn in _known_tests.items():
        if test_name in do_tests:
            test_fn()