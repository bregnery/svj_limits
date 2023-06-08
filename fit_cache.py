"""
This module implements a cache for fits, to avoid unneccesarily rerunning
the exact same fits over and over again.
"""

import os, os.path as osp, logging, pickle

from contextlib import contextmanager


DEFAULT_LOGGING_LEVEL = logging.INFO
def setup_logger(name='fitcache'):
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.info('Logger %s is already defined', name)
    else:
        fmt = logging.Formatter(
            fmt = '\033[33m%(levelname)s:%(asctime)s:%(module)s:%(lineno)s\033[0m %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
            )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(DEFAULT_LOGGING_LEVEL)
        logger.addHandler(handler)
    return logger
logger = setup_logger()


class nulllock:
    def acquire(self):
        pass

    def release(self):
        pass


class FitCache:
    def __init__(self, cache_file='fit_cache.pickle', lock=None):
        self.cache_file = cache_file
        self.lock = nulllock() if lock is None else lock
        self.cache = {}
        self._has_lock = False

    @contextmanager
    def lock_context(self):
        try:
            self.lock.acquire()
            logger.debug('Acquired lock')
            yield None
        finally:
            self.lock.release()
            logger.debug('Released lock')

    def read(self):
        if not osp.isfile(self.cache_file): return
        with open(self.cache_file, 'rb') as f:
            self.cache = pickle.load(f)

    def get(self, fithash):
        # First see if it's in the existing cache:
        if fithash in self.cache: return self.cache[fithash]
        # Otherwise, update the cache and try again
        with self.lock_context():
            self.read()
        if fithash in self.cache: return self.cache[fithash]
        # No cached result, return None
        return None

    def write(self, fithash, result):
        with self.lock_context():
            logger.debug('Reading, then writing')
            # First update, to check if this hash has been written since the fit started
            self.read()
            # If not written in the meantime, add it now
            if fithash not in self.cache:
                self.cache[fithash] = result
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)            
