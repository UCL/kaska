#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lifted from:
 https://gist.github.com/Jwink3101/494e741f07d33edea47d369bcfc4a54a"""

# __version__ = '20180726.0'
# __status__ = 'beta'

import multiprocessing as mp
import multiprocessing.dummy as mpd
from threading import Thread
import sys

try:
    import tqdm
except ImportError:
    tqdm = None

from functools import partial

if sys.version_info[0] > 2:
    unicode = str
    xrange = range
    imap = map
else:
    from itertools import imap

CPU_COUNT = mp.cpu_count()


def parmap(fun, seq, num_proc=None, num_threads=1, chunksize=1, ordered=True,
           daemon=False, progress=False):
    """
    parmap -- Simple parallel mapper that can split amongst processes
              (num_proc) and threads (num_threads) (within the processes).

              Does *NOT* require functions to be pickleable (unlike
              vanilla multiprocess.Pool.map)

    Inputs:
    -------
    fun
        Single input function. Use lambdas or functools.partial
        to enable/exapnd multi-input. See example

    sequence
        Sequence of inputs to map in parallel

    Options:
    --------
    num_proc [None] (integer or None)
        Number of processes to use. If `None`, will use the CPU_COUNT

    num_threads [1] (integer)
        Number of threads to use. See notes below on multi-threaded vs
        multi-processes.

    chunksize [1] (int)
        How to be break up the incoming sequence. Useful if also using threads.
        Will be (re)set to max(chunksize, num_threads)

    ordered [True] (bool)
        Whether or not to order the results. If False, will return in whatever
        order they finished.

    daemon [False] (bool)
        Sets the multiprocessing `daemon` flag. If  True, can not spawn child
        processes (i.e. cannot nest parmap) but should allow for CTRL+C type
        stopping. Supposedly, there may be issues with CTRL+C with it set to
        False. Use at your own risk

    progress [False] (bool)
        Display a progress bar or counter.
        Warning: Inconsistant in iPython/Jupyter notebooks and may clear
        other printed content. Instead, specify as 'nb' to use a Jupyter
        Widget progress

    Notes:
    ------

    Performs SEMI-lazy iteration based on chunksize. It will exhaust the input
    iterator but will yield as results are computed (This is similar to the
    `multiprocessing.Pool().imap` behavior)

    Explicitly wrap the parmap call in a list(...) to force immediate
    evaluation

    Threads and/or processes:
    -------------------------
    This tool has the ability to split work amongst python processes
    (via multiprocessing) and python threads (via the multiprocessing.dummy
    module). Python is not very performant in multi-threaded situations
    (due to the GIL) therefore, processes are the usually the best for CPU
    bound tasks and threading is good for those that release the GIL (such
    as IO-bound tasks). Note that many NumPy functions *do* release the GIL
    and can be threaded, but many NumPy functions are, themselves, multi-
    threaded.

    Alternatives:
    -------------

    This tool allows more data types, can split with threads, has an optional
    progress bar, and has fewer pickling issues, but these come at a small
    cost.
    For simple needs, the following may be better:

    >>> import multiprocessing as mp
    >>> pool = mp.Pool(num_proc) # Or mp.Pool() for num_proc=None
    >>> results = list( pool.imap(fun,seq) ) # or just pool.map
    >>> pool.close()

    Process Method:
    ---------------
    This code uses iterators/generators to handle and distribute the workload.
    By doing this, it is easy to have all results pass through a common
    counting function for display of the progress without the use of
    global (multiprocessing manager) variables and locks.

    With the exception of when num_proc == 1 (where it falls back to serial
    methods) the code works as follows:

    - A background thread is started the will iterate over the incoming
      sequence and add items to the queue. If the incoming sequence is
      exhausted, the worker sends kill signals into the queue.
        - The items are also chunked and indexed (used later)
    - After the background thread is started a function to pull from the OUTPUT
      queue is created. This counts the number of closed processes but
      otherwise yields the computed result items
    - A pool of workers is created. Each worker will read from the input queue
      and distribute the work amongst threads (if using). It will then
      return the resuts into a queue
    - Now the main work happens. It is done as chain of generators/iterators.
      The background worker has already begin adding items to the queue so
      now we work through the output queue. Note that this is in serial
      since the work was already done in parallel
            - Generator to pull from the result queue
            - Generator to count and display progress (if progress=True).
            - Generator to hold on to and return items in a sorted manner
              if sorting is requested. This can cause itermediate results to be
              stored until they can be returned in order
    - The output generator chain is iterated pulling items through and then
      are yielded.
    - cleanup.

    Last Updated:
    -------------
    2018-07-30
    """
    if num_proc is None:
        num_proc = CPU_COUNT

    chunksize = max(chunksize, num_threads)

    try:
        tot = len(seq)
    except TypeError:
        tot = None

    if tqdm is None:
        if isinstance(progress, (str, unicode))\
         and progress.lower() in ['jupyter', 'notebook', 'nb']:
            counter = partial(_counter_nb, tot=tot)
        else:
            counter = partial(_counter, tot=tot)
    else:
        if isinstance(progress, (str, unicode)) \
         and progress.lower() in ['jupyter', 'notebook', 'nb'] \
         and hasattr(tqdm, 'tqdm_notebook'):
            counter = partial(tqdm.tqdm_notebook, total=tot)
        else:
            # Set the total since tqdm won't be able to get it.
            counter = partial(tqdm.tqdm, total=tot)

    if num_proc == 1:
        if num_threads == 1:
            out = imap(fun, seq)
        else:
            # Thread pools don't have the pickle issues
            pool = mpd.Pool(num_threads)
            out = pool.imap(fun, seq)

        if progress:
            out = counter(out)
        for item in out:
            yield item

        if num_threads > 1:
            pool.close()
        return

    q_in = mp.JoinableQueue()
    q_out = mp.Queue()

    # Start the workers
    workers = [mp.Process(target=_worker, args=(fun, q_in, q_out, num_threads))
               for _ in xrange(num_proc)]

    for worker in workers:
        worker.daemon = daemon
        worker.start()

    # Create a separate thread to add to the queue in the background
    def add_to_queue():
        for iixs in _iter_chunks(enumerate(seq), chunksize):
            q_in.put(iixs)

        # Once (if ever) it is exhausted, send None to close workers
        for _ in xrange(num_proc):
            q_in.put(None)

    add_to_queue_thread = Thread(target=add_to_queue)
    add_to_queue_thread.start()

    # Generator we use to return
    def queue_getter():
        finished = 0
        while finished < num_proc:
            out = q_out.get()
            if out is None:
                finished += 1
                continue
            yield out

    # Chain generators on output
    out = queue_getter()
    if progress:
        out = counter(out)

    if ordered:
        out = _sort_generator_unique_integers(out, key=lambda a: a[0])

    # Return
    for item in out:
        yield item[1]

    # Clean up
    q_in.join()
    for worker in workers:
        worker.join()


def _counter(items, tot=None):
    for i, item in enumerate(items):
        if tot is not None:
            _txtbar(i, tot, ticks=50, text='')
        else:
            txt = '{}'.format(my_ii + 1)
            print('\r%s' % txt, end='')
            sys.stdout.flush()
        yield item


def _counter_nb(items, tot=None):
    from ipywidgets import FloatProgress, FloatText
    from IPython.display import display

    if tot is not None:
        my_text = FloatProgress(min=0, max=tot)
    else:
        my_text = FloatText()
        my_text.value = 0
    display(my_text)

    for my_ii, item in enumerate(items):
        my_text.value += 1
        yield item


def _worker(fun, q_in, q_out, num_threads):
    """ This actually runs everything """
    if num_threads > 1:
        pool = mpd.Pool(num_threads)
        _map = pool.map  # thread pools don't have the pickle issues
    else:
        _map = map

    while True:
        iixs = q_in.get()
        if iixs is None:
            q_out.put(None)
            q_in.task_done()
            break

#         for ix in iixs:
        def _ap(i_x):
            i, my_x = i_x
            q_out.put((i, fun(my_x)))
        list(_map(_ap, iixs))  # list forces the iteration
        q_in.task_done()

    if num_threads > 1:
        pool.close()


def _iter_chunks(seq, size):
    """
    Yield a len(size) tuple from seq. If not divisible, the last one
    would be less than size
    """
    _n = 0
    for item in seq:
        if _n == 0:
            group = [item]
        else:
            group.append(item)
        _n += 1

        if _n == size:
            yield tuple(group)
            _n = 0
    if _n > 0:
        yield tuple(group)


def _sort_generator_unique_integers(items, start=0, key=None):
    """
    Yield from `items` in order assuming UNIQUE keys w/o any missing!

    The items ( or key(item) ) MUST be an integer, without repeats, starting
    at `start`
    """
    queue = dict()
    for item in items:
        if key is not None:
            i_k = key(item)
        else:
            i_k = item

        if i_k == start:
            yield item
            start += 1

            # Get any stored items
            while start in queue:
                yield queue.pop(start)  # average O(1), worse-case O(N)
                start += 1              # but based on ref below,
        else:                           # should be O(1) for integer keys.
            queue[i_k] = item
            # Ref: https://wiki.python.org/moin/TimeComplexity

    # Exhaust the rest
    while start in queue:
        yield queue.pop(start)
        start += 1


def _txtbar(count, iter_size, ticks=50, text='Progress'):
    """
    Print a text-based progress bar.

    Usage:
        _txtbar(count, iter_size)

    Inputs:
        count     : Iteration count (start at 0)
        iter_size : Iteration size
        ticks     : [50] Number of ticks
        text      : ['Progress'] Text to display (don't include `:`)

    Prints a text-based progress bar to the terminal. Obviously
    printing other things to screen will mess this up.
    """

    count = int(count + 1)
    ticks = min(ticks, iter_size)
    is_count = int(1.0*count % round(1.0*iter_size/ticks)) == 0

    if not (is_count or count == 1 or count == iter_size):
        return

    num_pound = int(round(1.0 * count/iter_size*ticks))
    num_space = int(1.0 * ticks - num_pound)
    num_print = int(round(1.0 * count/iter_size*100))

    if count == 1:
        num_print = 0

    if len(text) > 0:
        text += ': '

    txt = '{:s}{:s}{:s} : {:3d}%  '.format(text, '#'*num_pound,
                                           '-'*num_space, num_print)
    print('\r%s' % txt, end='')
    sys.stdout.flush()
