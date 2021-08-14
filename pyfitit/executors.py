#  author "Nathaniel Williams"
#  license "Apache 2.0"

import threading, traceback, logging, sys
from queue import Queue


QUEUE_GET_TIMEOUT = 30 * 24 * 60 * 60

THREAD_DONE = object()


class ThreadException:
    def __init__(self, exception_info):
        self.exception_info = exception_info


class LazyThreadPoolExecutor(object):
    def __init__(self, num_workers=1):
        self.num_workers = num_workers
        self.result_queue = Queue()
        self.thread_sem = threading.Semaphore(num_workers)
        self._shutdown = threading.Event()
        self.threads = []
        self.iterable = None

    def map(self, predicate, iterable):
        self._shutdown.clear()
        self.iterable = ThreadSafeIterator(iterable)
        self._start_threads(predicate)
        return self._result_iterator()

    def shutdown(self, wait=True):
        self._shutdown.set()
        if wait:
            for t in self.threads:
                t.join()

    def _start_threads(self, predicate):
        for i in range(self.num_workers):
            t = threading.Thread(
                name="LazyChild #{0}".format(i),
                target=self._make_worker(predicate)
            )
            t.daemon = True
            self.threads.append(t)
            t.start()

    def _make_worker(self, predicate):
        def _w():
            try:
                with self.thread_sem:
                    for thing in self.iterable:
                        self.result_queue.put(predicate(thing))
                        if self._shutdown.is_set():
                            break
                self.result_queue.put(THREAD_DONE)
            except Exception:
                self.result_queue.put(ThreadException(sys.exc_info()))
        return _w

    def _result_iterator(self):
        while 1:
            # Queue.get is not interruptable w/ ^C unless you specify a
            # timeout.
            # Hopefully one year is long enough...
            # See http://bugs.python.org/issue1360
            result = self.result_queue.get(True, QUEUE_GET_TIMEOUT)
            if isinstance(result, ThreadException):
                print('Error in lazy execution thread', file=sys.stderr)
                traceback.print_exception(*result.exception_info)
            elif result is not THREAD_DONE:
                yield result
            else:
                # if all threads have exited
                # sorry, this is kind of a gross way to use semaphores
                if self.thread_sem._value == self.num_workers:
                    break
                else:
                    continue


class ThreadSafeIterator(object):
    def __init__(self, it):
        self._it = iter(it)
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        with self.lock:
            return self._it.__next__()
