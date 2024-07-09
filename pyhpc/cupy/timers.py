import cupy as cp

from contextlib import contextmanager
from timeit import default_timer

@contextmanager
def cupy_timer():
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    yield
    end.record()
    end.synchronize()
    elapsed_time = cp.cuda.get_elapsed_time(start, end)
    print(f'Elapsed time: {elapsed_time} ms')
    

@contextmanager
def cpu_timer():
    start = default_timer()
    yield
    end = default_timer()
    print(f'Elapsed time: {(end - start) * 1000} ms')