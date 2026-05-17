import time
import triton
from triton import runtime
from triton.testing import _summarize_statistics


def do_bench_count(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median", "all"]

    di = runtime.driver.active.get_device_interface()

    fn()
    di.synchronize()

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    # interpret ``warmup`` and ``rep`` as iteration counts directly
    n_warmup = warmup
    n_repeat = rep
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        runtime.driver.active.clear_cache(cache)
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


def do_bench_dict(fn_dict: dict, warmup: int, repeats: int) -> dict:
    results = {}
    for name, fn in fn_dict.items():
        if fn is None:
            results[f"{name}/time"] = None
            results[f"{name}/count"] = None
            continue
        time.sleep(0.5)
        results[f"{name}/time"] = triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=repeats,
        )
        time.sleep(0.5)
        results[f"{name}/count"] = do_bench_count(
            fn,
            warmup=warmup,
            rep=repeats,
        )

    time.sleep(0.5)
    return results
