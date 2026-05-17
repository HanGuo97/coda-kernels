import torch
import triton
import argparse
from dataclasses import dataclass
from typing import Callable
from models import (
    gpt,
    gpt_ref,
    utils,
)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 32
    n_kv_head: int = 32
    n_embd: int = 8192


def prepare_model(
    config: GPTConfig,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    model_ref = gpt_ref.GPT(config)
    model_ref = model_ref.to(dtype=gpt.DEFAULT_DTYPE, device="cuda")
    model_ref.init_weights()
    # model_ref.eval()
    model_ref = torch.compile(model_ref)

    model = gpt.GPT(config)
    model = model.to(dtype=gpt.DEFAULT_DTYPE, device="cuda")
    # model.eval()

    utils.convert_weights(model, model_ref)
    return model_ref, model


def prepare_data(
    config: GPTConfig,
    batch_size: int,
) -> tuple[tuple, dict]:
    indices = torch.randint(0, config.vocab_size, (batch_size, config.sequence_len), device="cuda")
    targets = torch.randint(0, config.vocab_size, (batch_size, config.sequence_len), device="cuda")
    model_args = (indices, targets)
    model_kwargs = {}
    return model_args, model_kwargs


def prepare_bench_fn(
    model: torch.nn.Module,
    model_args: tuple,
    model_kwargs: dict,
    no_grad: bool,
) -> Callable:
    if not no_grad:
        bench_fn = lambda: model(*model_args, **model_kwargs)
    else:
        @torch.no_grad()
        def bench_fn() -> object:
            return model(*model_args, **model_kwargs)

    return bench_fn


def profile(
    bench_fn: Callable,
    num_warmup: int = 2,
    num_profile: int = 1,
) -> None:

    for _ in range(num_warmup):
        bench_fn()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_profile):
        bench_fn()
    torch.cuda.cudart().cudaProfilerStop()


def benchmark(
    batch_size: int = 4,
    ncu: bool = False,
) -> tuple[float | None, float | None]:
    model_ref, model = prepare_model(
        config=GPTConfig,
    )
    model_args, model_kwargs = prepare_data(
        config=GPTConfig,
        batch_size=batch_size,
    )
    bench_fn_ref = prepare_bench_fn(
        model=model_ref,
        model_args=model_args,
        model_kwargs=model_kwargs,
        no_grad=True,
    )
    bench_fn = prepare_bench_fn(
        model=model,
        model_args=model_args,
        model_kwargs=model_kwargs,
        no_grad=True,
    )

    if not ncu:
        results_ref = triton.testing.do_bench(bench_fn_ref, warmup=2, rep=3)
        results = triton.testing.do_bench(bench_fn, warmup=2, rep=3)
        return results_ref, results
    else:
        raise NotImplementedError


if __name__ == "__main__":
    benchmark(ncu=True)
