import torch
import quack.cache
from pathlib import Path

_CODA_CORE = Path(__file__).resolve().parent
if _CODA_CORE not in quack.cache.EXTRA_SOURCE_DIRS:
    quack.cache.EXTRA_SOURCE_DIRS.append(_CODA_CORE)

torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 128)
