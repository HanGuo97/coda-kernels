from pathlib import Path

import quack.cache

_CODA_CORE = Path(__file__).resolve().parent
if _CODA_CORE not in quack.cache.EXTRA_SOURCE_DIRS:
    quack.cache.EXTRA_SOURCE_DIRS.append(_CODA_CORE)
