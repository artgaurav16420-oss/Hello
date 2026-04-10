"""
Windows-first OSQP import guard for Python 3.12+ processes.

On Windows, OSQP and scientific stacks such as NumPy/SciPy/Pandas can load
different BLAS/LAPACK runtimes when imported in an unsafe order. Under Python
3.12+ this can manifest as process-level access violations during extension
module initialization. Importing OSQP first pins its native dependency chain
before NumPy/Pandas-heavy modules initialize.

This module must be the very first import in every entrypoint and test module
that can transitively import numeric libraries.
"""
import sys
import logging

# Set up a module-level logger to track initialization
logger = logging.getLogger("osqp_preimport")

if sys.platform == "win32":
    try:
        import osqp as _osqp
        # No-op to ensure the import is not optimized away by linters
        _ = _osqp.__name__
    except Exception as e:
        # Failure here is fatal for the engine's numeric stability on Windows
        logger.critical("OSQP pre-import failed on Windows: %s", e)
        raise
