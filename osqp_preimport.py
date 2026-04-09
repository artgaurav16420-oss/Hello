"""
osqp_preimport.py — Windows DLL Load Order Patch
================================================
Early import of OSQP to ensure it owns the math ABI before other 
heavy numeric libraries like Scipy/Pandas are initialized.

RATIONALE:
On Windows with Python 3.12+, standard libraries (like Scipy) and OSQP 
may link to different math backends (e.g., MKL vs. OpenBLAS). If 
initialization order is not strictly controlled, C-extension loading 
can trigger process-level Access Violations (0xC0000005). Importing 
OSQP first ensures its binary dependencies are prioritized in the 
process address space.

MANDATORY: 
This module must be imported before any other numeric library in the 
entry point of the application.
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
