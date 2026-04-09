"""
osqp_preimport.py — Windows DLL Load Order Patch
================================================
Early import of OSQP to ensure it owns the math ABI before other 
heavy numeric libraries like Scipy/Pandas are initialized.
"""
import sys
import logging

# Set up a minimal logger to track if initialization happened
logger = logging.getLogger("osqp_preimport")

try:
    # On Windows/Python 3.12+, OSQP must be imported before standard libraries 
    # that link to varied math backends (MKL, OpenBLAS) to prevent 
    # process-level Access Violations during C-extension initialization.
    import osqp as _osqp
    # No-op to ensure it's not optimized away
    _ = _osqp.__name__
except Exception as e:
    # If it fails here, the process is likely unrecoverable for the engine
    print(f"CRITICAL: OSQP Pre-import failed on Windows: {e}", file=sys.stderr)
