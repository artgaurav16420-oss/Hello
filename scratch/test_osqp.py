import sys
import numpy as np
try:
    import osqp
    print(f"OSQP Version: {osqp.__version__}")
    print(f"NumPy Version: {np.__version__}")
    # Try a simple problem to trigger actual library calls
    import scipy.sparse as spa
    P = spa.csc_matrix([[4., 1.], [1., 2.]])
    q = np.array([1., 1.])
    A = spa.csc_matrix([[1., 1.], [1., 0.], [0., 1.]])
    l = np.array([1., 0., 0.])
    u = np.array([1., 0.7, 0.7])
    prob = osqp.OSQP()
    prob.setup(P, q, A, l, u, alpha=1.0)
    res = prob.solve()
    print("OSQP execution successful")
except Exception as e:
    print(f"Error: {e}")
except SystemError as e:
    print(f"System Error: {e}")
