Quickstart
==========

Minimal example
---------------

.. code-block:: python

   import numpy as np
   from mice import MICE
   from mice.policy import DropRestartClipPolicy

   def gradient(x, thetas):
       return x - thetas

   def sampler(n):
       return np.random.randn(n, 1)

   estimator = MICE(
       grad=gradient,
       sampler=sampler,
       eps=0.577,
       min_batch=10,
       policy=DropRestartClipPolicy(
           drop_param=0.5,
           restart_param=0.0,
           max_hierarchy_size=100,
       ),
       max_cost=10_000,
       stop_crit_norm=1e-6,
   )

   x = np.array([10.0], dtype=float)
   for _ in range(100):
       g = estimator(x)
       x = x - 0.1 * g

Notes
-----

- ``MICE`` is imported from ``mice`` (top-level re-export).
- The policy controls index-set operations (Add/Drop/Restart/Clip behavior).
- ``max_cost`` bounds total gradient evaluations.

