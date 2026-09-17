"""Import `relay` before anything else can pull in numpy.

`relay/__init__` caps the BLAS thread pools, and that only helps if it runs before OpenBLAS
loads. A test module that imported numpy at the top before importing relay would allocate
20 threads' worth of scratch buffers and abort on a busy machine.
"""

import relay  # noqa: F401  (imported for the side effect, deliberately)
