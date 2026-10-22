"""Data side of the reconstruction: raw I/O, coils, navigator, binning, operators.

Submodules are imported explicitly (``from dataloader.operators import ...``)
so that light-weight pieces such as ``binning`` do not pull in sigpy/cupy.
"""
