"""Analytic tactile geometry, light distribution, and CPU/CUDA renderers.

The numerical model is migrated from tactile_sensor_sim at upstream commit
865db6c656548687f2439cafb0dcea76ab01f9ad. Imports remain lazy so NumPy CPU
rendering does not require PyTorch or a CUDA runtime.
"""
