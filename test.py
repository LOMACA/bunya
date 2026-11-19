import numpy as np
a = np.load(r'C:\Users\uqlcapp1\Simulations\Bunya\output\sweep_test_out.npz')['S']
b = np.stack([np.load(rf'C:\Users\uqlcapp1\Simulations\Python\results\sweep_test\kx_{i:03d}.npz')['S'] for i in range(2)])
print(a.shape == b.shape, np.allclose(a, b, rtol=1e-6, equal_nan=True))
print(np.nanmax(np.abs(a / b - 1)))