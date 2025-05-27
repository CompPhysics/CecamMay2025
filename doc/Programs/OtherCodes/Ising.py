import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 50                                 # linear lattice size
temperatures = np.arange(2.0, 2.51, 0.1)  # T = 2.0, 2.1, ..., 2.5
n_steps = 100000                       # Monte Carlo steps per temperature

for T in temperatures:
    beta = 1.0 / T
    J = 1.0   # coupling strength
    # Initialize spins randomly to +1 or -1
    spins = np.random.choice([-1, 1], size=(L, L))
    
    # Metropolis update loop
    for step in range(n_steps):
        # Pick a random spin (i,j)
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        # Compute sum of the four neighbors (with periodic boundary conditions)
        total_neighbors = (
            spins[(i+1) % L, j] + spins[(i-1) % L, j] +
            spins[i, (j+1) % L] + spins[i, (j-1) % L]
        )
        # Energy change for flipping this spin
        dE = 2 * J * spins[i, j] * total_neighbors
        # Metropolis criterion: accept flip if dE <= 0 or with prob. exp(-dE*beta)
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] *= -1

    # Plot the final spin configuration
    plt.figure(figsize=(4, 4))
    # Use a grayscale colormap: -1 will map to black, +1 to white (vmin/vmax)
    plt.imshow(spins, cmap='gray', vmin=-1, vmax=1)
    plt.title(f'Ising Lattice (T = {T:.1f})')
    plt.axis('off')
    plt.savefig(f'ising_T{T:.1f}.png', dpi=150)
    plt.close()

"""    
Each saved image (e.g. ising_T2.0.png) visually depicts the spin
arrangement at that temperature. The code is easy to extend: changing
L, the temperatures list, or n_steps allows different lattice sizes or
more/fewer Monte Carlo sweeps. As expected, low-T images become mostly
uniform (a single color, since nearly all spins align), while high-T
images appear random (resembling “TV static”) . Around the critical
temperature (~2.27), one sees a mixture of black and white clusters
with fractal boundaries (as in the figure above) .
"""
