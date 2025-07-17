import matplotlib
import matplotlib.pyplot as plt
plt.style.use('default')
matplotlib.use('TkAgg')
import os
from shared import x

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,

})


def plot_all(results, save=False, output_dir="plots"):
    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # E field
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        plt.plot(x, data["E"], label=f'Field - {label}',
                 linestyle=data.get("linestyle", "-"),
                 color=data.get("linecolor", "k"))
    plt.xlabel('x (m)')
    plt.ylabel('E (V/m)')
    plt.title('Electric Field Comparison')
    plt.ylim(-2e6, 3e6)
    plt.xlim(0.001, 0.007)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"{output_dir}/electric_field_comparison.png", dpi=300)

    # Electron density
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        plt.plot(x, data["n"], label=f'Density - {label}',
                 linestyle=data.get("linestyle", "-"),
                 color=data.get("linecolor", "k"))
    plt.xlabel('x (m)')
    plt.ylabel(r'$n_{e}\ (\mathrm{m}^{-3})$')
    plt.title('Electron Density Comparison')
    plt.yscale('log')
    plt.ylim(1e12, 1e21)
    plt.xlim(0.001, 0.007)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(f"{output_dir}/electron_density_comparison.png", dpi=300)

    plt.show()

