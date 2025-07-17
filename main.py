import time
from concurrent.futures import ProcessPoolExecutor as Executor

from solvers.current_lim import current_lim
from solvers.rungekutta import exact
from solvers.semi_implicit import semi_implicit
from plotting import plot_all


def run_method(name):
    start = time.time()
    if name == "RK4":
        E, n = exact()
        ls, cl = '--','k'
    elif name == "Current-lim":
        E, n = current_lim()
        ls, cl = '-', 'g'
    elif name == "Semi-implicit":
        E, n = semi_implicit()
        ls, cl = '-', 'purple'
    else:
        raise ValueError(f"No corresponding method: {name}")
    end = time.time()
    print(f"{name} finished in {end - start:.3f} s")
    return name, E, n, ls, cl

# Main code
if __name__ == "__main__":
    method_names = ["Current-lim","RK4","Semi-implicit"]

    results = {}
    with Executor() as executor:
        futures = [executor.submit(run_method, name) for name in method_names]
        for future in futures:
            name, E, n, ls, cl = future.result()
            results[name] = {"E": E, "n": n, "linestyle":ls, "linecolor":cl}

    plot_all(results, save=False)
