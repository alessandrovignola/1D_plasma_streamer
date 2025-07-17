import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('default')
matplotlib.use('TkAgg')

# Sostituisci con il nome del tuo file di output
filename = r"C:\Users\aless\ProgettiFortran\semi_implicit_fluid-master\output\result_001001.txt"

# Carica i dati ignorando la prima riga (header)
data = np.loadtxt(filename, skiprows=1)

# Estrai colonne: x e potenziale (colonna 0 e colonna 4)
x = data[:, 0]
potential = data[:, 4]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, potential, linestyle='-')
plt.xlabel('Coordinata x [m]')
plt.ylabel('Potenziale')
#plt.ylim(-2e6, 3e6)

plt.title('Potenziale rispetto alla coordinata x')
plt.grid(True)
plt.show()
