import scipy.optimize as opt
import numpy as np

# Définition des constantes
gamma = 1.4  # Rapport des capacités thermiques (ex : air)
Ae_At = 1.5  # Exemple de rapport d'aire (à adapter)

# Équation à résoudre pour Me
def mach_equation(Me):
    return (1/Me) * ((2/(gamma+1)) * (1 + (gamma-1)/2 * Me**2))**((gamma+1)/(2*(gamma-1))) - Ae_At

# Trouver la solution pour Me > 1 (écoulement supersonique)
Me_solution = opt.fsolve(mach_equation, x0=0.0)  # x0=2.0 est une estimation initiale

print(Me_solution)