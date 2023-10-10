"""
Reglas de Mamdani
R1
- Si NOTA es BAJA entonces DESAPROBADO

R2
- SR1 Si CONCEPTO es REGULAR y NOTA es MEDIA entonces HABILITADO <-- min|__ max
- SR2 Si CONCEPTO es REGULAR y NOTA es ALTA entonces HABILITADO <--min  |

R3
- SR3 Si CONCEPTO es BUENO y NOTA es ALTA entonces PROMOCION   <-- min   | 
- SR4 Si CONCEPTO es EXCELENTE y NOTA es ALTA entonces PROMOCION <-- min |---> max
- SR5 Si CONCEPTO es EXCELENTE y NOTA es MEDIA entonces PROMOCION <-- min| 

"""

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

concept = np.arange(0, 10.2, 0.2)
numeric = np.arange(0, 101, 1)
total = np.arange(0, 101, 1)

# FUZZIFICATION

# Generate fuzzy membership function
conceptReg = fuzz.gaussmf(concept, 0, 4)
conceptBueno = fuzz.gaussmf(concept, 8, 1.5)
conceptExc = fuzz.gaussmf(concept, 10, 1.5)

numericBajo = fuzz.trimf(numeric, [0, 0, 50])
numericMed = fuzz.trimf(numeric, [30, 55, 80])
numericAlto = fuzz.trimf(numeric, [60, 100, 100])

totalMin = fuzz.trimf(total, [0, 0, 40])
totalMed = fuzz.trimf(total, [30, 50, 70])
totalMax = fuzz.trimf(total, [60, 100, 100])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(concept, conceptReg, 'b', linewidth=1.5, label='Regular')
ax0.plot(concept, conceptBueno, 'g', linewidth=1.5, label='Bueno')
ax0.plot(concept, conceptExc, 'r', linewidth=1.5, label='Excelente')
ax0.set_title('Concepto')
ax0.legend()

ax1.plot(numeric, numericBajo, 'b', linewidth=1.5, label='Bajo')
ax1.plot(numeric, numericMed, 'g', linewidth=1.5, label='Medio')
ax1.plot(numeric, numericAlto, 'r', linewidth=1.5, label='Alto')
ax1.set_title('Nota numerica')
ax1.legend()

ax2.plot(total, totalMin, 'b', linewidth=1.5, label='Desaprobacion')
ax2.plot(total, totalMed, 'g', linewidth=1.5, label='Habilitacion')
ax2.plot(total, totalMax, 'r', linewidth=1.5, label='Promocion')
ax2.set_title('Nota Final')
ax2.legend()

# INFERENCE

notaNumerica = 70
notaConcepto = 3 

concept_level_reg = fuzz.interp_membership(concept, conceptReg, notaConcepto)
concept_level_bueno = fuzz.interp_membership(concept, conceptBueno, notaConcepto)
concept_level_exc = fuzz.interp_membership(concept, conceptExc, notaConcepto)

numeric_level_bajo = fuzz.interp_membership(numeric, numericBajo, notaNumerica)
numeric_level_med = fuzz.interp_membership(numeric, numericMed, notaNumerica)
numeric_level_alto = fuzz.interp_membership(numeric, numericAlto, notaNumerica)

print(f'CONCEPTO REG {concept_level_reg} \n CONCEPTO BUENO {concept_level_bueno} \n CONCEPTO EXC {concept_level_exc} ')
print(f'NOTA BAJA {numeric_level_bajo} \n NOTA MEDIA {numeric_level_med} \n NOTA ALTA{numeric_level_alto} ')

# CONCEPTO REG 0.7548396019890073 
# CONCEPTO BUENO 0.0038659201394728076
# CONCEPTO EXC 1.866446911352057e-05
# NOTA BAJA 0.0
# NOTA MEDIA 0.4
# NOTA ALTA 0.25


# Rule 1 -DESAPROBAR
nota_final_des = np.fmin(numeric_level_bajo, totalMin)

# Rule 2 -HABILITAR
subrule1 = (concept_level_reg * 0.1 + numeric_level_med * 0.9)
subrule2 = (concept_level_reg* 0.1 + numeric_level_alto * 0.9)
# subrule1 or subrule2 --> fmax
active_rule2 = np.fmax(subrule1, subrule2) 

nota_final_hab = np.fmin(active_rule2, totalMed) 


# # Rule 3 - PROMOCIONAR 
subrule3 = (concept_level_exc * 0.1 + numeric_level_med * 0.9) 
active_rule3 = np.fmax(subrule3, numeric_level_alto)

# TRUNCA el grafico de nota final max
nota_final_promo = np.fmin(active_rule3, totalMax)

# pone un piso para rellenar la funcion truncada
nota0 = np.zeros_like(total) 

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(total, nota0, nota_final_hab, facecolor='b', alpha=0.7)
ax0.plot(total, totalMin, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(total, nota0, nota_final_des, facecolor='g', alpha=0.7)
ax0.plot(total, totalMed, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(total, nota0, nota_final_promo, facecolor='r', alpha=0.7)
ax0.plot(total, totalMax, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('INFERENCIA')
plt.tight_layout()

# AGREGATION
aggregated = np.fmax(nota_final_hab,
                     np.fmax(nota_final_des, nota_final_promo))

# DEFUZZIFICATION
nota = fuzz.defuzz(total, aggregated, 'centroid')
print(f'NOTA = {nota}')
nota_activation = fuzz.interp_membership(total, aggregated, nota)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(total, totalMin, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(total, totalMed, 'g', linewidth=0.5, linestyle='--')
ax0.plot(total, totalMax, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(total, nota0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([nota, nota], [0, nota_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('AGREGACION')

# Generar la función de membresía trapezoidal
x = np.arange(0, 5.05, 0.1)
mfx = fuzz.trapmf(x, [2, 2.5, 3, 4.5])

# Calcular el centroide (defuzzificación) - Método del centroide
defuzz_centroid = fuzz.defuzz(x, mfx, 'centroid')

# Mostrar la función de membresía trapezoidal y el centroide
plt.figure(figsize=(8, 5))
plt.plot(x, mfx, 'k')
plt.vlines(defuzz_centroid, 0, 1, label='centroid', color='r')
plt.ylabel('Pertenencia difusa')
plt.xlabel('Variable en el universo (arb)')
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()
