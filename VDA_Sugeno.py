import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import time

from scipy.spatial import distance_matrix

# -------------------------------------- Agregado ------------------------------------------------------


def cargar_datos(path):
    datos = []

    # Abre el archivo y lee los datos
    with open(path, "r") as file:
        for linea in file:
            # Convierte los valores de cadena a números de punto flotante
            valory = float(linea.strip())
            datos.append(valory)

        # Convierte las listas en arreglos NumPy
        datos = np.array(datos)
    return datos


def genero_datosX(cantidad):
    datos_x = []
    valorx = 0
    for i in range(cantidad):
        datos_x.append(valorx)
        valorx = valorx+2.5
    return datos_x


def dividir_datos(datos, porcentaje):
    datos_test = []
    cant_datos_test = int(len(datos)*porcentaje)

    # Selecciona datos de test al azar
    data_frame = pandas.DataFrame(datos)
    filas_aleatorias = data_frame.sample(n=cant_datos_test)
    datos_test = filas_aleatorias.values

    datos_training = []
    [datos_training.append(x) for x in datos if x not in datos_test]
    return np.array(datos_training), np.array(datos_test)

# -------------------------------------- Substractive Clustering --------------------------------------------------------


def subclust2(data, Ra, Rb=0, AcceptRatio=0.3, RejectRatio=0.1):
    # Si Rb no se proporciona, se establece como un 15% más grande que Ra
    if Rb == 0:
        Rb = Ra * 1.15

    # Escalar los datos al rango [0, 1]
    scaler = MinMaxScaler()
    scaler.fit(data)
    ndata = scaler.transform(data)

    # Calcular la matriz de distancias entre los datos y luego calcular el potencial
    P = distance_matrix(ndata, ndata)
    alpha = (Ra / 2) ** 2
    P = np.sum(np.exp(-P ** 2 / alpha), axis=0)

    # Inicializar la lista de centros con el punto de mayor potencial
    centers = []
    i = np.argmax(P)
    C = ndata[i]
    p = P[i]
    centers = [C]

    # Inicializar banderas para el proceso iterativo
    continuar = True
    restarP = True

    # Iniciar el proceso iterativo para seleccionar centros de manera dinámica
    while continuar:
        pAnt = p
        if restarP:
            # Restar el potencial del centro seleccionado de la matriz de potenciales
            P = P - p * \
                np.array([np.exp(-np.linalg.norm(v - C) ** 2 / (Rb / 2) ** 2)
                         for v in ndata])
        restarP = True

        # Seleccionar el punto con el mayor potencial como posible centro
        i = np.argmax(P)
        C = ndata[i]
        p = P[i]

        # Comprobar si el potencial es mayor que el umbral de aceptación
        if p > AcceptRatio * pAnt:
            centers = np.vstack((centers, C))
        # Comprobar si el potencial es menor que el umbral de rechazo
        elif p < RejectRatio * pAnt:
            continuar = False
        else:
            # Comprobar si el punto cumple con la condición de distancia relativa a los centros existentes
            dr = np.min([np.linalg.norm(v - C) for v in centers])
            if dr / Ra + p / pAnt >= 1:
                centers = np.vstack((centers, C))
            else:
                P[i] = 0
                restarP = False

        # Comprobar si no quedan puntos con potencial positivo
        if not any(v > 0 for v in P):
            continuar = False

    # Calcular las distancias entre los puntos y los centros para asignar etiquetas de cluster
    distancias = [[np.linalg.norm(p - c) for p in ndata] for c in centers]
    labels = np.argmin(distancias, axis=0)

    # Revertir la escala para obtener los centros en la escala original de los datos
    centers = scaler.inverse_transform(centers)

    # Devolver etiquetas de cluster y centros
    return labels, centers

# ------------------------------------------ SUGENO -------------------------------------------------------


def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean)**2.) / (2 * sigma**2.))


class fisInput:
    def __init__(self, min, max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids

    # Muestra las gaussianas
    def view(self, ax):
        x = np.linspace(self.minValue, self.maxValue, 20)
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x, m, s)
            ax.plot(x, y)


class fis:
    def __init__(self):
        self.rules = []
        self.memberfunc = []
        self.inputs = []

    def genfis(self, data, radii):

        labels, cluster_center = subclust2(data, radii)

        # ------------------------------- Grafico con los clusters --------------------------------

        # Crear una figura y un objeto de ejes para el gráfico
        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(10,5))

        # Graficar los datos
        ax1.scatter(data[:, 0], data[:, 1], c=labels,
                   cmap='viridis', label='Datos')

        # Graficar los centros de los clusters
        ax1.scatter(cluster_center[:, 0], cluster_center[:, 1],
                   c='red', marker='x', s=100, label='Centros de Clusters')

        ax1.set_xlabel('Tiempo [ms]')
        ax1.set_ylabel('VDA')
        ax1.set_title('Gráfico de Datos y Centros de Clusters')
        ax1.legend()
        ax2.set_title('Funciones de pertenencia')
        ax2.set_xlabel('Tiempo [ms]')
        ax2.set_ylabel('Pertenencia')
        ax2.legend()

        # ----------------------------------- termino de graficar ----------------------------------------

        cluster_center = cluster_center[:, :-1]
        P = data[:, :-1]  # Saco los targets (creo)
        # T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(
            maxValue[i], minValue[i], cluster_center[:, i]) for i in range(len(maxValue))]
        self.rules = cluster_center

        self.viewInputs(ax2)
        plt.show()

        self.entrenar(data)

    def entrenar(self, data):
        P = data[:, :-1]  # datos
        T = data[:, -1]  # targets

        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8)
                         for i in self.inputs])
        f = [np.prod(gaussmf(P, cluster, sigma), axis=1)
             for cluster in self.rules]

        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti, axis=1))
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0, n_vars), len(self.rules))
        acti = np.tile(nivel_acti, [1, n_vars])
        inp = P[:, orden]

        A = acti*inp/sumMu

        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        self.solutions = solutions  
        return 0

    def evalfis(self, data):
        sigma = np.array([(input.maxValue-input.minValue)
                         for input in self.inputs])/np.sqrt(8)
        f = [np.prod(gaussmf(data, cluster, sigma), axis=1)
             for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti, axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0, n_vars), n_clusters)
        acti = np.tile(nivel_acti, [1, n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu, axis=1)

    def viewInputs(self, ax):
        for input in self.inputs:
            input.view(ax)

    def calcular_mse(self, datos, salidas):
        error = mean_squared_error(datos[:, 1], salidas)
        plt.figure(figsize=(10,5))
        plt.scatter(datos[:, 0], datos[:, 1], label='Targets', c='m')
        plt.scatter(datos[:, 0], salidas,
                    label='Salida del modelo', c='orange')

        # Grafica las lineas entre el target y la salida del modelo
        # Si se equivocan con mas del 20% lo marca en rojo
        for i, point in enumerate(datos[:, 0]):
            x = datos[i, 0]
            y = datos[i, 1]
            color = 'red' if abs(salidas[i]-y)/100 > 0.2 else 'green'
            plt.plot([x, x], [y, salidas[i]], c=color,
                     alpha=0.5, linestyle='--')

        plt.xlabel('Tiempo [ms]')
        plt.ylabel('VDA')
        plt.title('Targets vs Salida del modelo')
        plt.text(200, 620, f'MSE = {error}', bbox={
                 'facecolor': 'oldlace', 'alpha': 0.5, 'pad': 8})
        plt.legend()
        plt.grid(True)
        plt.show()

        return error


def generarSugeno(data_train, data_test, radio):
    """Genera un modelo de Sugeno en base a los datos y el radio del clustering sustractivo
    -return: fis, mse"""

    sugeno = fis()
    # Con esto determinamos el radio de aceptacion para el cluster
    # Mientras mas grande, menos clusters
    sugeno.genfis(data_train, radio)
    salidas = sugeno.evalfis(np.vstack(data_test[:, 0]))
    return sugeno, sugeno.calcular_mse(data_test, salidas)


def graficarDatos(datos, xlabel, ylabel, title, plot=False):
    plt.figure(figsize=(10,5))
    if plot:
        plt.plot(datos[:, 0], datos[:, 1])
    else:
        plt.scatter(datos[:, 0], datos[:, 1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
# -------------------------------------------- MAIN -------------------------------------------------------


path = 'samplesVDA1.txt'

data_y = cargar_datos(path)
data_x = genero_datosX(len(data_y))
# .T hae una matriz traspuesta
# Hace que cada valor de x corresponda con el de y
data = np.vstack((data_x, data_y)).T

# INCISO A)
graficarDatos(data,'Tiempo [ms]','VDA','Mediciones de VDA')

# INCISO B)
# separa en train y test
data_train, data_test = dividir_datos(data, 0.4)
radios = [0.1, 0.25, 0.5, 1, 1.5, 2] #se me ocurrieron estos valores, podemos poner otros
errores = []
modelos = []
for radio in radios:
    modelo, mse = generarSugeno(data_train, data_test, radio)
    errores.append(mse)
    modelos.append(modelo)

estadisticas = np.vstack((radios, errores)).T
graficarDatos(estadisticas,'Radio de vecindad', 'MSE', 'MSE vs R', plot = True)

# Inciso C)
minimo = np.argmin(np.array(estadisticas)[:,1])
print(f'El mejor modelo es el de R = {radios[minimo]} y MSE = {errores[minimo]}')

# Inciso D)
sobremuestra = np.arange(min(data_x),max(data_x),0.1)
salidas = modelos[minimo].evalfis(np.vstack(sobremuestra))
graficarDatos(np.vstack((sobremuestra,salidas)).T, 'Tiempo [ms]', 'VDA', 'Sobremuestreo')

