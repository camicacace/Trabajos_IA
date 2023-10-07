import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import time

from scipy.spatial import distance_matrix

#-------------------------------------- Agregado ------------------------------------------------------
def cargar_datos(path):
    datos = []

    # Abre el archivo y lee los datos
    with open(path, "r") as file:
        valorx = 0
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

#-------------------------------------- Substractive Clustering --------------------------------------------------------

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
            P = P - p * np.array([np.exp(-np.linalg.norm(v - C) ** 2 / (Rb / 2) ** 2) for v in ndata])
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

#------------------------------------------ SUGENO -------------------------------------------------------

def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

class fisRule:
    def __init__(self, centroid, sigma):
        self.centroid = centroid
        self.sigma = sigma

class fisInput:
    def __init__(self, min,max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids

    # Muestra las gaussianas
    def view(self):
        x = np.linspace(self.minValue,self.maxValue,20)
        plt.figure()
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,s)
            plt.plot(x,y)

class fis:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []



    def genfis(self, data, radii):

        start_time = time.time()
        labels, cluster_center = subclust2(data, radii)

        print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)

        # ------------------------------- Grafico con los clusters --------------------------------

        # Crear una figura y un objeto de ejes para el gráfico
        fig, ax = plt.subplots()

        # Graficar los datos
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', label='Datos')

        # Graficar los centros de los clusters
        ax.scatter(cluster_center[:, 0], cluster_center[:, 1], c='red', marker='x', s=100, label='Centros de Clusters')

        ax.set_xlabel('Eje X')
        ax.set_ylabel('Eje Y')
        ax.set_title('Gráfico de Datos y Centros de Clusters')
        ax.legend()

        plt.show()

        # ----------------------------------- termino de graficar ----------------------------------------


        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1] # Saco los targets (creo)
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T
        print("nivel acti")
        print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        print("sumMu")
        print(sumMu)
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]


        A = acti*inp/sumMu

        # A = np.zeros((N, 2*n_clusters))
        # for jdx in range(n_clusters):
        #     for kdx in range(nVar):
        #         A[:, jdx+kdx] = nivel_acti[:,jdx]*P[:,kdx]/sumMu
        #         A[:, jdx+kdx+1] = nivel_acti[:,jdx]/sumMu

        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions #.reshape(n_clusters,n_vars)
        print(solutions)
        return 0

    def evalfis(self, data):
        sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
        f = [np.prod(gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0,n_vars), n_clusters)
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu,axis=1)


    def viewInputs(self):
        for input in self.inputs:
            input.view()


#-------------------------------------------- MAIN -------------------------------------------------------

path = 'C:\\Users\\Usuario\\OneDrive\\Desktop\\Cami\\IA\\MisProyectos\\samplesVDA1.txt'
data_y = cargar_datos(path)
data_x = genero_datosX(len(data_y))

plt.plot(data_x, data_y)
plt.xlim(min(data_x),max(data_x))

data = np.vstack((data_x, data_y)).T

fis2 = fis()

# Con esto determinamos el radio de aceptacion para el cluster
# Mientras mas grande, menos clusters
radioAceptacion = 0.5
fis2.genfis(data, radioAceptacion)

fis2.viewInputs()

r = fis2.evalfis(np.vstack(data_x))

plt.figure()
plt.plot(data_x,data_y)
plt.plot(data_x,r,linestyle='--')

fis2.solutions

plt.plot(data_x,data_y)
plt.show()