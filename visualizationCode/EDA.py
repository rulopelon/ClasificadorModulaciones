import os
import re
import matplotlib.pyplot as plt

# Ruta del directorio donde están los archivos
directorio = 'C:\\Users\\raul\\Documents\\GitHub\\ClasificadorModulaciones\\data\\datasetTotal'

# Lista para almacenar los valores de SNR
snrs = []

# Listar todos los archivos en el directorio
archivos = os.listdir(directorio)

# Extraer el SNR de cada archivo
for archivo in archivos:
    # Buscar el patrón SNR seguido de cualquier número
    match = re.search(r'_(-?\d+)', archivo)

    if match:
        # Convertir el valor encontrado a entero y añadirlo a la lista
        snr = int(match.group(1))
        snrs.append(snr)

# Crear el histograma
plt.hist(snrs, bins=30, color='blue', edgecolor='black')
plt.title('Histograma de SNR')
plt.xlabel('SNR')
plt.ylabel('Frecuencia')
plt.show()