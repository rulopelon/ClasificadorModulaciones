
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
import torch
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn


img_size = 224
class SimpleCNN(nn.Module):
    def __init__(self,base_model,n_layers,n_classes,unfreezed_layers,list_dropouts,list_neuronas_salida):
        super(SimpleCNN, self).__init__()
        self.base_model = base_model
        self.num_classes = n_classes
        self.n_layers = n_layers
        self.in_features = self.base_model.classifier[1].in_features
        # Freeze convolutional layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze specified number of layers
        for name, child in self.base_model.features.named_children():
            if int(name) >= unfreezed_layers:  # Descongelar capas a partir del bloque 14
                for param in child.parameters():
                    param.requires_grad = True

        new_classifier = nn.Sequential()

        for i in range(1, self.n_layers):
            if i== 1:
                new_classifier.add_module(f'dp{i}',nn.Dropout(list_dropouts[i]))
                new_classifier.add_module(f'fc{i}', nn.Linear(self.in_features, list_neuronas_salida[i]))
                new_classifier.add_module(f'relu{i}',nn.ReLU())
                
            else:

                new_classifier.add_module(f'dp{i}',nn.Dropout(list_dropouts[i]))
                new_classifier.add_module(f'fc{i}', nn.Linear(list_neuronas_salida[i-1], list_neuronas_salida[i]))
                new_classifier.add_module(f'relu{i}',nn.ReLU())
        # Add a new softmax output layer
        new_classifier.add_module(f'dp{i}',nn.Dropout(list_dropouts[-1]))
        new_classifier.add_module(f'fc{i}', nn.Linear(list_neuronas_salida[-2], self.num_classes))
        new_classifier.add_module(f'relu{i}',nn.Softmax(dim=1))

        self.base_model.classifier = new_classifier

    

    def forward(self, x):

        x = self.base_model(x)

        return x
    
    
def load_model():
    #dict = torch.load("DeepLearning/model_48.pth",map_location=torch.device('cpu'))
    dict = torch.load("EvaluationCode/VggFinal.pth")
     
    n_layers = 22
    lr =  0.004832166760643208
    optimizer_name = "SGD"
    n_classes = 9
    unfreezed_layers =7
    # Eligiendo el número de capas internas
    list_neuronas_salida  = [11, 22, 4, 70, 14, 93, 1, 40, 26, 46, 3, 88, 87, 77, 30, 49, 69, 14, 7, 12, 6, 9]

    list_dropouts =[0.20867398564003678, 0.14812829031403424, 0.010586726141562947, 0.36212119453376546, 0.072041156086945, 0.0683279606288445, 0.20022931539203853, 0.3819951247876895, 0.24349613158108577, 0.16906119513536613, 0.30051365978928807, 0.4173251179196897, 0.3722084692143477, 0.3286968498651176, 0.49455000055308523, 0.241722420014097, 0.4718793395158464, 0.23018361650083752, 0.11922626261033709, 0.16319844025295294, 0.1498884405304198, 0.09483176445102617]

    base_model = torchvision.models.vgg11(weights='DEFAULT')



 
    model = SimpleCNN(base_model,n_layers,n_classes,unfreezed_layers,list_dropouts,list_neuronas_salida)
    dict = torch.load("EvaluationCode/VggFinal.pth")
    model.load_state_dict(dict,strict =False)
    model.eval()
    return model

model = load_model()


class_to_index = {'16qam':0, '8qam':1, 'am':2,'fsk':3,'qpsk':4,'64qam':5,'ask':6,'fm':7,'pm':8}
class_to_index = {0:'16qam', 1:'8qam', 2:'am',3:'fsk',4:'qpsk',5:'64qam',6:'ask',7:'fm',8:'pm'}
errors_by_class = {class_name: 0 for class_name in class_to_index.values()}
elements_by_class = {class_name: 0 for class_name in class_to_index.values()}


# Configura tu dataset de validación y DataLoader aquí
# Por ejemplo:
transformaciones = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor() # Convertir a tensor
    ])

    
carpeta_origen = "data/datasetTotal"
#carpeta_origen = "DeepLearning/datasetcompleto"


# Evalúa el modelo
# Cargar las imágenes utilizando ImageFolder
archivos = [f for f in os.listdir(carpeta_origen) if os.path.isfile(os.path.join(carpeta_origen, f))]

errores =0 
# Iterar sobre el dataset transformado y guardar las imágenes
for i,archivo in enumerate(archivos):
    imagen_path = os.path.join(carpeta_origen, archivo)
    imagen = Image.open(imagen_path)
    #print("Se ha cargado {}".format(imagen_path))
    class_name = archivo.split('_')[0]

    # Aplicar transformaciones de nuevo a la misma imagen
    input_tensor = transformaciones(imagen)
    
    input_tensor = input_tensor.unsqueeze(0)

    outputs = model(input_tensor)
    salida = class_to_index[outputs.argmax(1).item()]
    elements_by_class[class_name] = elements_by_class[class_name]+1
    print("La clase de entrada era: {}, la predicha es: {}".format(class_name,salida))
    if salida== class_name:
        # Se actualiza el diccionario de errores y se pone un error 
        pass
    else:
        errors_by_class[class_name] = errors_by_class[class_name]+1
        errores = errores+1
result_dict = {key: (elements_by_class[key] -errors_by_class[key])*100 / elements_by_class[key] for key in errors_by_class if key in elements_by_class}
print("La precisión final es de: {}".format((len(archivos)-errores)/len(archivos)))

plt.figure(figsize=(10, 6))
plt.bar(result_dict.keys(), result_dict.values(), color='skyblue')
plt.xlabel('Clase')
plt.ylabel('Valor')
plt.title('Valor por Clase')
plt.xticks(rotation=45)
plt.show()