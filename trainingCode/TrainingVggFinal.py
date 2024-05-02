import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import scipy
import os
from tqdm import tqdm
import time
from PIL import Image
import torchvision
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directorioTrain = "training"
directorioTest = "validation"


class_to_index = {'16qam':0, '8qam':1, 'am':2,'fsk':3,'qpsk':4,'64qam':5,'ask':6,'fm':7,'pm':8}
n_classes = len(class_to_index.values())


class CustomDataset(Dataset):
    """Dataset para cargar archivos .jpg."""
    
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directorio con todos los archivos .mat.
            transform (callable, optional): Opcional transformación a ser aplicada
                en una muestra.
        """
        self.directory = directory
        self.transform = transform
        self.files = [f for f in os.listdir(directory) if f.endswith('.png')]
        self.filenames = os.listdir(directory)
        self.class_to_index = class_to_index
        self.classes = n_classes

        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = self.filenames[idx]

        class_name = filename.split('_')[0]
        
        # Convertir la etiqueta de clase a un índice
        class_idx = self.class_to_index[class_name]
       
        label_one_hot = torch.zeros(self.classes)
        label_one_hot[class_idx] = 1
        image = Image.open(os.path.join(self.directory,filename))
        convertir_a_tensor =  transforms.Compose([
    #transforms.Grayscale(num_output_channels=1), # Convertir a escala de grises
    transforms.Resize(size=(224,224)),
    transforms.ToTensor() # Convertir a tensor
])

        image = convertir_a_tensor(image)
        return image,label_one_hot



class EarlyStopper:
    def __init__(self, patience=30, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        # TODO: La epoch anterior menos la actual tien que ser menor que el delta 0.0001 paciencia 0
        return False
    
class SimpleCNN(nn.Module):
    def __init__(self,base_model,n_layers,n_classes,unfreezed_layers,list_dropouts,list_neuronas_salida):
        super(SimpleCNN, self).__init__()
        self.earlyStopper = EarlyStopper()
        self.base_model = base_model
        self.num_classes = n_classes
        self.n_layers = n_layers
        self.in_features = self.base_model.classifier[0].in_features
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
    
    
def train_and_validate(model, device, train_loader, val_loader, epochs, optimizer):
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_accuracy': [], 'valid_loss': [], 'valid_accuracy': []}
    print("Empieza el entrenamiento")

    creado = False
    for epoch in tqdm(range(epochs)):
        model.train()
        training_losses = []
        tiempo_comienzo = time.time()
        correct_training = 0
        train_loss = 0.0
        train_accuracy = 0.0

        tiempo_inicial = time.time()
        for images, labels in train_loader:
            tiempo_actual = time.time()
            tiempo_inicial =tiempo_actual
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += (outputs.argmax(1) == labels.argmax(1)).sum().item()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
        train_accuracy = train_accuracy*100
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        print(f'Epoch {epoch + 1}/{epochs} - '
                f'Train Loss: {train_loss:.4f}, '
                f'Train Accuracy: {train_accuracy:.4f}')

        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            valid_accuracy += (outputs.argmax(1) == labels.argmax(1)).sum().item()

        if model.earlyStopper.early_stop(valid_loss):
                model_file = f"VggFinal.pth"
                torch.save(model.state_dict(), model_file)
                #wandb.save(model_file)
        valid_loss /= len(val_loader)
        valid_accuracy /= len(val_loader.dataset)
        valid_accuracy = valid_accuracy*100
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)

        print(f'Epoch {epoch + 1}/{epochs} - '
                f'Validation Loss: {valid_loss:.4f}, '
                f'Validation Accuracy: {valid_accuracy:.4f}')




    # UNa vez que ha terminado de entrenar, lo guarda
    model_file = f"VggFinal.pth"
    torch.save(model.state_dict(), model_file)
    #wandb.save(model_file)

    return valid_accuracy,history

def main():



    # Hiperparámetros a optimizar
    n_layers = 22
    lr =  0.004832166760643208
    optimizer_name = "SGD"

    unfreezed_layers =7
    # Eligiendo el número de capas internas
    list_neuronas_salida  = [11, 22, 4, 70, 14, 93, 1, 40, 26, 46, 3, 88, 87, 77, 30, 49, 69, 14, 7, 12, 6, 9]

    list_dropouts =[0.20867398564003678, 0.14812829031403424, 0.010586726141562947, 0.36212119453376546, 0.072041156086945, 0.0683279606288445, 0.20022931539203853, 0.3819951247876895, 0.24349613158108577, 0.16906119513536613, 0.30051365978928807, 0.4173251179196897, 0.3722084692143477, 0.3286968498651176, 0.49455000055308523, 0.241722420014097, 0.4718793395158464, 0.23018361650083752, 0.11922626261033709, 0.16319844025295294, 0.1498884405304198, 0.09483176445102617]

    base_model = torchvision.models.vgg11(weights='DEFAULT')



 
    model = SimpleCNN(base_model,n_layers,n_classes,unfreezed_layers,list_dropouts,list_neuronas_salida).to(device)
    model = model.to(device)

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    """
    DATA AUGMENTATION
    """

    
    train_dataset = CustomDataset(directory=directorioTrain)
    test_dataset = CustomDataset(directory=directorioTest)

    # Datos de entrenamiento y validación
    batch_size = 128 #TODO Aumentar batch size 128 meter en optuna
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,num_workers=2)

    val_loader = DataLoader(test_dataset,
                            batch_size=batch_size, shuffle=True,num_workers=2)

    epochs = 1000  # Puedes ajustar esto según sea necesario
   
    accuracy = train_and_validate(model, device, train_loader, val_loader, epochs, optimizer)

    return accuracy


                                  
if __name__ == '__main__':

    accuracy, history = main()
    # Gráfico de losses
    # Gráfico de losses
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['valid_loss'], label='Valid Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_vs_validation_loss.png')  # Guardar el gráfico antes de mostrarlo

    # Gráfico de accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['valid_accuracy'], label='Valid Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_vs_validation_accuracy.png')  # Guardar el gráfico antes de mostrarlo










