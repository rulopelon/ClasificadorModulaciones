import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import optuna
import numpy as np
import scipy
import os
from tqdm import tqdm
import time
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directorioTrain = "DeepLearning/datasetAugmented/training"
directorioTest = "DeepLearning/datasetAugmented/validation"
archivo_registro = 'registro_carpetas.txt'

class CustomDataset(Dataset):
    """Dataset para cargar archivos .jpg."""
    
    def __init__(self, directory, classes,transform=None):
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
        self.classes = classes
        self.class_to_index = {cls: idx for idx, cls in enumerate(classes)}

        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = self.filenames[idx]

        class_name = filename.split('_')[0]
        
        # Convertir la etiqueta de clase a un índice
        class_idx = self.class_to_index[class_name]
       
        label_one_hot = torch.zeros(len(self.classes))
        label_one_hot[class_idx] = 1
        image = Image.open(os.path.join(self.directory,filename))
        convertir_a_tensor =  transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Convertir a escala de grises
    transforms.resize(size=(256,256)),
    transforms.ToTensor(), # Convertir a tensor
])

        image = convertir_a_tensor(image)
        return image,label_one_hot



class EarlyStopper:
    def __init__(self, patience=4, min_delta=0):
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
    def __init__(self,n_layers,n_classes,list_kernels,list_paddings,channel_list,bias_list,list_strides,features_lineales,list_pool_kernels, list_pool_strides):
        super(SimpleCNN, self).__init__()
        self.earlyStopper = EarlyStopper()
        self.list_kernels = list_kernels
        self.convolutional =_layers = []
        self.list_pool_kernels = list_pool_kernels
        self.list_pool_strides = list_pool_strides
        self.features_lineales = features_lineales
        self.n_layers = n_layers
        self.n_classes = n_classes
                
        for i in range(1,self.n_layers+1):
            if i== 1:
                setattr(self, f'conv{i}', nn.Conv2d(1, channel_list[i-1], 
                                              kernel_size=list_kernels[i-1], 
                                              stride=list_strides[i-1], 
                                              padding=list_paddings[i-1], 
                                              bias=bias_list[i-1]))
            else:
                setattr(self, f'conv{i}', nn.Conv2d(channel_list[i-2], channel_list[i-1], 
                                              kernel_size=list_kernels[i-1], 
                                              stride=list_strides[i-1], 
                                              padding=list_paddings[i-1], 
                                              bias=bias_list[i-1]))
    

    def forward(self, x):

        for i in range(1, self.n_layers+1):  # Asume que 'channel_list' tiene la cantidad correcta de canales.
            conv_layer = getattr(self, f'conv{i}')
            x = conv_layer(x)
            relu = nn.ReLU()
            x = relu(x)
            maxPool = nn.MaxPool2d(self.list_pool_kernels[i-1], stride=self.list_pool_strides[i-1] )
            x = maxPool(x)
        dimension_entrada = x.shape[1]*x.shape[2]*x.shape[3]

        self.fc1 = nn.Linear(dimension_entrada,self.features_lineales).to(device)
        self.fc2 = nn.Linear(self.features_lineales,self.n_classes).to(device)

        x = x.view(-1,dimension_entrada)
        x = self.fc1(x)
        relu = nn.ReLU()
        x = relu(x)
        x = self.fc2(x)
        softmax=  nn.Softmax(dim=1)
        x = softmax(x)
        return x
    
    
def train_and_validate(model, device, train_loader, val_loader, epochs, optimizer,trial):
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
            try:
                outputs = model(images)
            except Exception as e:
                print(e)
                torch.cuda.empty_cache()
                print("Combinación incorrecta")
                raise optuna.exceptions.TrialPruned
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += (outputs.argmax(1) == labels.argmax(1)).sum().item()


        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
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

        valid_loss /= len(val_loader)
        valid_accuracy /= len(val_loader.dataset)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)

        print(f'Epoch {epoch + 1}/{epochs} - '
                f'Validation Loss: {valid_loss:.4f}, '
                f'Validation Accuracy: {valid_accuracy:.4f}')


        trial.report(valid_accuracy,step = epoch)
        if trial.should_prune():
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

        if model.earlyStopper.early_stop(valid_loss):
            print("Se ha hecho early stopping")
            return valid_accuracy
            #validation_accuracy = input("Dime la precisión")
            #trial.report(validation_accuracy,epoch+1)
            #raise optuna.exceptions.TrialPruned()
        
    # UNa vez que ha terminado de entrenar, lo guarda
    model_file = f"model_{trial.number}.pth"
    torch.save(model.state_dict(), model_file)

    return valid_accuracy

def objective(trial):


    with open(archivo_registro, 'r') as archivo:
        clases = archivo.readlines()

    # Remover los saltos de línea y posibles duplicados
    clases = list(set([nombre_clase.strip() for nombre_clase in clases]))
    n_classes = len(clases)
    # Hiperparámetros a optimizar
    n_layers = trial.suggest_int("n_layers",5,30)
    lr = trial.suggest_float("lr", 1e-5, 1e-1,log = True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    features_lineales = trial.suggest_int("features_lineales",1,500)
    # Eligiendo el número de capas internas
    kernel_list = []
    padding_list = []
    bias_list = []
    channel_list = []
    stride_list = []
    list_pool_kernels = []
    list_pool_strides  = []

    for i in range(n_layers):
        # Kernel
        kernel_size = trial.suggest_int("kernel_size_"+str(i),1,5)
        kernel_list.append(kernel_size)
        # Padding
        padding = trial.suggest_int("padding_"+str(i), 1,2)
        padding_list.append(padding)
        # Bias
        bias = trial.suggest_categorical("bias_"+str(i),[True,False])
        bias_list.append(bias)
        # Channels
        channels = trial.suggest_int("channels_"+str(i),1,2)
        channel_list.append(channels)
        # Stride
        stride = trial.suggest_int("strides"+str(i),1,2)
        stride_list.append(stride)
        # Pool kernel
        pool_kernel = trial.suggest_int("pool_kernel"+str(i),1,2)
        list_pool_kernels.append(pool_kernel)
        # Stride kernel
        stride_kernel = trial.suggest_int("stride_kernel"+str(i),1,2)
        list_pool_strides.append(stride_kernel)


    try:
        model = SimpleCNN(n_layers,n_classes,kernel_list,padding_list,channel_list,bias_list,stride_list,features_lineales,list_pool_kernels, list_pool_strides).to(device)
        model = model.to(device)
    except Exception as e:
        print(e)

        print("Demasiado grande")
        torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    """
    DATA AUGMENTATION
    """

    with open(archivo_registro, 'r') as archivo:
        clases = archivo.readlines()

    # Remover los saltos de línea y posibles duplicados
    clases = list(set([nombre_clase.strip() for nombre_clase in clases]))
    train_dataset = CustomDataset(directory=directorioTrain,classes=clases)
    test_dataset = CustomDataset(directory=directorioTest,classes=clases)

    # Datos de entrenamiento y validación
    batch_size = 64 #TODO Aumentar batch size 128 meter en optuna
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,num_workers=2)

    val_loader = DataLoader(test_dataset,
                            batch_size=batch_size, shuffle=False,num_workers=2)

    epochs = 1000  # Puedes ajustar esto según sea necesario
   
    accuracy = train_and_validate(model, device, train_loader, val_loader, epochs, optimizer,trial)

    return accuracy


                                  
if __name__ == '__main__':
    #torch.multiprocessing.set_start_method("spawn")# good solution !!!!

    study = optuna.create_study(direction="maximize")
    #study = optuna.load_study(study_name="DatasetCompleto", storage="mysql://root:T3mp0r4l@192.168.253.109/optimizacion")
        
    #study.optimize(objective, n_trials=100)

    study.optimize(objective, n_trials=10000)  # Ajusta el número de trials según sea necesario

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")









