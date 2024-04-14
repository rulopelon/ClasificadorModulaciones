import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optuna


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, channels=32):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels*2, channels*4, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(channels*4*4*4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
def train_and_validate(model, device, train_loader, val_loader, epochs, optimizer):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    return accuracy

def objective(trial):
    # Hiperparámetros a optimizar
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    channels = trial.suggest_categorical("channels", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    model = SimpleCNN(channels=channels).to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Datos de entrenamiento y validación
    batch_size = 64
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5,), (0.5,))
                                               ])),
                              batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5,), (0.5,))
                                           ])),
                            batch_size=batch_size, shuffle=False)

    epochs = 10  # Puedes ajustar esto según sea necesario
    accuracy = train_and_validate(model, device, train_loader, val_loader, epochs, optimizer)
    return accuracy
    
                                  

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # Ajusta el número de trials según sea necesario

print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")