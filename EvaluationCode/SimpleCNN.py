import torch.nn as nn



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

        for i in range(1, self.n_layers-1):
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


            
        x = x.view(-1,self.in_features)
        for i in range(1, self.n_layers-1):  # Asume que 'channel_list' tiene la cantidad correcta de canales.
            fc = getattr(self, f'fc{i}')
            x = fc(x)
        x = self.fc_final(x)

        return x