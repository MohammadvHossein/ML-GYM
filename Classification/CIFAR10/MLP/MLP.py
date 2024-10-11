import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

X_train = []
y_train = []
for images, labels in trainloader:
    X_train.append(images.view(images.size(0), -1))
    y_train.append(labels)

X_train = torch.cat(X_train).numpy()
y_train = torch.cat(y_train).numpy()

X_test = []
y_test = []
for images, labels in testloader:
    X_test.append(images.view(images.size(0), -1))
    y_test.append(labels)

X_test = torch.cat(X_test).numpy()
y_test = torch.cat(y_test).numpy()

print(f"Data shape: {X_train.shape}, Target shape: {y_train.shape}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pickle.dump(scaler, open("standardScaler.pkl", 'wb'))

mlp = MLPClassifier(max_iter=1000)

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)], 
    'activation': ['relu'], 
    'solver': ['adam'], 
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

grid_search = GridSearchCV(estimator=mlp,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

best_mlp = grid_search.best_estimator_
y_pred = best_mlp.predict(X_test)

with open("MLPClassifier_best.pkl", 'wb') as model_file:
    pickle.dump(best_mlp, model_file)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
