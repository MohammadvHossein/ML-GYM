from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

data = datasets.load_digits()
X = data.data
y = data.target
print(f"Data shape: {X.shape}, Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pickle.dump(scaler, open("standardScaler.pkl", 'wb'))

gbc = GradientBoostingClassifier()

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=gbc,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")

best_gbc = grid_search.best_estimator_
y_pred = best_gbc.predict(X_test)

with open("GradientBoosting_best.pkl", 'wb') as model_file:
    pickle.dump(best_gbc, model_file)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
