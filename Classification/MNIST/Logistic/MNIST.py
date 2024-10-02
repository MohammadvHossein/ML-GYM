from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

clf = LogisticRegression(C=1, penalty='l2', max_iter=1000)
clf.fit(X_train, y_train)

with open("logistic_model.pkl", 'wb') as model_file:
    pickle.dump(clf, model_file)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
