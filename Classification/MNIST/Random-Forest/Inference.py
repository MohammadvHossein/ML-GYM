import pickle
import numpy as np

scaler = pickle.load(open("standardScaler.pkl", 'rb'))
with open("RandomForest_best.pkl", 'rb') as model_file:
    clf = pickle.load(model_file)


def predict_digit(new_data):
    new_data_scaled = scaler.transform(new_data)

    predictions = clf.predict(new_data_scaled)

    return predictions


new_data = np.random.randint(0, 255, size=(64)).reshape(1, -1)
predictions = predict_digit(new_data)

print("Predicted labels:", predictions)
