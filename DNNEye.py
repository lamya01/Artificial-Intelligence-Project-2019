from datetime import datetime

import numpy as np
import pandas as pd
from keras.layers import Activation, Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC

startTime = datetime.now()

# eyes = [f"{i}{x}" for i in range(1, 13) for x in ('X', 'Y')]

# reading in the csv as a dataframe
dataset = np.loadtxt('EYES.csv', dtype=int, delimiter=',', skiprows=1)
y = dataset[:, -1]
X = dataset[:, 1: - 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Data Normalization
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# layering up the cnn
model = Sequential()
model.add(Dense(4, input_dim=X_train.shape[1]))
model.add(Activation("relu"))
model.add(Dense(4))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

# model compilation
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# training
model.fit(X_train, y_train, batch_size=2, epochs=50,
          validation_data=(X_test, y_test), verbose=0)

# using the learned weights to predict the target
y_pred = model.predict(X_test)

# setting a confidence threshhold of 0.9
y_pred_labels = list(y_pred > 0.9)

# print(y_pred_labels)

y_pred_labels = [1 if int(i) == 1 else 0 for i in y_pred_labels]

# plotting a confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)
print("Confusion Matrix : ")
print(cm)

# creating a dataframe to show results
df_results = pd.DataFrame()
df_results["Actual label"] = y_test
df_results["Predicted value"] = y_pred
df_results["Predicted label"] = y_pred_labels
df_results.to_csv("results.csv")

# printing execution time of script
print("Execution time in seconds = ", datetime.now() - startTime)
