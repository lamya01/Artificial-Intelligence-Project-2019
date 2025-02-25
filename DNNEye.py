# Author : Ameya Bhamare

from datetime import datetime
startTime = datetime.now()

l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l

eyes = generateColumns(1, 12)

# reading in the csv as a dataframe
import pandas as pd
df = pd.read_csv('EYES.csv')

# selecting the features and target
X = df[eyes]
y = df['truth_value']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)

# Data Normalization
from sklearn.preprocessing import StandardScaler as SC
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import numpy as np
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# importing the required layers from keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

# layering up the cnn
model = Sequential()
model.add(Dense(4, input_dim = X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# model compilation
opt = 'adam'
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

# training
model.fit(X_train, y_train, batch_size = 2, epochs = 50, validation_data = (X_test, y_test), verbose = 2)

'''
# serialize model to JSON
model_json = model.to_json()
with open("modelEyes.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("modelEyes.h5")
print("Saved model to disk")
'''

# using the learned weights to predict the target
y_pred = model.predict(X_test)

# setting a confidence threshhold of 0.9
y_pred_labels = list(y_pred > 0.9)

for i in range(len(y_pred_labels)):
    if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
    else : y_pred_labels[i] = 0

# plotting a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_labels)
print("\n")
print("Confusion Matrix : ")
print(cm)
print("\n")

# creating a dataframe to show results
df_results = pd.DataFrame()
df_results['Actual label'] = y_test
df_results['Predicted value'] = y_pred
df_results['Predicted label'] = y_pred_labels
df_results.to_csv('Results.csv')

# printing execution time of script
print("\n")
print("Execution time in seconds = ", datetime.now() - startTime)
