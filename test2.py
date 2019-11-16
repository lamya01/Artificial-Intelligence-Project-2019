 #!/usr/bin/env python -W ignore::DeprecationWarning
import mlrose
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.metrics import accuracy_score

startTime = datetime.now()
print("\nStarting execution\n")

l = []
def generateColumns(start, end):
    for i in range(start, end+1):
        l.extend([str(i)+'X', str(i)+'Y'])
    return l
eyes = generateColumns(1, 12)
import pandas as pd
df = pd.read_csv('Eyes.csv')
first_column = df.columns[0]
df = df.drop([first_column],axis = 1)
# print("hello")
X = df[eyes]
y = df['truth_value']   #the actual class labels
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# Data Normalization
from sklearn.preprocessing import StandardScaler as SC
sc = SC()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)
#not scaling y since it's already 0s and 1s
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
#converting all the scaled data to numpy arrays

iters = 50
initial_acc = 50
acc = 0

def genetic(iterations):
    nn_model_genetic = mlrose.NeuralNetwork(
	hidden_nodes = [4],
	activation = 'relu',
	algorithm = 'genetic_alg',
	max_iters = iterations,
	is_classifier = True,
	learning_rate = 0.0001,
    early_stopping = True, 
    clip_max = 5,
	max_attempts = 100,
	random_state = 3
    )
    nn_model_genetic.fit(X_train, y_train)

    y_test_pred = nn_model_genetic.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    return (y_test_pred, y_test_accuracy)



def random_hill_climb(iterations):
    nn_model1 = mlrose.NeuralNetwork(
        hidden_nodes=[4],
        activation ='relu', 
        algorithm='random_hill_climb', 
        max_iters=iterations,
        bias=True,
        is_classifier = True, 
        learning_rate=0.0001,
        early_stopping = True, 
        clip_max = 5,
        max_attempts=100,
        random_state = 3)

    nn_model1.fit(X_train_scaled, y_train)

    y_test_pred = nn_model1.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    return(y_test_pred, y_test_accuracy)

def gradDesc(iterations):
    nn_model_gradDesc = mlrose.NeuralNetwork(hidden_nodes = [4],
    activation = 'relu', 
    algorithm = 'gradient_descent', 
    max_iters = iterations,
    bias = True,
    is_classifier = True, 
    learning_rate = 0.0001, 
    early_stopping = True, 
    clip_max = 5, 
    max_attempts = 100, 
    random_state = 3)

    nn_model_gradDesc.fit(X_train_scaled, y_train)

    y_test_pred = nn_model_gradDesc.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test, y_test_pred)
    return (y_test_pred, y_test_accuracy)


dict = {}

acc3 = genetic(iters)
print("Genetic Algorithm: ",round(acc3[1]*100,2), "%\n")
dict['genetic'] = acc3[1]

acc1 = random_hill_climb(iters)
print("Random Hill Climbing: ",round(acc1[1]*100,2), "%\n")
dict['random_hill_climb'] = acc1[1]

acc2 = gradDesc(iters)
print("Gradient descent: ",round(acc2[1]*100,2), "%\n")
dict['gradDesc'] = acc2[1]


k = list(dict.keys())
v= list(dict.values())
max_acc_algo = k[v.index(max(v))]
max_acc = max(v)
# print(max_acc)

acc = max_acc

if max_acc_algo == 'gradDesc':
    algo = 1
elif max_acc_algo == 'random_hill_climb':
    algo = 2
else:
    algo = 3

print("\nExploiting algorithm: ", max_acc_algo,"\n")

loop = 1

while (loop < 5 and acc < 98):
    iters = iters + 550
    if algo == 1:
        y_test_accuracy = gradDesc(iters)
    elif  algo == 2:
        y_test_accuracy = random_hill_climb(iters)
    else:
        try:
            y_test_accuracy = genetic(iters)
        except:
            print("")
    
    if (y_test_accuracy[1] * 100 == acc):
        loop = loop + 1
    else:
        loop = 0

    acc = y_test_accuracy[1] * 100
    print("Current accuracy:", round(acc,2), "In", iters, "Iterations")
    print("Current execution time elapsed = ", datetime.now() - startTime,"\n")

    
print("The final accuracy is: ", round(acc,2), "Total", iters,"Iterations")
print("Execution time in seconds = ", datetime.now() - startTime,"\n")
