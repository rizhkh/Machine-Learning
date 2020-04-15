import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

###### THIS PART IS FROM CLASSIFICATION
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[ : , 3:13 ].values
y = dataset.iloc[ :, 13 ].values


#Note: Execute above part first and then input X to check data before it is encoded again

# Encoding categorical data & the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_One = LabelEncoder()
X[:, 1] = labelencoder_X_One.fit_transform(X[:, 1]) # X[:, 1] we put 1 as the index we need is '1' which is France Spain etc
## Input X to check : Once above code is executed you will notice that France has become 0, Spain is 1 and Germany 2
labelencoder_X_Two = LabelEncoder()
X[:, 2] = labelencoder_X_Two.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Removing dummy variable index 0 to not fall in dummy variable trap
X = X[: , 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling required for Deep learning as it requires a lot of computation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#########################

## Artificial Neural Network

from keras.models import Sequential
from keras.layers import Dense

## Initialize the artificial neural network
# Two ways to initialize the ann: by defining the sequence of layers or defing the graph
# we are doing it with sequence of layers
classifier = Sequential()
#Input layer and then the hidden layers
classifier.add( Dense( output_dim = 6, init = 'uniform', activation = 'relu', input_dim=11 ) )
#ABOVE LINE: glorot_uniform or just uniform is used to initialize the weights
# activation is the activation layer we want in hidden layer
#Input_dim is compulsory without it we get exception : input_dim is number of independent variables

#Second Hidden Layer
classifier.add( Dense( output_dim = 6, init = 'uniform', activation = 'relu') ) # adding another hidden layer

#Output layer (final layer of NN) -  we are replacing rectifier 'relu' activating function to sigmoid act. function
classifier.add( Dense( output_dim = 1, init = 'uniform', activation = 'sigmoid') ) # adding another hidden layer

# Compiling the ANN - applying stochastic gradient descent on whole NN
classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# Fitting ANN to trainign set
classifier.fit( X_train, y_train, batch_size = 10 , nb_epoch = 100)

#########################

## Predictions and Evaluating the models
## Note : here we choose number of epochs( the number of times we train our ANN on whole training set)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)