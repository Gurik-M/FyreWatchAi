import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
import numpy
from keras.callbacks import EarlyStopping, ModelCheckpoint

#THIS IS USED TO TRAIN THE MODEL

# Importing the dataset
dataset = pd.read_csv('dataset.csv')
dataset.head()

X=dataset.iloc[:,0:8]
Y=dataset.iloc[:,8]

X.head()
obj=StandardScaler()
X=obj.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X, Y, test_size=0.10)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


classifier = Sequential()

# Adding the input layer and first hidden layer
# Adding input dim for first input layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 
                                                      'relu', input_dim = 8)) 
# Adding a dropout
classifier.add(Dropout(rate =  0.20))

# Adding the second hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation 
                                                                   = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation 
                                                               = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics 
                                                          = ['accuracy'])


history = classifier.fit(X_train, y_train, validation_split=0.20, batch_size = 9, epochs = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.96)

# summarize history for loss and accuracy (note that val stands for "validate")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# saving the classifier 
classifier.save("model.h5")
