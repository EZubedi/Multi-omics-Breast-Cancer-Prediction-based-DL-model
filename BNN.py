import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix,mean_squared_error,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report , roc_curve, f1_score, accuracy_score, recall_score , roc_auc_score,make_scorer
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load the dataset
df = pd.read_csv('selected_features.csv', encoding_errors= 'replace')
# split into input (X) and output (y) variables
data = df.values
X =data[:,0:17]
print(X.shape)
Y = df.tumour_confirmed

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for LSTM input
"""X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])"""

sns.set(font_scale=1.7)
# define the Binary Neural Network (BNN) model with relu activation mode and batch_size 32
model = Sequential()
model.add(Dense(12, input_shape=(17,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, Y, epochs=200, batch_size=32, verbose=0)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
# evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('accuracy: %.2f' % (accuracy*100))

#Fitting the to the Training set
model.fit(X_train, y_train, epochs=200, batch_size=32)
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)
hist = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_test,y_test))
print(model.evaluate(X_test, y_test)[1])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('BNN')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
results = model.evaluate(X_test,y_test, verbose = 0)
print('test loss, test acc:', results)
results1 = model.evaluate(X_train, y_train, verbose = 0)
print('train loss, train acc:', results1)

# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
#Plot the confusion matrix.
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=23)
plt.xlabel('Actual',fontsize=23)
plt.title('BNN',fontsize=23)
plt.show()

from sklearn.metrics import precision_score
# Calculate precision
precision = precision_score(y_test, y_pred)

print("Precision:", precision)

from sklearn.metrics import recall_score
# Calculate recall
recall = recall_score(y_test, y_pred)

print("Recall:", recall)

from sklearn.metrics import f1_score
# Calculate F1 score
f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)
