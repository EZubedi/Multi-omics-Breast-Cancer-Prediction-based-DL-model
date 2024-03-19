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
from sklearn.metrics import precision_score
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load the dataset

df = pd.read_csv('selected_features.csv', encoding_errors= 'replace')

# split into input (X) and output (y) variables
data = df.values
X =data[:,0:17]
print(X.shape)
y = df.tumour_confirmed


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the data for LSTM input
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=32, input_shape=(1, X.shape[1])))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model

history= model.fit(X_train,y_train,epochs=200,verbose=1, batch_size=32, validation_data=(X_test,y_test))
y_pred = model.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)
#Plots of Accuracy and Loss
sns.set(font_scale=1.7)
def plotLearningCurve(history,epochs):
  epochRange = range(1,epochs+1)
  """plt.plot(epochRange,history.history['accuracy'])
  plt.plot(epochRange,history.history['val_accuracy'])
  plt.title('Model Accuracy for LSTM', fontsize=23)
  plt.xlabel('Epoch', fontsize=23)
  plt.ylabel('Accuracy', fontsize=23)
  plt.legend(['Train','Validation'],loc='upper left', fontsize=23)
  plt.show()"""

  plt.plot(epochRange,history.history['loss'])
  plt.plot(epochRange,history.history['val_loss'])
  plt.title('Model Loss for LSTM', fontsize=23)
  plt.xlabel('Epoch', fontsize=23)
  plt.ylabel('Loss', fontsize=23)
  plt.legend(['Train','Validation'],loc='upper left', fontsize=23)
  plt.show()
plotLearningCurve(history,200)


# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
#Plot the confusion matrix.
sns.heatmap(cm, annot=True, annot_kws={'size': 23}, fmt='g')
           
plt.ylabel('Prediction',fontsize=23)
plt.xlabel('Actual',fontsize=23)
plt.title('LSTM',fontsize=23)
plt.show()

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