# import necessary libraries
import os
import torch
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPool1D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# load the dataset
df = pd.read_csv('selected_features.csv', encoding_errors= 'replace')
# split into input (X) and output (y) variables

data = df.values
X =data[:,0:17]
print(X.shape)
y = df.tumour_confirmed
print(y)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.5,stratify=y)
print(X_train.shape)
print(y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.metrics import confusion_matrix,mean_squared_error,precision_score,recall_score,f1_score

ann=tf.keras.models.Sequential()
print(ann)
#Hidden Layers
ann.add(tf.keras.layers.Dense(units=8,activation='relu')) #1st hidden layer i.e input layer
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
#ann.add(tf.keras.layers.Dense(units=8,activation='relu'))

#Output Layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

ann.compile(optimizer=Adam(learning_rate=0.1),loss='binary_crossentropy',metrics=['accuracy'])
history= ann.fit(X_train,y_train,epochs=200,verbose=1,validation_data=(X_test,y_test))

y_pred = ann.predict(X_test)
y_pred =(y_pred>0.5)
list(y_pred)

#Plots of Accuracy and Loss

sns.set(font_scale=1.7)

def plotLearningCurve(history,epochs):
  epochRange = range(1,epochs+1)
  """plt.plot(epochRange,history.history['accuracy'])
  plt.plot(epochRange,history.history['val_accuracy'])
  plt.title('Model Accuracy for ANN', fontsize=23)
  plt.xlabel('Epoch', fontsize=23)
  plt.ylabel('Accuracy', fontsize=23)
  plt.legend(['Train','Validation'],loc='upper left', fontsize=23)
  plt.show()"""

  plt.plot(epochRange,history.history['loss'])
  plt.plot(epochRange,history.history['val_loss'])
  plt.title('Model Loss for ANN', fontsize=23)
  plt.xlabel('Epoch', fontsize=23)
  plt.ylabel('Loss', fontsize=23)
  plt.legend(['Train','Validation'],loc='upper left', fontsize=23)
  plt.show()
plotLearningCurve(history,200)

#making Predictions
ann_pred=ann.predict(X_test)
print(ann_pred)


# compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
#Plot the confusion matrix.
sns.heatmap(cm, annot=True, annot_kws={'size': 23}, fmt='g')
           
plt.ylabel('Prediction',fontsize=23)
plt.xlabel('Actual',fontsize=23)
plt.title('ANN',fontsize=23)
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

