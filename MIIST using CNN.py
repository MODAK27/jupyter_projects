#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train_path = "mnist_train.csv"
data = pd.read_csv(train_path)
data.describe()


# In[2]:


#https://www.kaggle.com/rahuldshetty/mnist-hand-written-digit-classification
data = data.values
y = data[:,0]
x = data[:,1:]
print(y.shape,x.shape)


# In[3]:


#reshaping into 3D
x = x.reshape((-1,28,28,1))


# In[4]:


print(x.shape)


# In[8]:


import cv2
import matplotlib.pyplot as plt

def display(image):
    plt.imshow(image)
    plt.show()
    
#reshaping into 2d shape to show in image    
sample  = x[2].reshape((28,28))
display(sample)


# In[9]:


import keras
from keras.models import Sequential
from keras.layers import Dense,ReLU,Conv2D,MaxPooling2D,Activation,Flatten,Dropout,BatchNormalization


# In[10]:


#sequential api used for connection of layers piecewise
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(28,28,1),name='0'))
#after paddding size= (n+2p-f+1)=same as 16
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
# output layer after max pooling is divide by 2 of i/p
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',name='1'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',name='2'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',name='3'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
# 4  convolution kernel layers that is convolved with the layer input to produce a tensor of outputs. 

model.add(Flatten())

model.add(Dense(64))
#64 neurons
model.add(Activation('relu'))
model.add(Dropout(0.29))
#Dropout: A Simple Way to Prevent Neural Networks from Overfitting
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.21))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()


# In[11]:



model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#since categoraical loss function sued so we need o/p in categorical format


# In[12]:


from keras.utils import to_categorical
y = to_categorical(y)#scaling it such that 9 digit doesn't get higher priority than 1 for o/p .


# In[13]:


y.shape


# In[14]:


history = model.fit(x,y,validation_split=0.2,epochs = 18,verbose=1)


# In[15]:


model.save('model.h5')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[20]:


def get_result(image):
    # image should be 28x28x1 with [0-1] values
    images = np.array([image])
    res = list(model.predict(images)[0])
    mx = max(res)
    return res.index(mx)
idx = 960
sample = x[idx]
print(y[idx])
display(sample.reshape(28,28))
print(get_result(sample))


# In[ ]:




