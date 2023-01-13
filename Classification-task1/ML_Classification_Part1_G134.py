'''
Machine Learning 21-22
Project - Classification First Problem

@authors: Group 134
Diogo Delgado, 92676
Mariana Lima, 92707
'''

import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from eval_scores import scores
import random

seed = 18
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# Load data
Xtrain = np.load('Xtrain_Classification_Part1.npy') 
Ytrain = np.load('Ytrain_Classification_Part1.npy')
Xtest = np.load('Xtest_Classification_Part1.npy')

#print(Xtrain.shape) (6470,2500)
#print(Xtest.shape)  (1164,2500)

# Scaling Data
Xtrain = np.true_divide(Xtrain, 255)
Xtest = np.true_divide(Xtest, 255)

# Reshape Data
XtrainRS = np.zeros(shape= (6470,50,50,1))
XtestRS = np.zeros(shape= (1164,50,50,1))

for i in range(0,6470):
    XtrainRS[i] = np.reshape(Xtrain[i,:], (50,50,1))
    
for i in range(0,1164):
    XtestRS[i] = np.reshape(Xtest[i,:], (50,50,1))

    
# Splitting Data
Xtraining, Xvalidation, Ytraining, Yvalidation = train_test_split(XtrainRS, Ytrain, test_size=0.15, random_state = 0)
YvalidationScore = Yvalidation

#print(Xtraining.shape)     (5499, 50, 50, 1)
#print(Xvalidation.shape)   (971, 50, 50, 1)

# Flip Horizontal Data Augmentation
XtrainingFlip = np.zeros(shape= (10998,50,50,1))

for i in range(0,5499):
    XtrainingFlip[i] = Xtraining[i]
    XtrainingFlip[i+5499] = np.fliplr(Xtraining[i])
    
YtrainingFlip = np.tile(Ytraining, 2)    


'''     
# ----------------------------- Plotting data ---------------------------------
# creating a plot
pixel_plot = plt.figure()
   
# plotting a plot
pixel_plot.add_subplot()
   
# customizing plot
plt.title("Image")
pixel_plot = plt.imshow(XtrainingFlip[8+5499], cmap='gray', interpolation='nearest')
   
plt.colorbar(pixel_plot)
     
# show plot
plt.show(pixel_plot)       
'''


# One-hot encode target column
YtrainingFlip = to_categorical(YtrainingFlip)
Yvalidation = to_categorical(Yvalidation)

# Create convolutional base
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.15))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.15))
#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.15))

model.add(Flatten())

#model.summary()

model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

#model.summary()


# Compile and train model
#opt = opt = SGD(lr=0.01, momentum=0.9)
opt = 'adam'
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

EarlyStop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(XtrainingFlip, YtrainingFlip, validation_data=(Xvalidation, Yvalidation), epochs=200, callbacks=[EarlyStop])


# Evaluate model 
print()
_, acc = model.evaluate(Xvalidation, Yvalidation, verbose=0)
print('> Validation Accuracy = %.3f' % (acc * 100.0))
print()

# Plots 
train_metrics = history.history['loss']
val_metrics = history.history['val_loss']
epochs = range(1, len(train_metrics) + 1)
plt.plot(epochs, train_metrics)
plt.plot(epochs, val_metrics)
plt.title('Cross entropy loss')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.legend(["Training loss", 'Validation loss'])
plt.show()

train_metrics = history.history['accuracy']
val_metrics = history.history['val_accuracy']
epochs = range(1, len(train_metrics) + 1)
plt.plot(epochs, train_metrics)
plt.plot(epochs, val_metrics)
plt.title('Classification accuracy')
plt.xlabel("Epochs")
plt.ylabel('Accuracy')
plt.legend(["Training accuracy", 'Validation accuracy'])
plt.show()

'''
# Check 
Ypredict = model.predict(Xvalidation)
#print(model.predict(Xvalidation[:20]))
#print(Yvalidation[:20])

Yprediction = np.zeros(971)
for i in range (0,971):
    if Ypredict[i,0]>Ypredict[i,1]:
        Yprediction[i]=0
    else:
        Yprediction[i]=1

scores(YvalidationScore, Yprediction,'c')
'''

# =============================================================================
# Final prediction

YtestPredict = model.predict(XtestRS)

Ytest = np.zeros(1164)
for i in range (0, 1164):
    if YtestPredict[i,0]>YtestPredict[i,1]:
        Ytest[i]=0
    else:
        Ytest[i]=1    


np.save('Ytest_Classification_Part1_G134_18', Ytest, allow_pickle=True, fix_imports=True)




