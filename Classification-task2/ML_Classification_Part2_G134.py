'''
Machine Learning 21-22
Project - Classification Second Problem

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
Xtrain = np.load('Xtrain_Classification_Part2.npy') 
Ytrain = np.load('Ytrain_Classification_Part2.npy')
Xtest = np.load('Xtest_Classification_Part2.npy')

#print(Xtrain.shape) #(7366,2500)
#print(Xtest.shape)  #(1290,2500)

SizeTrain = Xtrain.shape[0]
SizeTest = Xtest.shape[0]


# Scaling Data
Xtrain = np.true_divide(Xtrain, 255)
Xtest = np.true_divide(Xtest, 255)


# Reshape Data
XtrainRS = np.zeros(shape= (SizeTrain,50,50,1))
XtestRS = np.zeros(shape= (SizeTest,50,50,1))

for i in range(0,SizeTrain):
    XtrainRS[i] = np.reshape(Xtrain[i,:], (50,50,1))
    
for i in range(0,SizeTest):
    XtestRS[i] = np.reshape(Xtest[i,:], (50,50,1))

    
# Splitting Data
Xtraining, Xvalidation, Ytraining, Yvalidation = train_test_split(XtrainRS, Ytrain, test_size=0.15, random_state = 0)
YvalidationScore = Yvalidation

#print(Xtraining.shape)     #(6261, 50, 50, 1)
#print(Xvalidation.shape)   #(1105, 50, 50, 1)

SizeTraining = Xtraining.shape[0]
SizeValidation = Xvalidation.shape[0]

# Weighted Classes

caucasian = np.count_nonzero(Ytraining == 0)
african = np.count_nonzero(Ytraining == 1)
asian = np.count_nonzero(Ytraining == 2)
indian = np.count_nonzero(Ytraining == 3)

w0 = (caucasian + african + asian + indian)/(caucasian*4)
w1 = (caucasian + african + asian + indian)/(african*4)
w2 = (caucasian + african + asian + indian)/(asian*4)
w3 = (caucasian + african + asian + indian)/(indian*4)

ClassWeight = {0: w0,
               1: w1,
               2: w2,
               3: w3
               }

# Flip Horizontal Data Augmentation
XtrainingFlip = np.zeros(shape= (2*SizeTraining,50,50,1))

for i in range(0,SizeTraining):
    XtrainingFlip[i] = Xtraining[i]
    XtrainingFlip[i+SizeTraining] = np.fliplr(Xtraining[i])
    
YtrainingFlip = np.tile(Ytraining, 2)    

'''
# ----------------------------- Plotting data ---------------------------------
# creating a plot
pixel_plot = plt.figure()
   
# plotting a plot
pixel_plot.add_subplot()
   
# customizing plot
plt.title("Image")
pixel_plot = plt.imshow(Xvalidation[569], cmap='gray', interpolation='nearest')
   
plt.colorbar(pixel_plot)
     
# show plot
plt.show(pixel_plot)       
'''


# One-hot encode target column
YtrainingFlip = to_categorical(YtrainingFlip, 4)
Yvalidation = to_categorical(Yvalidation, 4)

# Create convolutional base
model = Sequential()

model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
#model.add(Conv2D(256, (3, 3), activation='relu'))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.2))
#model.add(Conv2D(512, (3, 3), activation='relu'))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.2))

model.add(Flatten())

#model.summary()

model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='softmax'))

#model.summary()


# Compile and train model
#opt = opt = SGD(lr=0.01, momentum=0.9)
opt = 'adam'
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

EarlyStop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(XtrainingFlip, YtrainingFlip, class_weight=ClassWeight, validation_data=(Xvalidation, Yvalidation), epochs=200, callbacks=[EarlyStop])


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
YpredProb = model.predict(Xvalidation)
#print(YpredProb[569:570])
#print(Yvalidation[569:570])

Ypredict = np.argmax(YpredProb, axis=-1)

scores(YvalidationScore, Ypredict,'c')

ConfusionMatrix = skl.metrics.confusion_matrix(YvalidationScore, Ypredict, normalize='true')
'''

# =============================================================================
# Final prediction

YtestPredict = model.predict(XtestRS)
Ytest=np.argmax(YtestPredict, axis=-1)
np.save('Ytest_Classification_Part2_G134', Ytest, allow_pickle=True, fix_imports=True)


