
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf

#Preprocessing the images
trdata = ImageDataGenerator(rescale=None, validation_split=0.2,
                            shear_range=0.2, zoom_range=0.2,
                            horizontal_flip=True, rotation_range=70)
traindata1 = trdata.flow_from_directory(directory="C:/Users/mai29/Documents/dataset/balanced_oneAugmented", target_size=(224,224), subset='training', class_mode='binary')#path to dataset
testdata1 = trdata.flow_from_directory(directory="C:/Users/mai29/Documents/dataset/balanced_oneAugmented", target_size=(224,224), subset='validation', class_mode='binary')

#####Input layer
input_shape = Input(shape=(224, 224, 3))

######First tower
resnet50 = ResNet50(weights='imagenet', include_top=False, input_tensor=input_shape) #ResNet50

for layer in resnet50.layers[:]:    #freezing the layers in the model
    layer.trainable = False

tower1 = resnet50(input_shape)
tower1 = Conv2D(512, (1,1), activation='relu', padding='same')(tower1)  #adding trainable convolution layers in the tower
tower1 = Conv2D(512, (1,1), activation='relu', padding='same')(tower1)
tower1 = Conv2D(512, (1,1), activation='relu', padding='same')(tower1)

tower1 = Dense(512, activation='relu')(tower1)
tower1 = Dropout(0.3)(tower1)


######Second tower
vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=input_shape)  #VGG16

for layer in vgg16.layers[:]:
    layer.trainable = False

tower2 = vgg16(input_shape)
tower2 = Conv2D(512, (1,1), activation='relu', padding='same')(tower2)
tower2 = Conv2D(512, (1,1), activation='relu', padding='same')(tower2)
tower2 = Conv2D(512, (1,1), activation='relu', padding='same')(tower2)

tower2 = Dense(512, activation='relu')(tower2)
tower2 = Dropout(0.3)(tower2)



######Third tower
denseNet = DenseNet121(weights='imagenet', include_top=False, input_tensor=input_shape)  #denseNet

for layer in denseNet.layers[:]:
    layer.trainable = False

tower3 = denseNet(input_shape)
tower3 = Conv2D(512, (1,1), activation='relu', padding='same')(tower3)
tower3 = Conv2D(512, (1,1), activation='relu', padding='same')(tower3)
tower3 = Conv2D(512, (1,1), activation='relu', padding='same')(tower3)

tower3 = Dense(512, activation='relu')(tower3)
tower3 = Dropout(0.3)(tower3)

#####merged layer
merged = concatenate([tower1, tower2, tower3], axis=1)
merged = Flatten()(merged)

#####out layer
out = Dense(512, activation='relu')(merged)
out = Dropout(0.3)(out)
out = Dense(1, activation='softmax')(out)

#defining whole model
model = Model(input_shape, out)

#compiling model
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), loss='binary_crossentropy', metrics=['acc', 'Precision', 'Recall', 'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives'])

model.summary()

history = model.fit(traindata1, steps_per_epoch=445, epochs=10, validation_data=testdata1, validation_steps=111, workers=1)

val_acc = history.history['val_acc']
acc = history.history['acc']
epochs = range(len(val_acc))

plt.plot(epochs, acc, 'b')
plt.plot(epochs, val_acc, 'r')
plt.ylim(0,1)
plt.show()

#confusion matrix
Y_pred = model.predict(testdata1, testdata1.samples/32)
y_pred = np.argmax(Y_pred, axis=1)
print("Confusion matrix")
print(confusion_matrix(testdata1.classes, y_pred))