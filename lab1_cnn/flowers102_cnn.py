# 4th STEP --> 102 Flowers
# Feed the CNN net with Keras (base DL code from Dario)
# Olga Valls
#
from __future__ import division
import numpy as np
import pandas as pd
import keras
print('Using Keras version', keras.__version__)

# Load the 102 Flowers dataset (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
# We have the train_images and the test_images folders, each with 102 subfolders,
# corresponding to the different 102 classes of flowers (0 to 101)

# Find which format to use (depends on the backend), and compute input_shape
from keras import backend as K
# dataset images resolution
img_rows, img_cols, channels = 256, 256, 3        # flower images are 256x256px and RGB (3 channels)

train_dir = 'train_images'
test_dir = 'test_images'
num_train_samples = 7414
num_test_samples = 775
epochs = 30         # 1 epoch: processar tot el dataset forward i backward
batch_size = 64      # quantes imatges (training example) tracto alhora en cada batch (de cop)

if K.image_data_format() == 'channels_first':
    input_shape = (channels, img_rows, img_cols)
    chanDim = 1
else:
    input_shape = (img_rows, img_cols, channels)
    chanDim = -1

# Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# Two hidden layers
nn = Sequential()

# ppi de l'exemple: https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
#nn.add(Conv2D(32, (3, 3), input_shape=input_shape))
#nn.add(Activation('relu'))
#nn.add(MaxPooling2D(pool_size=(2, 2)))

nn.add(Conv2D(128, (7, 7), strides=(2, 2), padding='same', activation='relu', input_shape=(256, 256, 3)))    # update pq. surt warning python
nn.add(BatchNormalization(axis=chanDim))
#nn.add(MaxPooling2D(pool_size=(2, 2)))
#nn.add(Dropout(0.5))

nn.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(256, 256, 3)))
nn.add(BatchNormalization(axis=chanDim))
nn.add(MaxPooling2D(pool_size=(2, 2)))
nn.add(Dropout(0.5))

nn.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(256, 256, 3)))
nn.add(BatchNormalization(axis=chanDim))
nn.add(MaxPooling2D(pool_size=(2, 2)))
#nn.add(Dropout(0.7))

nn.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(256, 256, 3)))
nn.add(BatchNormalization(axis=chanDim))
nn.add(MaxPooling2D(pool_size=(2, 2)))
#nn.add(Dropout(0.5))

nn.add(Flatten())

#nn.add(Dense(2048, activation='relu'))
#nn.add(BatchNormalization(axis=chanDim))
#nn.add(Dropout(0.5))

#nn.add(Dense(1024, activation='relu'))
#nn.add(BatchNormalization(axis=chanDim))
#nn.add(Dropout(0.5))

nn.add(Dense(512, activation='relu'))
nn.add(BatchNormalization(axis=chanDim))
nn.add(Dropout(0.5))

#nn.add(Dense(256, activation='relu'))
#nn.add(BatchNormalization(axis=chanDim))
#nn.add(Dropout(0.5))

nn.add(Dense(102, activation='softmax'))
nn.summary()

# Model visualization
# We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
#from keras.utils import plot_model
#plot_model(nn, to_file='nn_conf40.png', show_shapes=True)

# Compile the NN
#nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

# Tractament de les imatges en batches per no petar la memòria
# copiat de https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# classes: Optional list of class subdirectories (e.g. ['dogs', 'cats']).
# Default: None. If not provided, the list of classes will be automatically inferred
# from the subdirectory names/structure under directory, where each subdirectory
# will be treated as a different class (and the order of the classes, which
# will map to the label indices, will be alphanumeric).
# The dictionary containing the mapping from class names to class indices
# can be obtained via the attribute class_indices.

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_rows, img_cols),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical')

history = nn.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    epochs=epochs,
    # validation_data hauria de ser amb dades de validació, que no tinc
    # poso les de test i llavors ja no cal que faci l'evaluate_generator amb test més avall.
    # evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
    validation_data=test_generator,
    validation_steps=num_test_samples // batch_size)


'''
# * evaluate_generator uses both your test input and output.
# It first predicts output using training input and then evaluates performance
# by comparing it against your test output. So it gives out a measure of performance,
# i.e. accuracy in your case.

# * predict_generator takes your test data and gives you the output.

# Evaluates the model on a data generator
# Returns: Scalar test loss (if the model has a single output and no metrics)
# or list of scalars (if the model has multiple outputs and/or metrics).
# The attribute model.metrics_names will give you the display labels for the scalar outputs.
'''
# Evaluate the model with test set
# score[0]: loss; score[1]: accuracy
#score = nn.evaluate_generator(generator=test_generator, steps=len(test_generator))
#print('Test loss: ', score[0])
#print('Test accuracy: ', score[1])

#y_test = test_generator.class_indices      # això em treu el diccionari de class_001: 1 ...
y_test = test_generator.classes             # això hauria de treure les true labels de test

# Print parameters used
print('\ntraining folder: ', train_dir)
print('test folder: ', test_dir)
print('num train samples: ', num_train_samples)
print('num test samples: ', num_test_samples)
print('image size: ', img_rows)
print('channels: ', channels)
print('epochs: ', epochs)
print('batch size: ', batch_size)
print('\nValidation loss (with test data): ', history.history['loss'])
print('Validation accuracy (with test data): ', history.history['acc'])
print('\nTest labels: ', y_test)


###############
# Start training
#history = nn.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.15)

# Evaluate the model with test set
#score = nn.evaluate(x_test, y_test, verbose=0)
#print('test loss:', score[0])
#print('test accuracy:', score[1])

# Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('flowers102_cnn_accuracy_conf40.pdf')
plt.close()
# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('flowers102_cnn_loss_conf40.pdf')
plt.close()


# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# Compute probabilities
test_generator.reset()
# Generates predictions for the input samples from a data generator.
# * predict_generator takes your test data and gives you the output.
# Returns: Numpy array(s) of predictions.
Y_pred = nn.predict_generator(test_generator, verbose=1, steps=len(test_generator))
# Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)

print('\nPredicted labels (from test data):', y_pred)


# Plot statistics
print('\nAnalysis of results')
# All flower classes (102; de 0 a 101)
target_names = ['class_000','class_001','class_002','class_003','class_004','class_005','class_006','class_007','class_008','class_009','class_010','class_011','class_012','class_013','class_014','class_015','class_016','class_017','class_018','class_019','class_020','class_021','class_022','class_023','class_024','class_025','class_026','class_027','class_028','class_029','class_030','class_031','class_032','class_033','class_034','class_035','class_036','class_037','class_038','class_039','class_040','class_041','class_042','class_043','class_044','class_045','class_046','class_047','class_048','class_049','class_050','class_051','class_052','class_053','class_054','class_055','class_056','class_057','class_058','class_059','class_060','class_061','class_062','class_063','class_064','class_065','class_066','class_067','class_068','class_069','class_070','class_071','class_072','class_073','class_074','class_075','class_076','class_077','class_078','class_079','class_080','class_081','class_082','class_083','class_084','class_085','class_086','class_087','class_088','class_089','class_090','class_091','class_092','class_093','class_094','class_095','class_096','class_097','class_098','class_099','class_100','class_101']
#print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names))
print(classification_report(y_test, y_pred, target_names=target_names))
#print(confusion_matrix(np.argmax(y_test, axis=1), y_pred))

#print(confusion_matrix(y_test, y_pred))
confmat = confusion_matrix(y_test, y_pred)
print(confmat)


# Pintar confusion matrix amb les 102 classes
ticks=np.linspace(0, 101,num=102)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.legend('Confusion Matrix', loc='upper left')
#plt.show()
plt.savefig('flowers102_cnn_confmat_conf40.pdf')


# Saving model and weights
from keras.models import model_from_json
nn_json = nn.to_json()
with open('nn.json', 'w') as json_file:
    json_file.write(nn_json)
weights_file = "weights-FLOWERS102_" + str(history.history['acc'][-1]) + "_conf40.hdf5"
nn.save_weights(weights_file, overwrite=True)

print('history.history:')
print(history.history)

######## Plots -- moguda al final
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('flowers102_cnn_loss_acc_conf40.pdf')
###


'''
# Loading model and weights
json_file = open('nn.json', 'r')
nn_json = json_file.read()
json_file.close()
nn = model_from_json(nn_json)
nn.load_weights(weights_file)

#nn.save_weights('try1.h5')
'''


# seaborn heatmap --> confusion matrix
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data, annot=True, fmt="d")