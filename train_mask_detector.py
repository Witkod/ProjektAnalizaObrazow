# USAGE
# python train_mask_detector.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from imutils import paths

import numpy as np
import argparse
import os



####################
#                  #
# Arguments Parser #
#                  #
####################



# prepare the parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",
	default="dataset",
	help="(direcotry path) to input dataset")
ap.add_argument("-m", "--model",
	default="mask_detector.model",
	help="(file path) to output mask detector model")
args = vars(ap.parse_args())

# initial learning rate
INIT_LR = 1e-4
# number of epochs to train for
EPOCHS = 5
# size of batch
# this should probably be changed depending on GPU / CPU calculations
# and available memory (not tested)
BatchSize = 8



#######################
#                     #
# Load & prepare data #
#                     #
#######################



print("-I-: loading images...")
# list of images in dataset directory
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the images in imagePaths
for imagePath in imagePaths:
	# get the subdirectory name of image set (with_mask / without_mask)
	# [-2] is 2nd element counting from end
	label = imagePath.split(os.path.sep)[-2]

	# load the input image
	# (224x224) is widely used in deep learning for some reason
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	# preprocess it using MobileNetV2 preprocessor
	image = preprocess_input(image)

	# update the data and labels lists
	data.append(image)
	labels.append(label)

# convert the data to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
# one-hot is (001000000) - only one 1
# this clacifies images as value and not some "label"
# very important for deep learning as now we can get a one value as output
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# set categories from labes (its one-hot encoded already)
labels = to_categorical(labels)

# split the data in 2 sets
# training - 75%
# testing - 25%
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)

# data augmentation - modify images slightly for much larger data pool
# rotate / zoom / shear / scale / flip images to generate more data
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")



#########################
#                       #
# CREATE NEURAL NETWORK #
#                       #
#########################



# using MobileNetV2
# using FC (Fully Connected) layers
# https://developer.ridgerun.com/wiki/index.php?title=Keras_with_MobilenetV2_for_Deep_Learning
# https://keras.io/api/applications/mobilenet/

# include_top
# whether to include the fully-connected layer at the top of the network.
# we don't want that because we create our model layers later

# input_tensor
# we use 224px x 224px images in RGB so we input shape (224,224,3)
baseModel = MobileNetV2(include_top=False, weights="imagenet",
						input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output

# change this parameter to determine how big details are taken into account
# affetcts accuracy (to some extent)
# greater value - slightly harder to detect mask (less torelable)
# lower value - mask will be detected on small (distant) faces even without mask
# (7,7) is max
# for example (1,1) wil be almost 0-1 detections (~100% mask or ~100% no mask)
# and even tiniest element covering mouth/nose will trigger mask to be detected
poolSizeTuple = (5,5)
# convolutional and pooling layer
headModel = AveragePooling2D(pool_size=poolSizeTuple)(headModel)
headModel = Flatten(name="flatten")(headModel)

# relu / softmax
# https://keras.io/api/layers/activations/

# change this parameter to determine how many details are taken into account
# affetcts accuracy (to some extent)
# greater value - slightly harder to detect mask (less torelable)
ReluLayerSize = 128
headModel = Dense(ReluLayerSize, activation="relu")(headModel)

# we have 2 classes - with_mask / without_mask so we have output space of size 2
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

print("-I-: compiling model...")
# optimize model
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# compile model
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])



########################
#                      #
# TRAIN NEURAL NETWORK #
#                      #
########################



# train network using images pool and with augmented versions
print("-I-: training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BatchSize),
	steps_per_epoch=len(trainX) // BatchSize,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BatchSize,
	epochs=EPOCHS)



#######################
#                     #
# TEST NEURAL NETWORK #
#                     #
#######################



# check model on test set
print("-I-: evaluating network...")
predIdxs = model.predict(testX, batch_size=BatchSize)

# get label of detection with larger prediction
predIdxs = np.argmax(predIdxs, axis=1)

# show classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# save model
print("-I-: saving mask detector model...")
model.save(args["model"], save_format="h5")
