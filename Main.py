import math
import os
import random
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plot
import numpy as np
from PIL import Image, ImageDraw
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import load_model
from tqdm import tqdm

# Dynamic Variables
TRAIN_PATH = 'train/'
GENERATION_PATH = 'generate/'
OUTPUT_PATH = "output/"
LOG_PATH = "log/"
BASE_WIDTH = 32
MODEL_SAVE_AS = "Autoencoder"
CHECKPOINTS_SAVE_AS = "Checkpoint"

EPOCHS = 200
BATCH_SIZE = 5

# Static Variables
RGB_MIN = 0  # The lowest values of a RGB tuple (i.e. (0, 0, 0) which is black)
RGB_MAX = 255  # The highest values of a RGB tuple (i.e. (255, 255, 255) which is white)


# Functions
def normalize_list(your_list, minimum, maximum):
    def normalize(number, mini, maxi):
        return float((number - mini)) / float((maxi - mini))

    for list_value in your_list:
        your_list[your_list.index(list_value)] = normalize(list_value, minimum, maximum)

    return your_list


def un_normalize_list(your_list, minimum, maximum):
    def un_normalize(number, mini, maxi):
        return int(number * (maxi - mini) + mini)

    for list_value in your_list:
        your_list[your_list.index(list_value)] = un_normalize(list_value, minimum, maximum)

    return your_list


# GET IMAGES
trainingImages = next(os.walk(TRAIN_PATH))[2]
generationImages = next(os.walk(GENERATION_PATH))[2]

if ".DS_Store" in trainingImages or ".DS_Store" in generationImages:
    try:
        generationImages.remove(".DS_Store")
    except ValueError:
        pass

    try:
        trainingImages.remove(".DS_Store")
    except ValueError:
        pass

# RESIZE IMAGES
for _, id_ in tqdm(enumerate(trainingImages), total=len(trainingImages),
                   desc="Loading training files"):  # Update the size of the images
    path = TRAIN_PATH + id_
    currentImg = Image.open(path)

    wPercent = (BASE_WIDTH / float(currentImg.size[0]))
    hSize = int((float(currentImg.size[1]) * float(wPercent)))
    currentImg = currentImg.resize((BASE_WIDTH, hSize), Image.ANTIALIAS)
    currentImg.save(path)

for _, id_ in tqdm(enumerate(generationImages), total=len(generationImages),
                   desc="Loading testing files"):  # Update the size of the images
    path = GENERATION_PATH + id_
    currentImg = Image.open(path)

    wPercent = (BASE_WIDTH / float(currentImg.size[0]))
    hSize = int((float(currentImg.size[1]) * float(wPercent)))
    currentImg = currentImg.resize((BASE_WIDTH, hSize), Image.ANTIALIAS)
    currentImg.save(path)

# Get random image's size to be used as the standard
standardImage = Image.open(TRAIN_PATH + trainingImages[random.randint(0, len(trainingImages) - 1)])
STANDARD_WIDTH = standardImage.size[0]  # A global constant
STANDARD_HEIGHT = standardImage.size[1]  # A global constant

# PREPROCESSING
# X_train
X_train = []
for _, image in tqdm(enumerate(trainingImages), total=len(trainingImages), desc="Vectoring x_train files"):
    imageData = []
    currentImage = Image.open(TRAIN_PATH + image)
    currentImage = currentImage.convert("RGB")  # Just in case it is in RGBA format

    '''
    Imagine an image with size W x H.
    
    Let W  = 3 and H = 4,
    Thus the image will be expressed in a list as follows:
    
    [
    [[R, G, B], [R, G, B], [R, G, B]],  <--- Three (3) pixels, each with RGB values.
    [[R, G, B], [R, G, B], [R, G, B]],
    [[R, G, B], [R, G, B], [R, G, B]],
    [[R, G, B], [R, G, B], [R, G, B]]   <--- Repeated this process "H" times (4)
    ]   <--- End of one (1) images
    
    '''

    for heightPixel in range(0, STANDARD_HEIGHT):
        currentHeightPixels = []  # Store the pixels with the same height here
        for widthPixels in range(0, STANDARD_WIDTH):
            RGB = currentImage.getpixel((widthPixels, heightPixel))
            newRGB = normalize_list([float(RGB[0]), float(RGB[1]), float(RGB[2])], RGB_MIN, RGB_MAX)

            currentHeightPixels.append(newRGB)
        imageData.append(currentHeightPixels)

    X_train.append(imageData)

# X_test
X_test = []
for _, image in tqdm(enumerate(generationImages), total=len(generationImages), desc="Vectoring x_test files"):
    imageData = []
    currentImage = Image.open(GENERATION_PATH + image)
    currentImage = currentImage.convert("RGB")  # Just in case it is in RGBA format

    for heightPixel in range(0, STANDARD_HEIGHT):
        currentHeightPixels = []  # Store the pixels with the same height here
        for widthPixels in range(0, STANDARD_WIDTH):
            RGB = currentImage.getpixel((widthPixels, heightPixel))  # The original tuple
            newRGB = normalize_list([float(RGB[0]), float(RGB[1]), float(RGB[2])], RGB_MIN,
                                    RGB_MAX)  # Normalize that tuple

            currentHeightPixels.append(newRGB)
        imageData.append(currentHeightPixels)

    X_test.append(imageData)

# Convert x_train and x_test to np.array
X_train = np.array(X_train).reshape((len(X_train), STANDARD_HEIGHT, STANDARD_WIDTH, 3))
X_test = np.array(X_test).reshape((len(X_test), STANDARD_HEIGHT, STANDARD_WIDTH, 3))

# GENERATE AUTO ENCODER
if os.path.isfile(MODEL_SAVE_AS + ".h5"):
    print "Previous autoencoder found. Loading it."
    autoEncoder = load_model(MODEL_SAVE_AS + ".h5")

else:
    if not os.path.isdir(LOG_PATH):  # Check if there exists a log directory
        os.makedirs(LOG_PATH)

    inputImg = Input(
        shape=(STANDARD_HEIGHT, STANDARD_WIDTH,
               3))  # Format: (no. of images, STANDARD HEIGHT, STANDARD WIDTH, NUMBER OF RGB CHANNELS)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputImg)  # 16 layers (channels) of 3x3 images (kernels)
    x = MaxPooling2D((2, 2), padding='same')(x)  # Resize images to 2x2 size with 16 layers
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8 layers of 3x3 images
    x = MaxPooling2D((2, 2), padding='same')(x)  # Resize images to 2x2 size with 8 layers
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8 layers of 3x3 images
    encoded = MaxPooling2D((2, 2), padding='same')(x)  # Resize images to 2x2 size with 8 layers

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)  # 8 layers of 3x3 images
    x = UpSampling2D((2, 2))(x)  # Resize images to 2x2 size with 8 layers
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)  # 8 layers of 3x3 images
    x = UpSampling2D((2, 2))(x)  # Resize images to 2x2 size with 8 layers
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)  # 16 layers of 3x3 images
    x = UpSampling2D((2, 2))(x)  # Resize images to 2x2 size with 16 layers
    decoded = Conv2D(3, (STANDARD_HEIGHT, STANDARD_WIDTH), activation='sigmoid', padding='same')(x)

    autoEncoder = Model(inputImg, decoded)

    print "\nRun tensorboard to see the current statistics of the autoencoder!\n"

    # Callbacks
    tensorBoard = TensorBoard(log_dir=LOG_PATH)
    checkpointMaker = ModelCheckpoint(monitor="loss", filepath=CHECKPOINTS_SAVE_AS + ".hdf5",
                                      verbose=1, save_best_only=True)
    reduceLearningRate = ReduceLROnPlateau(monitor="loss", patience=2, verbose=1,
                                           factor=0.9)
    stopEarly = EarlyStopping(monitor="loss", patience=10, verbose=1)

    # Compile
    autoEncoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=["acc"])

    # Train
    startTime = time.time()  # Start stopwatch

    predict = autoEncoder.fit(X_train, X_train,
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              validation_data=(X_test, X_test),
                              callbacks=[tensorBoard, checkpointMaker, reduceLearningRate, stopEarly])
    endTime = time.time()  # Stop stopwatch

    print "The program took", str(int(math.floor((endTime - startTime) / 60))), "minutes and", str(int(
        round((endTime - startTime) - (math.floor((endTime - startTime) / 60) * 60),
              0))), "seconds to complete the training on", str(
        len(X_train)), "images in", str(EPOCHS), "epochs."

    autoEncoder.save(MODEL_SAVE_AS + ".h5")

    # Display graph
    loss = predict.history["loss"]
    accuracy = predict.history["acc"]
    valLoss = predict.history["val_loss"]
    valAccuracy = predict.history["val_acc"]

    plot.plot(range(0, len(loss)), loss, label="Training Loss", color="green")
    plot.plot(range(0, len(accuracy)), accuracy, label="Training Accuracy", color="blue")
    plot.plot(range(0, len(valLoss)), valLoss, label="Validation Loss", color="red")
    plot.plot(range(0, len(valAccuracy)), valAccuracy, label="Validation Accuracy", color="orange")

    plot.xlabel('Epoch')
    plot.ylabel('Percentage')
    plot.title('Autoencoder Results')

    plot.legend(loc="best")

    plot.show()

# GENERATE OUTPUT FOLDER
'''
The generated outputs will be saved in the following manner:
./{OUTPUT_PATH}/{EPOCHS} epochs/{TIMESTAMP}/
'''
if not os.path.isdir(OUTPUT_PATH):  # If it does not exist
    os.makedirs(OUTPUT_PATH)

if not os.path.isdir(OUTPUT_PATH + str(EPOCHS) + " epochs/"):
    os.makedirs(OUTPUT_PATH + str(EPOCHS) + " epochs/")

OUTPUT_PATH = OUTPUT_PATH + str(EPOCHS) + " epochs/" + str(datetime.now().strftime('%d-%m-%Y %H%M')) + "/"

os.makedirs(OUTPUT_PATH + "/Images/")  # For the images
os.makedirs(OUTPUT_PATH + "/Model/")  # For the model
# PREDICTION  & UN-NORMALIZATION
if os.path.isfile(CHECKPOINTS_SAVE_AS + ".hdf5"):
    autoEncoder.load_weights(CHECKPOINTS_SAVE_AS + ".hdf5")

Prediction = autoEncoder.predict(X_test)  # This is outputted as a np.array
Prediction = Prediction.tolist()  # Convert np.array to list

baseImages = []
for _, image in tqdm(enumerate(Prediction), total=len(Prediction), desc="Formatting predictions"):
    newHeightPixel = []

    for heightPixels in image:
        newWidthPixel = []

        for widthPixels in heightPixels:
            newWidthPixel.append(un_normalize_list(widthPixels, RGB_MIN, RGB_MAX))  # Undo the normalization

        newHeightPixel.append(newWidthPixel)

    baseImages.append(newHeightPixel)

# PREDICTION FORMATTING & OUTPUT
imageCount = 0
for _, image in tqdm(enumerate(baseImages), total=len(baseImages), desc="Generating images"):
    predictionOutput = Image.new("RGB", (STANDARD_WIDTH, STANDARD_HEIGHT),
                                 (255, 255, 255))
    predictionLayer = ImageDraw.Draw(predictionOutput)

    height = 0
    for heightLayer in image:
        width = 0

        for widthLayer in heightLayer:
            predictionLayer.point((width, height), fill=(widthLayer[0], widthLayer[1], widthLayer[2]))

            width += 1

        height += 1

    del predictionLayer

    predictionOutput.save(OUTPUT_PATH + "/Images/output" + str(imageCount) + ".png")
    imageCount += 1

# RELOCATE MODEL AND CHECKPOINT FILES
shutil.move(str(MODEL_SAVE_AS + ".h5"), OUTPUT_PATH + "/Model/")
shutil.move(str(CHECKPOINTS_SAVE_AS + ".hdf5"), OUTPUT_PATH + "/Model/")

# REMOVE LOGS
map(os.unlink, (os.path.join(LOG_PATH, f) for f in os.listdir(LOG_PATH)))
