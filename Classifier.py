# Importing all necessary libraries
import keras
# import tensorflow
from keras import backend as K

from DataGenerator import Generator
from Model_CNN import Model_SAPI

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np


train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/test'
nb_train_samples = 450
nb_validation_samples = 225
epochs = 4000
batch_size = 100

img_width = 128
img_height = 128

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(128, 128))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


# this is instantiation of the differnt classes
def train():
    model = Model_SAPI(input_shape)

    model = model.squentialModel()

    generator = Generator(model, img_width, img_height, batch_size)
    generatormodel = generator.generator_path(train_data_dir, validation_data_dir, nb_train_samples, epochs,nb_validation_samples)

    generatormodel.save("predictor.h5")


# # load model
def prediction(file_path):
    keras.backend.clear_session()
    model = load_model("predictor.h5")
    new_image = load_image(file_path)
    # check prediction
    pred = model.predict(new_image)
    print("pred",pred)
    return pred









