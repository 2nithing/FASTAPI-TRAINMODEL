from tensorflow.keras import layers,Sequential,callbacks
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Input
import numpy as np

def load_model():
    feature_extractor_layer = MobileNet(weights="imagenet", include_top=False, input_tensor=Input(shape=(160,160,3)))
    return feature_extractor_layer

def train(data,labels):
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    print(data.shape)

    feature_extractor_layer = load_model()
    feature_extractor_layer.trainable = False
    input_tensor=Input(shape=(160,160,3))
    x = feature_extractor_layer(input_tensor)
    x = Flatten()(x)
    x = Dense(16, activation='relu', name='hidden_layer')(x)
    x = Dropout(.5)(x)
    output = Dense(2, activation='softmax', name='output')(x)
    
    model = Model(inputs=input_tensor,outputs=output)
    model.compile(optimizer=Adam(),loss="categorical_crossentropy",metrics=["accuracy"],run_eagerly=True)


    aug = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
    labels = to_categorical(labels)

    model.fit(aug.flow(data, labels),epochs=50)
    return model
