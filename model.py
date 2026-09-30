# Keras
from tensorflow.keras.applications.imagenet_utils import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19

# Load VGG19 with ImageNet pretrained weights
model = VGG19(weights="imagenet")

# Save the model
model.save("vgg19.keras")

print("VGG19 model saved successfully as vgg19.keras")