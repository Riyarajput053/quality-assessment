import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("ResNet50.h5")
print(model.output_shape)
test_input = np.random.rand(1, 224, 224, 3)  # Dummy image
prediction = model.predict(test_input)
print("Raw Model Output:", prediction)