import numpy as np
from tensorflow.keras.models import load_model

CLASS_NAMES = ['A', 'D', 'M', 'N', 'O', 'S', 'T', 'U', 'V', 'Y', 'blank']

def load_trained_model(model_path="sign_language_model.h5"):
    return load_model(model_path)

def predict_sequence(model, input_sequence):
    input_sequence = np.expand_dims(input_sequence, axis=0)
    prediction = model.predict(input_sequence)
    return prediction, CLASS_NAMES[np.argmax(prediction)]
