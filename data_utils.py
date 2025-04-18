import numpy as np
import cv2

IMG_SIZE = 64
SEQ_LENGTH = 10

def preprocess_uploaded_images(uploaded_files):
    if len(uploaded_files) != SEQ_LENGTH:
        return None, "Please upload exactly 10 images."

    sequence = []
    for file in uploaded_files:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, f"Could not decode: {file.name}"
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        sequence.append(img)

    sequence = np.array(sequence).reshape(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 1)
    return sequence, None
