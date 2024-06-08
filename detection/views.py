from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from .forms import LogoUploadForm
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.DEBUG)

# Create your views here.

def index(request):
    form = LogoUploadForm()
    return render(request, 'detection/index.html', {'form': form})

def verify_model():
    # Load and compile the model within the request context
    model = load_model('fake_logo_detector.h5')
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Load a known test image
    img = image.load_img('path_to_known_test_image.jpg', target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0

    # Make prediction using the loaded and compiled model
    prediction = model.predict(img_tensor)
    logging.debug(f"Verification Prediction: {prediction}")
    result = 'Fake' if prediction < 0.5 else 'Real'
    logging.debug(f"Verification Result: {result}")

def predict(request):
    if request.method == 'POST':
        form = LogoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['logo']
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_url = fs.url(filename)

            img = image.load_img(fs.path(filename), target_size=(150, 150))
            img_tensor = image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.0

            logging.debug(f"Image Tensor Shape: {img_tensor.shape}")

            model = load_model('fake_logo_detector.h5')

            model.compile(optimizer=Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

            prediction = model.predict(img_tensor)
            logging.debug(f"Prediction: {prediction}")
            result = 'Fake' if prediction < 0.5 else 'Real'
            logging.debug(f"Result: {result}")

            return render(request, 'detection/result.html', {'result': result, 'file_url': file_url})
    return render(request, 'detection/index.html', {'form': form})
#
verify_model()