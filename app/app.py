from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image as keras_image

app = Flask(__name__)

# Load the model
model = load_model('C:/Users/vtu24/Downloads/Blood-Cancer-Detection-CNN-master/Blood-Cancer-Detection-CNN-master/mymodel.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    try:
        img = Image.open(file)
        img = img.resize((150, 150))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        predictions = model.predict(images, batch_size=10)
        predicted_class = np.argmax(predictions)
        result = "Cancer" if predicted_class == 1 else "Normal"
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', message=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
