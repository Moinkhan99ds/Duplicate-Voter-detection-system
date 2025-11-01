from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__, template_folder="templates")

model = load_model(os.path.join(os.getcwd(), "model.h5"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img = Image.open(img).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        result = "Duplicate Voter"
    else:
        result = "New Voter"

    return result

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
