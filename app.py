from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

model = load_model("model.h5")  # Your trained model file

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
