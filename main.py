from flask import Flask, render_template, request, url_for
from build_model import build_model
from prediction import get_predictions
import numpy as np
import cv2
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    img = request.files['img']
    if img.filename != '':
        img.save(img.filename)
    img = cv2.imread(f"{img.filename}", cv2.IMREAD_GRAYSCALE)
    def get_uploaded_image(img):
        """
        Converts uploaded image to a (1, 200, 50, 1) numpy array, with 1 or 0 entries
        """
        return np.load("TestImages.npy")[10:11] / 255

    image = get_uploaded_image(img)
    model = build_model(200, 50)
    model.load_weights("final.h5")
    pred = get_predictions(model, image)
    return render_template("prediction.html", pred=pred)

# start the flask app, allow remote connections
app.run()
# app.run(host='0.0.0.0')