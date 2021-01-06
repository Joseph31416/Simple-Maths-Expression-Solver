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

    def resize_image(img):
        """
        Converts uploaded image to a (1, 200, 50, 1) numpy array, with 1 or 0 entries
        """
        img = cv2.resize(img, (200, 50))
        img = img.transpose([1, 0])
        for a in range(200):
            for b in range(50):
                if img[a][b] < 225:
                    img[a][b] = 0
                else:
                    img[a][b] = 1
        img = img[:, :, np.newaxis]
        img = np.array([img])
        return img

    image = resize_image(img)
    model = build_model(200, 50)
    model.load_weights("final.h5")
    pred = get_predictions(model, image)
    return render_template("prediction.html", pred=pred)

# start the flask app, allow remote connections
app.run()
# app.run(host='0.0.0.0')