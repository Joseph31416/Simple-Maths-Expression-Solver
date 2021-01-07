from flask import Flask, render_template, request, send_from_directory, redirect
from build_model import build_model
from prediction import get_predictions
import numpy as np
import os
import glob
import cv2
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def root():
    files = glob.glob('./uploads/*')
    for f in files:
        os.remove(f)
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    img = request.files['img']
    if img.filename != '':
        img.save(f"./uploads/{img.filename}")
    else:
        return redirect('/')
    fname = img.filename
    img = cv2.imread(f"./uploads/{img.filename}", 0)
    def resize_image(img):
        """
        Converts uploaded image to a (1, 200, 50, 1) numpy array, with 1 or 0 entries
        """
        for a in range(img.shape[0]):
            for b in range(img.shape[1]):
                if img[a][b] > 220:
                    img[a][b] = 255
                else:
                    img[a][b] = 0
        img_neg = cv2.bitwise_not(img)
        coords = cv2.findNonZero(img_neg)
        x, y, w, h = cv2.boundingRect(coords)
        out = img[y:y + h, x:x + w]
        out = cv2.resize(out, (200, 50), interpolation=cv2.INTER_CUBIC)
        out = out.transpose([1, 0])
        out = out[:, :, np.newaxis]
        return np.array([out]) / 255
    image = resize_image(img)
    model = build_model(200, 50)
    model.load_weights("final.h5")
    pred = get_predictions(model, image)
    output = {'pred': pred, 'fname': fname}
    del img, image, model, pred, fname
    return render_template("prediction.html", output=output)


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory('uploads', filename)


app.run()
#set host to "0.0.0.0" for remote connections
#app.run(host="0.0.0.0")