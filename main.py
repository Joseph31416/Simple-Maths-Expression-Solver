from flask import Flask, render_template, send_from_directory
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    fname = 'sample_response.jpg'
    pred = {'correct': '1, 2, 3, 5, 6, 9, 10, 13, 14',
            'incorrect': '4, 7, 8, 11, 12, 15',
            'score': '9 out of 15'}
    output = {'pred': pred, 'fname': fname}
    return render_template("prediction.html", output=output)


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory('uploads', filename)

# start the flask app, allow remote connections
app.run()