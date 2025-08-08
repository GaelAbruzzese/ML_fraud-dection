import joblib
import pandas as pd
import pytz
from flask import Flask, flash, Response, url_for, send_from_directory
from flask import request, render_template, redirect
from flask_cors import CORS
from werkzeug.utils import secure_filename
from io import StringIO
import os

from FraudDetectionModel.pipe import predict_result

app = Flask(__name__)
app.secret_key = 'secret_key'

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PATH_TO_MODEL = os.path.join(THIS_FOLDER, "model/my_pipeline.joblib")
UPLOAD_FOLDER = os.path.join(THIS_FOLDER, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['txt', 'csv'])

pipe = joblib.load(PATH_TO_MODEL)

def run_model(input):
    '''predictor function'''
    result = predict_result(input, pipe)
    return result

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return 'your app is running!'

@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if container is healthy
    """
    predictions = pd.read_csv(os.path.join(THIS_FOLDER,'FraudDetectionModel/data/data_to_predict.csv'))
    try:
        result = run_model(predictions)
        return Response(response='{"status": "ok"}', status=200, mimetype='application/json')
    except:
        return Response(response='{"status": "error"}', status=500, mimetype='application/json')

@app.route('/model_details')
def model():
    with open(os.path.join(THIS_FOLDER,'model/model_details.txt'), "r") as f:
        content = f.read()
    return Response(content, mimetype='text/plain')

@app.route('/display_result')
def result():
    filename = os.path.join(THIS_FOLDER,'uploads/data_to_predict.csv')
    data = pd.read_csv(filename)
    pred = run_model(data)
    values = list(pred.values)
    return render_template('display_result.html', values=values)


@app.route('/upload')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('file successfully uploaded')
            return redirect('/upload')
        else:
            flash('file type not allowed')
            return redirect(request.url)

#@app.route('/upload/<filename>')
#def uploaded_file(filename):
#    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

'''
@app.route('/invocations', methods=["GET","POST"])
def predict():

    if request.method=='GET':
        return('<form action="/test" method="post"><input type="submit" value="Send" /></form>')

    elif request.method=='POST':
        if flask.request.content_type == 'text/csv':
            X_train = flask.request.data.decode('utf-8')
            X_train = pd.read_csv(StringIO(X_train))
        else:
            return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

        results = run_model(X_train)
        results_str = ",\n".join(results.astype('str'))
        return Response(response=results_str, status=200, mimetype='text/csv')
    else:
        return ('ok')

'''


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

'''


class UploadCSV(Resource):

    def post(self):
        file = request.files['file']
        data = pd.read_csv(file)
        print(data)

        '''
