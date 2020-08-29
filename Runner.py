import os

from flask import Flask, render_template, request, jsonify, json
from werkzeug.utils import secure_filename
from Classifier import train, prediction

app = Flask(__name__)


# This is the train model

@app.route('/dashboard',methods=['GET'])
def dashboard():
    return render_template("dashboard.html")

@app.route('/setup',methods=['GET'])
def setup():
    return render_template("setup.html")

@app.route('/index',methods=['GET'])
def main():
    return render_template("home.html")

@app.route('/', methods=['GET'])
def index():
    return render_template("home.html")

@app.route('/home', methods=['GET'])
def home():
    return render_template("base.html")


@app.route('/train', methods=['POST'])
def train_runner():
    # epoch and bach size
    call = ""
    try:
        train()
        call = "Successfull"
        return render_template('./base.html',call=call)

    except Exception as e:
        call = "Fail"
        print(e)
        return render_template('./base.html', call=e)
    #return redirect("home")


# this is for the model
@app.route('/process', methods=['POST'])
def train_prediction():
    print(request.files['file'])
    try:
        if request.method=="POST" and request.files['file']:
            f = request.files['file']
            filename = secure_filename(f.filename)
            print(filename)
            path = os.path.join(os.path.abspath(""), os.path.join("UPLOAD_FOLDER",filename))
            f.save(path)
            pred = prediction(path)
            cls = sorted(os.listdir('v_data/train'))
            result = []
            i = 1
            for name, percent in zip(cls, pred[0]):
                # print("trying...", i)
                result.append('["' + name + '",' + str(percent * 100) + ']')
                i=i+1
            # print(result)
        print("Result",result)
        return render_template('./home.html', result=result)
    except Exception as e:
        return str(e)


# This is the main method , in my view
if __name__ == '__main__':
    app.run(host='0.0.0.0')

