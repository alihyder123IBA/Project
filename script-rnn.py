import os
import numpy as np
import flask
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import model_from_json


app = Flask(__name__)


# load json and create model
json_file = open('modelRNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("modelRNN.h5")
print("Loaded model from disk")


@app.route('/')
@app.route('/index', methods=['GET'])
def index():
	return render_template('index.html')


def predictor(predict_list):
	to_predict = np.array(predict_list).reshape(1,1,2)
	result = loaded_model.predict(to_predict)
	return result[0]

@app.route('/result', methods=['POST'])
def result():
	if request.method == 'POST':
		predict_list = request.form.to_dict()
		predict_list = list(predict_list.values())
		predict_list = list(map(int,predict_list))
		result = predictor(predict_list)

		predictions = "Predicted: " + str(result)

		return flask.render_template('result.html', prediction=predictions)

if __name__ == "__main__":
	app.run(debug=True)