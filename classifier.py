import os
import numpy as np
import flask
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Flatten

import rnnmodel as rnn

app = Flask(__name__)

max_time_steps = 5
num_features = 11

all_chars = '0123456789+'

char_to_index = dict((c, i) for i, c in enumerate(all_chars))
index_to_char = dict((i, c) for i, c in enumerate(all_chars))


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

@app.route('/')
@app.route('/index', methods=['POST','GET'])
def index():
	return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
	rnn.generate_data()

	output = loaded_model.predict(rnn.pred)
	result = {
		'Input: ': rnn.devectorize_example(rnn.a),
		'Output: ': rnn.devectorize_example(rnn.b),
		'Prediction: ': output
	}

	return render_template('result.html',result=result)

if __name__ == "__main__" :
 	app.run(debug=True)
 	

 
