import os
import numpy as np
import flask
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import model_from_json


max_time_steps = 5
num_features = 11

all_chars = '0123456789+'



char_to_index = dict((c, i) for i, c in enumerate(all_chars))
index_to_char = dict((i, c) for i, c in enumerate(all_chars))

app = Flask(__name__)


def generate_data():
	example= ''
	label = ''
	with app.test_request_context():
		#predict_list = request.form.to_dict()
		#predict_list = list(predict_list.values())
		#predict_list = list(map(int,predict_list))
		if request.method == 'POST':
			first_num = request.form["num_one"]
			second_num = request.form["num_two"]
			example = ((first_num) + '+' + (second_num))
			label =  (first_num+second_num)

	return example,label



def vectorize_example(example, label):
	
	x = np.zeros((max_time_steps, num_features))
	y = np.zeros((max_time_steps, num_features))
	
	diff_x = max_time_steps - len(example)
	diff_y = max_time_steps - len(label)
	
	for i, c in enumerate(example):
		x[diff_x + i, char_to_index[c]] = 1

	for i in range(diff_x):
		x[i, char_to_index['0']] = 1

	for i, c in enumerate(label):
		y[diff_y + i, char_to_index[c]] = 1

	for i in range(diff_y):
		y[i, char_to_index['0']] = 1
		
	return x, y

e, l = generate_data()
a, b = vectorize_example(e, l)



def devectorize_example(example):
	result = [index_to_char[np.argmax(vec)] for i, vec in enumerate(example)]
	return ''.join(result)


def create_dataset(num_examples=2000):

    x_train = np.zeros((num_examples, max_time_steps, num_features))
    y_train = np.zeros((num_examples, max_time_steps, num_features))
    
    #x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    

    for i in range(num_examples):
        e, l = generate_data()
        x, y = vectorize_example(e, l)
        x_train[i] = x
        y_train[i] = y
    
    return x_train, y_train

x_test, y_test = create_dataset(num_examples = 20)

pred = devectorize_example(np.array(list(a)))