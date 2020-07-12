# Simple RNN (Sum Prediction of Two Strings)
In this project the sum of two strings is predicted. This is simple recurrent neural network hands on project. In this project we are using all characters from 0-9 and plus sign and creating a string of all of them. And we have created the dataset using this string of characters. Then I have generated 2000 random numbers using numpy and train and test my model using RNN. The purpose is to tell that on the text data RNN(Recurrent Neural Network) works very well rather than CNN(Convolutional Neural Network)

In this Hands on project I have used Simple Recurrent Neural Network. And One Hot Encoded Vector. The purpose of using this vector is to create a group of bits among which the legal combinations of values are only those with a single high bit and all the others low.

## Dependencies
* tensorflow 2.0   	pip install tensorflow==2.0
* keras 2.3
* Python 3.6        If you are using anaconda3 by default it uses python 3.7 then you need to create a separete environment in which you will install python 3.6 in windows command                     prompt type conda create -n [env_name] python =3.6 , after that comman you will run another command that is conda activate [env_name] , Now your python 3.6                         environmnet is ready. And if you want to deactivate the environment type conda deactivate. 
* flask		   	      pip install flask
* numpy			        pip install numpy
* pandas		        pip install pandas


## To Run this Project

Download/clone this repository from this link: [GitHub Repository]() in your system. 
Open command prompt of windwos or  (anaconda prompt), go to the folder that contains all project files. Then run the command set FLASK_APP=classifier.py now you have set the variable after that run the command python -m flask run now your localhost server is runnig 
which will give you an address like localhost:5000 copy that and paste it in the address bar of web browser. 
Project's interface will load on the web browser.
