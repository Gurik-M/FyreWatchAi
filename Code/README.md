## This folder contains the following opensourced code for this project:

* ```data_collection.py``` - This is the code used to construct the original dataset. It uses the OpenWeatherMap API to collect near-time 
environmental conditions data based on user-inputted coordinates. A variation of this is also used in the app to collect conditions information.

* ```ndvi_kmeans.py``` - This is the k-means clustering algorithm used to find the RGB values of the dominant colour in an NDVI image. Since NDVI 
imagery colour-codes crop health and fuel moisture, with "red" being dry and "green" being healthy/moist, knowing the dominant colour's RGB values effectively quantifies this. 
More information can be found in the project paper.

* ```nn_training.py``` - Finally, this is the code used to train the classifier artificial neural network. It contains a variety of different 
hyperparameters and after training on our dataset, achieved an accuracy of ~98% (please refer to the paper for detailed anlysis). Although our approach used an artificial 
neural network, subsequent machine learning development is encouraged.
