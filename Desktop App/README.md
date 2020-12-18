## Proof-of-Concept Desktop App

This project focused on developing the underlying neural network technology for making accurate and reliable predictions. However, to showcase the potential, FyreWatch developed a prototype app using the python Tkinter framework. 

The app works by first prompting the user to input the coordinates of the location they want to test, along with a corresponding NDVI jpg image. It then uses  ```data_collection.py``` and ```ndvi_kmeans.py``` code to collect the various datapoints which are fed into the trained neural network model. The network's
classification of whether or not wildfire conditions are present is presented as well as all the individual datapoints so that users still have the option of individual judgement.

Screenshots of the prototype app running are provided as examples in this ```Desktop App``` folder, along with the Python code written for it. 
