# FyreWatch.ai :fire:
### An AI Powered Wildfire Detection System 

The purpose of this project is to build an artificially intelligent system capable of detecting wildfires through a myriad of different environmental datapoints. 

The procedure of this project was split into four main steps, 
* (1) determining which environmental conditions play a decisive role in wildfire environments so that they can be used as inputs for the neural network, 
* (2) obtaining data and constructing a dataset using the chosen inputs,  
* (3) training a classification neural network on the gathered dataset for accurate detection of wildfire conditions, 
* (4) integrating this trained machine learning model into an easy-to-use, prototype desktop interface with the possibility of completing a mobile-version for firefighters to use in the field. 

These 6 environmental conditions were included as datapoints in the dataset: temperature, humidity, wind speed, soil temperature, soil moisture, and K-means Red Green Blue values of dominant colors from corresponding NDVI imagery to quantify vegetation health. 
This data was collected from a variety of locations from around North America to account for regional bias, and the final dataset which was used to train a binary-classification neural network contained ~2000 cases. Finally the trained model, which achieved an accuracy of 98%, was integrated into
a proof-of-concept desktop app (with potential to be mobile). This app takes a firefighter's inputted coordinates, uses a collection of web APIs to collect the necessary near-time datapoints, and runs it through the AI model. Through this project, we aim to make firefighting safer, easier, and drastically
more efficient. For a much more thorough and detailed analysis, please read the included research paper.

*By Gurik Mangat*
