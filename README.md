# FyreWatch :fire:
### An Ai Powered Wildfire Condition Detection System 

The purpose of this project is to build an artificially intelligent system capable of detecting wildfire conditions through a myriad of different environmental datapoints. 

The procedure of this project was split into four main steps, 
* (1) determining which environmental conditions play a decisive role in wildfire environments so that they can be used as inputs for the neural network, 
* (2) obtaining data and constructing a dataset using the chosen inputs,  
* (3) training a classification neural network on the gathered dataset for accurate detection of wildfire conditions, 
* (4) integrating this trained machine learning model into an easy-to-use, prototype desktop interface with the possibility of completing a mobile-version for firefighters to use in the field. 

These 6 environmental conditions were included as datapoints in the dataset: temperature, humidity, wind speed, soil temperature, soil moisture, and K-means Red Green Blue values of dominant colors from corresponding NDVI imagery to quantify vegetation health. This data was collected from a variety of locations from around North America to account for regional bias, and the final dataset which was used to train a binary-classification neural network contained ~2000 cases. Finally the trained model, which achieved an accuracy of 98%, was integrated into a proof-of-concept desktop app (with potential to be mobile). This app takes a firefighter's inputted coordinates, uses a collection of web APIs to collect the necessary near-time datapoints, and runs it through the AI model. 

In the "Code" folder, we have included files for environmental data collection, kmeans clustering for dominant color detection in NDVI imagery, and the code used to train the neural network. We have also included screenshots of our prototype desktop app in the "Desktop App" folder. 

Through this project, we aim to make firefighting safer, easier, and drastically more efficient. For a more thorough and detailed analysis of our methodology and technology, please read the included research paper. All improvements are welcome, and do not hesitate to reach out for support, feedback, or collaboration opportunities!


![FyreWatchAi Logo](https://github.com/Gurik-M/FyreWatchAi/blob/master/LICENSE/fyrewatchai_logo.jpg)


*Gurik Mangat - Developer FyreWatchAi*
