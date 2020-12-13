from tkinter import *
from tkinter import Tk
from tkinter import filedialog
import os
import tkinter
import requests                                                             
from pprint import pprint
import csv
from sklearn.cluster import KMeans
import cv2
import pyowm
from pyowm.utils.geo import Polygon as GeoPolygon
from pyowm.commons.enums import ImageTypeEnum
from pyowm.agroapi10.enums import SatelliteEnum, PresetEnum
from pyowm.agroapi10.enums import PaletteEnum
import geopandas as gpd
from shapely import geometry
import json
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import ImageTk
from PIL import Image
import matplotlib.image as mpimg
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import csv
import warnings;
warnings.filterwarnings('ignore')

window = Tk()
 
window.title("Wildfire Calculator")

topframe = Frame(window, bg="#fd5f45", width=900, height=100)
topframe.place(relx=0.00001, rely=0.00001)

lbl_title = Label(window, text="Wildfire Calculator", font=("MS Serif", 50), bg="#fd5f45")
window.geometry('650x670')
lbl_title.place(relx=0.20, rely=0.02)

def file_chooser1(): #for image
    filename1 = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetypes =
        (("jpg files","*.jpg"),("all files","*.*")) )
    lbl_normalized = Label(window, text=("File: " + filename1), font=("MS Serif", 11))
    lbl_normalized.place(relx=0.29, rely=0.55)
    filename = filename1
    txtdata = open("filename.txt","w") 
    txtdata.writelines(filename) 



#Textbox to type in latitude/longitutde
loc_lbl = tkinter.Label(window, text="Enter Latitude and Longitude", font=("MS Serif", 26), bg="#fd5f45")
loc_lbl.place(relx=0.255, rely=0.18)
E1 = Entry(window, bd=2)
E1.insert(END, 'Latitude')
E1.place(relx=0.37, rely=0.27)

E2 = Entry(window, bd=2)
E2.insert(END, 'Longitude')
E2.place(relx=0.37, rely=0.33)

#Button to upload image
normalized_upload_lbl = tkinter.Label(window, text="Upload NDVI .jpg Image", font=("MS Serif", 25), bg="#fd5f45")
normalized_upload_lbl.place(relx=0.29, rely=0.42)
upload_img = PhotoImage(file="upload.gif")
upload = upload_img.subsample(3,3)
normalized_upload_btn = tkinter.Button(command = file_chooser1, text="Upload File", image=upload,
                                        compound=LEFT)
normalized_upload_btn.place(relx=0.40, rely=0.48)

#Color Dominance Algorithm
def make_histogram(cluster):
    """
    Count the number of pixels in each cluster
    :param: KMeans cluster
    :return: numpy histogram
    """
    
    numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    hist, _ = np.histogram(cluster.labels_, bins=numLabels)
    hist = hist.astype('float32')
    hist /= hist.sum()
    return hist


def make_bar(height, width, color):
    """
    Create an image of a given color
    :param: height of the image
    :param: width of the image
    :param: BGR pixel values of the color
    :return: tuple of bar, rgb values 
    """
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)


"""
METHOD FOR PREDICT BUTTON
"""
def predict():
    lat = E1.get() #string
    lon = E2.get() #string
    API_key = " dae2b752022b59ac615eecbdd4b6a7c0"# replace with new user OpenWeatherMap API key (one listed here has been deprecated)
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    Final_url = ("http://api.openweathermap.org/data/2.5/weather?appid=dae2b752022b59ac615eecbdd4b6a7c0&"+ "&lat=" + lat + "&lon=" + lon)
    weather_data = requests.get(Final_url).json()
    #weather data
    temp = ("%.1f" % ((weather_data['main']['temp']) - 273))
    wind_speed = ("%.1f" % ((weather_data['wind']['speed']) * 3.6))
    humidity = ("%.1f" % (weather_data['main']['humidity']))
    #collect soil data
    lon_float = float(lon)
    lat_float = float(lat)
    gdf = gpd.GeoDataFrame(geometry=[geometry.Point(lon_float, lat_float)])
    gdf.crs = {'init': 'epsg:4326'}
    gdf = gdf.to_crs({'init': 'epsg:3857'})
    gdf['geometry'] = gdf['geometry'].apply(lambda x: x.buffer(1000).envelope)
    gdf = gdf.to_crs({'init': 'epsg:4326'})
    gdf.plot()
    data = json.loads(gdf.to_json())
    points = data['features'][0]['geometry']['coordinates'][0]
    def extractDigits(lst): 
        return [el for el in lst] 
                    
    lst = points 
    coordinates = [extractDigits(lst)]
    owm = pyowm.OWM('dae2b752022b59ac615eecbdd4b6a7c0')
    mgr = owm.agro_manager()
    polygon_points = GeoPolygon(coordinates)
    
    warnings.filterwarnings('ignore')
    
    polygon = mgr.create_polygon(polygon_points, 'api test 9')
    polygon.id
    polygon.user_id
    soil = mgr.soil_data(polygon)
    soil.polygon_id                                    
    soil.reference_time(timeformat='unix')
    soil_temp = ("%.1f" %(soil.surface_temp(unit='celsius')))
    soil_moist = ("%.2f" %(soil.moisture))


    #kmeans
    txtdata = open("filename.txt")
    imgname = txtdata.readlines()
    imgname_1 = ' '.join([str(elem) for elem in imgname]) 

    img = cv2.imread(imgname_1)
    height, width, _ = np.shape(img)

    # reshape the image to be a simple list of RGB pixels
    image = img.reshape((height * width, 3))

    # get the most common color
    num_clusters = 1
    clusters = KMeans(n_clusters=num_clusters)
    clusters.fit(image)

    # count and group dominant colors 
    histogram = make_histogram(clusters)
    # then sort them, most-common first
    combined = zip(histogram, clusters.cluster_centers_)
    combined = sorted(combined, key=lambda x: x[0], reverse=True)


    # output a graphic showing the colors in order
    bars = []
    hsv_values = []
    for index, rows in enumerate(combined):
        bar, rgb = make_bar(100, 100, rows[1])
        dominance_text = (f'  RGB values: {rgb}')
        rgb_list = list(rgb)
    kmeansR = rgb_list[0]
    KmeansG = rgb_list[1]
    kmeansB = rgb_list[2]

    data_labels_row = str('SoilMoisture, SoilTemperature, Temperature, Wind Speed, Humidity, kmeansr, kmeansg, kmeansb')

        #write data to csv file
    with open('weather_data.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows([data_labels_row])
        writer.writerow([soil_moist, soil_temp, temp, wind_speed, humidity,kmeansR,KmeansG,kmeansB])
        writer.writerow([0.071,3.33,6.38,11.18,85,220,191,188])


    #for new analytics window
    root = tkinter.Toplevel(window) 
    root.geometry('1000x2000')
    root.configure(background='light gray')

    topframe = Frame(root, bg="#fd5f45", width=1500, height=90)
    topframe.place(relx=0.00001, rely=0.00001)

    title_lbl = tkinter.Label(root, text=("Fire Sector Analysis"), font=("MS Serif", 50), bg="#fd5f45")
    title_lbl.place(relx=0.29, rely=0.02)

    topframe = Frame(root, bg="light green", width=358, height=265)
    topframe.place(relx=0.635, rely=0.12)

    image1 = PhotoImage(file = 'ndvi.gif')
    Image_label1 = Label(root, image = image1)
    Image_label1.place(relx=0.01, rely=0.12)


    kmeans_lbl = tkinter.Label(root, text=("KMeans Data"), font=("MS Serif", 23), bg="light green")
    kmeans_lbl.place(relx=0.75, rely=0.12)


    r_str = str(kmeansR)
    g_str = str(KmeansG)
    b_str = str(kmeansB)
    kmeans_titles = tkinter.Label(root, text=(" KMeans Fuel Moisture RGB Raw Values:\n" + "R = " + r_str + "  G = " + g_str + "  B = " + b_str), font=("MS Serif", 16), bg="light green")
    kmeans_titles.place(relx=0.644, rely=0.17)

    rgb_list = [r_str, g_str, b_str]
    # Data to plot
    labels = ['Dry', 'Moderate', 'Moist']
    bar_values = rgb_list
    colors = ['red', 'yellow', 'green']
    plt.pie(bar_values, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.savefig('kmeans.png', bbox_inches='tight')

    def display_chart_img():
        img = mpimg.imread('kmeans.png')
        plt.imshow(img)
        plt.show()


    display_btn = tkinter.Button(root, text="Display Kmeans Chart", command=display_chart_img)
    display_btn.config(height=4, width=29)
    display_btn.place(relx=0.67, rely=0.23)

    data_lbl = tkinter.Label(root, text=(" KMeans Fuel Moisture RGB Raw Values:\n" + "R = " + r_str + "  G = " + g_str + "  B = " + b_str), font=("MS Serif", 16), bg="light green")
    data_lbl.place(relx=0.644, rely=0.17)
 
    topframe = Frame(root, bg="white", width=1500, height=100)
    topframe.place(relx=0.00001, rely=0.5)
    
    lbl_data_titles = Label(root, text="   SoilMoisture,   SoilTemperature,  Temperature,   WindSpeed,   Humidity,   Fuel Moisture "  , font=("Arial Bold", 16), bg="light gray")
    lbl_data_titles.place(relx = 0.212, rely = 0.47)

    fuel_moisture = (kmeansR/255) * 100
    soil_moist_str = str(soil_moist)
    soil_temp_str = str(soil_temp)
    temp_str = str(temp)
    wind_speed_str = str(wind_speed)
    humidity_str = str(humidity)
    fuel_moisture_str = str(fuel_moisture)

    root_data = Label(root, text=(soil_moist_str + "% Moist" + "    
                                  " + soil_temp_str + "^C" + "                     
                                  " + temp_str + "^C" + "         
                                  " + wind_speed_str + "m/s" + "        
                                  " + humidity_str + "%(dew point)" + "   
                                  " + fuel_moisture_str + "%"), 
                                  font=("Arial Bold", 15), bg="light gray")
                      
    root_data.place(relx = 0.03, rely = 0.54)

    #neural network
    dataset = pd.read_csv('weather_data.csv')
    dataset.head()
    X=dataset.iloc[:,0:8]
    Y=dataset.iloc[:,1]
    X.head()
    obj=StandardScaler()
    X=obj.fit_transform(X)
    model = keras.models.load_model("model.h5")
    y_pred=model.predict(X)
    for prediction in y_pred:
        if prediction>0.96:
            pred = ("Danger: Risk of Fire")
            lbl_color = "red"
            predict_text = ("The neural network has detected a significant risk of wildfire danger. ")
        else:
            pred = "Low Danger: Low Risk of Fire"
            lbl_color = "light Green"
            predict_text = ("The neural network has not detected a significant risk of wildfire conditions. ")

    #topframe = Frame(root, bg=lbl_color, width=1500, height=150)
    #topframe.place(relx=0.00001, rely=0.5)

    topframe = Frame(root, bg=lbl_color, width=1500, height=100)
    topframe.place(relx=0.00001, rely=0.69)

    lbl_pred = Label(root, text=pred, font=("Arial Bold", 20), bg=lbl_color)
    lbl_pred.place(relx = 0.355, rely = 0.64)

    lbl_predict = Label(root, text=predict_text, font=("Arial Bold", 20), bg="light gray")
    lbl_predict.place(relx = 0.14, rely = 0.72)


    root.mainloop()


    

#END METHOD FOR PREDICT BUTTON


# "Predict" Button
pred_btn = tkinter.Button(window, text="Generate Prediction", bg="blue", command=predict) #command linked
pred_btn.config(height=4, width=29)
pred_btn.place(relx=0.29, rely=0.66)

# Instruction label
instruct_lbl = tkinter.Label(window, text=("Instructions - This software is to be used with satellite imagery/data." +
    " Simply upload\n" + " one NDVI image, and the latitude/longitude\n " + 
    " 'XY' coordinates of the location of the images. Click 'Generate Prediction' to get a predicted\n" + 
    "fire danger rating and detailed analysis of your sector."), 
    font=("MS Serif", 13), bg="light gray")
instruct_lbl.place(relx=0.07, rely=0.79)

#app closing 
window.configure(background='light gray')
window.mainloop()
