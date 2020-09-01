import pyowm
from pyowm.utils.geo import Polygon as GeoPolygon
from pyowm.commons.enums import ImageTypeEnum
from pyowm.agroapi10.enums import SatelliteEnum, PresetEnum
from pyowm.agroapi10.enums import PaletteEnum
import requests
from pprint import pprint
import geopandas as gpd
from shapely import geometry
import json

import warnings;
warnings.filterwarnings('ignore')


#Coordinates Script
lat = str(input("Enter Latitude: ")) 
lon = str(input("Enter Longitude: "))
lon_float = float(lon)
lat_float = float(lat)
gdf = gpd.GeoDataFrame(geometry=[geometry.Point(lon_float, lat_float)])
gdf.crs = {'init': 'epsg:4326'}
gdf = gdf.to_crs({'init': 'epsg:3857'})
gdf['geometry'] = gdf['geometry'].apply(lambda x: x.buffer(1000).envelope)
gdf = gdf.to_crs({'init': 'epsg:4326'})
gdf.plot()
data = json.loads(gdf.to_json())

fig= plt.figure()
plt.plot(range(10))
fig.savefig("save_file_name.pdf")
plt.close()

points = data['features'][0]['geometry']['coordinates'][0]

def extractDigits(lst): 
    return [el for el in lst] 
                 
lst = points 
coordinates = [extractDigits(lst)]
lat_url = str(lat)
lon_url = str(lon)

API_key = " dae2b752022b59ac615eecbdd4b6a7c0"
base_url = "http://api.openweathermap.org/data/2.5/weather?"
Final_url = ("http://api.openweathermap.org/data/2.5/weather?appid=dae2b752022b59ac615eecbdd4b6a7c0&"+ "&lat=" + lat_url + "&lon=" + lon_url)

weather_data = requests.get(Final_url).json()

temp = ("%.1f" % ((weather_data['main']['temp']) - 273))
wind_speed = ("%.1f" % ((weather_data['wind']['speed']) * 3.6))
humidity = ("%.1f" % (weather_data['main']['humidity']))


#soil temperature script
owm = pyowm.OWM('dae2b752022b59ac615eecbdd4b6a7c0')
mgr = owm.agro_manager()

polygon_points = GeoPolygon(coordinates)

polygon = mgr.create_polygon(polygon_points, 'api test 9')
polygon.id
polygon.user_id
soil = mgr.soil_data(polygon)
soil.polygon_id                                    
soil.reference_time(timeformat='unix')
soil_temp = ("%.3f" %(soil.surface_temp(unit='celsius')))
soil_moist = ("%.3f" %(soil.moisture))

# Printing Data
print("\nData Collected on 6/14/2020 at 8am:")
print('\nHumidity : ',humidity) 
print('\nWind Speed : ',wind_speed)
print('\nTemperature : ',temp)
print('\nSoil Temperature : ',soil_temp)
print('\nSoil Moisture : ',soil_moist)