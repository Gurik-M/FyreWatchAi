import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import image as img
from sklearn.cluster import KMeans
import cv2


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

# Load image
img = cv2.imread('YOUR_NDVI_IMAGE')
height, width, _ = np.shape(img)

# Reshape the image to be a list of RGB pixels
image = img.reshape((height * width, 3))

#convert from BGR to RGB for kmeans
img_kmeans = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# define a cluster
num_clusters = 1
clusters = KMeans(n_clusters=num_clusters)
clusters.fit(image)

# count the dominant colors and sort them into groups
histogram = make_histogram(clusters)

# sort the groups, most-common first
combined = zip(histogram, clusters.cluster_centers_)
combined = sorted(combined, key=lambda x: x[0], reverse=True)

# get rgb values from image to 1D array
r, g, b = cv2.split(img_kmeans)
r = r.flatten()
g = g.flatten()
b = b.flatten()

# plotting 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(r, g, b)
print("K-Means Graphical output on high fuel moisture NDVI image:")
plt.show()

# output the RGB values of the most dominant colour from the "High Fuel Moisture Image"
bars = []
hsv_values = []
for index, rows in enumerate(combined):
    bar, rgb = make_bar(100, 100, rows[1])
    dominance_text = (f'  RGB values: {rgb}')
    rgb_list = list(rgb)
    print("RGB coordinates of the most dominant colour in the high fuel moisture NDVI image: " + str(rgb_list))