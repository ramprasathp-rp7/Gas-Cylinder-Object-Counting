from ultralytics import YOLO

import cv2

from PIL import Image, ExifTags
import os

import numpy as np
import easyocr
from datetime import datetime

import matplotlib.patches as patches
from matplotlib import pyplot as plt

from statistics import mode
from sklearn.cluster import AgglomerativeClustering

import csv


predModelTop = YOLO("<top model path>")

predModelSide = YOLO("<side model path>")

predModelANPR = YOLO("<anpr model path>")

#input top and side images
truckNo = '1'
ext = '.jpg'

inputDir = '/content/drive/MyDrive/YOLO/DemoInputs'
inputTop = inputDir + '/Truck' + truckNo + 'top' + ext
inputSide = inputDir + '/Truck' + truckNo + 'side' + ext

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def crop_single_image(input_path, output_left_path, output_right_path):
    # Open an image file
    with Image.open(input_path) as img:
        # Correct the orientation of the image
        img = correct_orientation(img)

        # Get image dimensions
        width, height = img.size
        # Define box for left half (left, upper, right, lower)
        left_half = img.crop((0, 0, width // 2, height))
        # Define box for right half (left, upper, right, lower)
        right_half = img.crop((width // 2, 0, width, height))

        # Save the cropped images
        if not (os.path.exists(output_left_path)):
            left_half.save(output_left_path)
            right_half.save(output_right_path)

# Example usage
leftSide = inputDir + '/Truck' + truckNo + 'sideLeft' + ext
rightSide = inputDir + '/Truck' + truckNo + 'sideRight' + ext

crop_single_image(inputSide, leftSide, rightSide)

def correct_text(text):
  char_mapping = {
    'O': '0', '0': 'O',
    'o': '0',
    'l': '1',
    'I': '1', '1': 'I',
    'i': '1',
    'j': '1',
    'L': '1',
    'J': '1',
    'Z': '2', '2': 'Z',
    'E': '3', '3': 'E',
    'A': '4', '4': 'A',
    'S': '5', '5': 'S',
    's': '5',
    'G': '6', '6': 'G',
    'b': '6',
    'T': '7', '7': 'T',
    'Z': '7',
    'F': '7',
    'B': '8', '8': 'B',
    'g': '9', '9': 'g'
  }
  final_text = ""
  for char in text:
    if char.isalnum():
      final_text += char
  size = len(final_text)
  corrected_text = ''
  for i in range(2):
    if not final_text[i].isalpha():
      corrected_text += char_mapping[final_text[i]].upper()
    else:
      corrected_text += final_text[i].upper()
  for i in range(2, 4):
    if not final_text[i].isdigit():
      corrected_text += char_mapping[final_text[i]].upper()
    else:
      corrected_text += final_text[i].upper()
  var = -1
  if size == 11:
    var = 3
  elif size == 10:
    var = 2
  elif size == 9:
    var = 1
  else:
    var = 0
  for i in range(4, 4 + var):
    if not final_text[i].isalpha():
      corrected_text += char_mapping[final_text[i]].upper()
    else:
      corrected_text += final_text[i].upper()
  for i in range(4 + var, size):
    if not final_text[i].isdigit():
      corrected_text += char_mapping[final_text[i]].upper()
    else:
      corrected_text += final_text[i].upper()
  return corrected_text

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Run YOLOv8 detection
results = predModelANPR(leftSide)
image = cv2.imread(leftSide)
if (len(results[0].boxes) == 0):
  results = predModelANPR(rightSide)
  image = cv2.imread(rightSide)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.equalizeHist(gray)

# Extract bounding boxes and confidences
box = results[0].boxes[0]
final_text = ""
x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
confidence = box.conf[0]

# Crop the region of interest (ROI) from the image
roi = image[y1:y2, x1:x2]

# Use EasyOCR to detect text in the ROI
text_results = reader.readtext(roi)

# Draw bounding box and text on the image
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
for (bbox, text, prob) in text_results:
    (tl, tr, br, bl) = bbox
    tl = tuple(map(int, tl))
    tr = tuple(map(int, tr))
    br = tuple(map(int, br))
    bl = tuple(map(int, bl))
    final_text += text
    cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.polylines(image, [np.array([tl, tr, br, bl], np.int32)], True, (0, 255, 255), 2)

plate_number = correct_text(final_text)

topImg = cv2.imread(inputTop)
topRotImg = cv2.rotate(topImg, cv2.ROTATE_90_CLOCKWISE)
inputRotTop = inputDir + '/Truck' + truckNo + 'sideRot.jpg'

# Save the rotated image with the full path
if not (os.path.exists(inputRotTop)):
  cv2.imwrite(inputRotTop, topRotImg)

topImg = cv2.cvtColor(topImg, cv2.COLOR_BGR2RGB) #convert from bgr to rgb because cv2 is bgr
topRotImg = cv2.cvtColor(topRotImg, cv2.COLOR_BGR2RGB)

sideImg = cv2.imread(inputSide)
sideImg = cv2.cvtColor(sideImg, cv2.COLOR_BGR2RGB)

topResults = predModelTop.predict([inputTop], show_labels=False, iou=0.5)
topRotResults = predModelTop.predict([inputRotTop], show_labels=False, iou=0.5)
sideResults = predModelSide.predict([inputSide], show_labels=False, iou=0.5)

def plot_bounding_boxes(image_path, bounding_boxes, color='b'):
  # Load the image
  image = plt.imread(image_path)

  # Create a figure and axes
  fig, ax = plt.subplots(1)

  # Display the image
  ax.imshow(image)

  # Plot bounding boxes
  for box in bounding_boxes:
    x_min, y_min, x_max, y_max = box

    # Calculate the width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min

    # Create a rectangle patch
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor=color, facecolor='none')

    # Add the patch to the axes
    ax.add_patch(rect)

    # Add label (class and confidence) to the bounding box
    # label = f'{class_label}: {confidence:.2f}'
    # ax.text(x_min, y_min, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.7))

  # Show the plot
  plt.show()

#print(type(results[0].path), type(results[0].boxes.xyxy))
plot_bounding_boxes(topResults[0].path, topResults[0].boxes.xyxy.cpu())
print("Number of boxes: ", len(topResults[0].boxes))

plot_bounding_boxes(topRotResults[0].path, topRotResults[0].boxes.xyxy.cpu())
print("Number of boxes: ", len(topRotResults[0].boxes))

plot_bounding_boxes(sideResults[0].path, sideResults[0].boxes.xyxy.cpu())
print("Number of boxes: ", len(sideResults[0].boxes))

def showCluster(image, boundingBoxes, clusters, clusterNum, color='blue'):
  arr = []
  for i, cluster in enumerate(clusters):
    if cluster == clusterNum:
      arr.append(boundingBoxes[i])
  plot_bounding_boxes(image, arr, color)


def showClusters(image, boundingBoxes, clusters):
  temp = np.unique(clusters)
  print("Showing cluster classes: ", temp)
  colors = ['red', 'green', 'blue', 'yellow', 'purple', 'white', 'black', 'cyan', 'pink', 'magenta', 'aquamarine', 'darkseagreen', 'lightseagreen', 'teal', 'turquoise', 'cadetblue', 'lightcyan', 'darkcyan']
  for i in range(len(temp)):
    # print(clusters[i], "cluster")
    print("Showing cluster: ", temp[i])
    showCluster(image, boundingBoxes, clusters, temp[i], colors[3])

import numpy as np
from statistics import mode
from sklearn.cluster import AgglomerativeClustering

def doAgglomerativeClustering(boundingBoxes, flag):

  # Assuming you have a list of bounding box coordinates as tuples (x_min, y_min, x_max, y_max)

  # Step 1: Compute the horizontal centers of bounding boxes
  centers = np.array([(x_min + x_max) / 2 for x_min, _, x_max, _ in boundingBoxes]).reshape(-1, 1)

  # Step 2: Perform Agglomerative Clustering
  n_clusters = None  # Set the number of clusters. You can specify the exact number or leave it as None for automatic determination.
  affinity = 'euclidean'  # You can use different affinity metrics based on your data
  linkage = 'average'  # You can choose different linkage methods for agglomerative clustering

  distance_threshold = 0
  if flag == 0:
    distance_threshold = 150
    print("Rows")
  elif flag == 1:
    distance_threshold = 150
    print("Columns")
  elif flag == 2:
    print("Height")
    distance_threshold = 100

  agglomerative = AgglomerativeClustering(n_clusters=n_clusters, metric=affinity, linkage=linkage,  distance_threshold=distance_threshold)
  clustering = agglomerative.fit(centers)

  # Step 3: Get the number of columns (excluding noise points)
  num_columns = len(np.unique(clustering.labels_))

  unique_elements, counts = np.unique(clustering.labels_, return_counts=True)

  # Find the index of the maximum count
  max_count_index = np.argmax(counts)

  # Find the maximum occurring element
  max_occurring_element = unique_elements[max_count_index]
  print(unique_elements)
  maxCounts = max(counts)
  print("Maximum: ", maxCounts)
  print("1st Mode: ", mode(counts))

  if flag == 0 or flag == 1:
    num_columns = mode(counts)
  else:
    counts = [x for x in counts if x != mode(counts)]
    if len(counts) != 0:
      print("2nd Mode: ", mode(counts))
      if maxCounts >= mode(counts):
        num_columns = mode(counts)

  print("Final:", num_columns)

  return num_columns, clustering.labels_

totCount = 1
num_columns, cluster_labels = doAgglomerativeClustering(topResults[0].boxes.xyxy.cpu(), 0)
totCount *= num_columns
showClusters(topResults[0].path, topResults[0].boxes.xyxy.cpu(), cluster_labels)

num_columns, cluster_labels = doAgglomerativeClustering(topRotResults[0].boxes.xyxy.cpu(), 1)
totCount *= num_columns
showClusters(topRotResults[0].path, topRotResults[0].boxes.xyxy.cpu(), cluster_labels)

num_columns, cluster_labels = doAgglomerativeClustering(sideResults[0].boxes.xyxy.cpu(), 2)
totCount *= num_columns
showClusters(sideResults[0].path, sideResults[0].boxes.xyxy.cpu(), cluster_labels)

print("Total Count of Cylinders:", totCount)

def append_to_csv(image_path_top, image_path_side, plate_number, count, csv_file='output.csv'):
    # Get the current date and time
    current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Define the row to be appended
    row = [image_path_top, image_path_side, current_datetime, plate_number, count]

    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Check if the file is empty to write the header
        if file.tell() == 0:
            header = ['Image Path Top', 'Image Path Side', 'Date and Time', 'Plate Number', 'Count']
            writer.writerow(header)

        # Append the row to the CSV file
        writer.writerow(row)

append_to_csv(inputTop, inputSide, plate_number, totCount)