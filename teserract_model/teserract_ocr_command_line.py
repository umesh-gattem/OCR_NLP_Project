import argparse
import os

import cv2
import pytesseract
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--image_path', type=str, default='../data/wantok_images/Wantok_namba_15_page-0002.jpg',
                    help='location of the image file')

parser.add_argument('--output_folder_path', type=str, default='teserract_model',
                    help='Location of the output folder path')

args = parser.parse_args()

file_path = args.image_path

# construct the argument parse and parse the arguments
preprocess = None

# load the example image and convert it to grayscale
image = cv2.imread(file_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# check to see if we should apply thresholding to preprocess the
# image
if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# make a check to see if median blurring should be done to remove
# noise
elif preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = args.output_folder_path + "/" + "result.jpg"
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
with open(args.output_folder_path + "/" + "ocr_predicted.txt", "w") as file:
    file.write(text)
print(text)
