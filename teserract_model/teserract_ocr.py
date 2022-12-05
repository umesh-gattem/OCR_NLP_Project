import os

import cv2
import pytesseract
from PIL import Image

# file_path = "../data/resume/resume.png"

file_path = "../data/wantok_images/Wantok_namba_15_page-0002.jpg"

# images_path = list(Path(file_path + "/wantok_images").glob("*"))

# image = Image.open(images_path[0])


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
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
with open("ocr_predicted.txt", "w") as file:
    file.write(text)
print(text)
# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
