import argparse

from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from transformers import LayoutLMv3FeatureExtractor

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--image_path', type=str, default='../data/wantok_images/Wantok_namba_15_page-0002.jpg',
                    help='location of the image file')

parser.add_argument('--output_folder_path', type=str, default='paddle_ocr',
                    help='Location of the output folder path')

args = parser.parse_args()

# Change this values to match your project
IMAGE_FILE = args.image_path

file_path = IMAGE_FILE

image = Image.open(file_path)
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=True, ocr_lang='eng')

features = feature_extractor(image)

print(f"Words: {features['words'][0]}")
# print(f"Boxes: {features['boxes'][0]}")
print(f"Image pixels: {features['pixel_values'][0].shape}")

predicted_ocr = ""
for word in features['words'][0]:
    predicted_ocr += word + " "

print(predicted_ocr)

with open(args.output_folder_path + "/" + "ocr_predicted.txt", "w") as file:
    file.write(predicted_ocr)

image = Image.open(file_path)

# image = Image.open(file_path + "/resume_images/layout_image.png")

# image = Image.open(file_path + "/others/telugu_image.png")
draw = ImageDraw.Draw(image)

width_scale = image.width / 1000
height_scale = image.height / 1000

for boundary_box in features['boxes'][0]:
    draw.rectangle([boundary_box[0] * width_scale, boundary_box[1] * height_scale,
                    boundary_box[2] * width_scale, boundary_box[3] * height_scale],
                   outline='red', width=2)
filename = args.output_folder_path + "/" + "result.png"
plt.imshow(image)
plt.savefig(filename)
