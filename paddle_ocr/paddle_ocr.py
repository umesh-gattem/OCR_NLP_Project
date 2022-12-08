from PIL import Image, ImageFont
from paddleocr import PaddleOCR, draw_ocr

# from google.colab.patches import cv2_imshow


ocr = PaddleOCR(use_angle_cls=True, lang='en')

file_path = "../data/wantok_images/Wantok_namba_15_page-0002.jpg"

# img = cv2.imread(file_path)

result = ocr.ocr(file_path, cls=True)

predicted_text = ""
for line in result[0]:
    print(line[1][0])
    predicted_text += line[1][0]
with open("ocr_predicted.txt", "w") as file:
    file.write(predicted_text)
print(predicted_text)

# draw result

image = Image.open(file_path).convert('RGB')

boxes = [line[0] for line in result[0]]
txts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]
font = ImageFont.load_default()
print(scores)
print(len(boxes))
im_show = draw_ocr(image, boxes, txts, scores, font_path='../.fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
