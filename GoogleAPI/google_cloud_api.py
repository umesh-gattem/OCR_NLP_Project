import argparse
from base64 import b64encode

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--image_path', type=str, default='../data/wantok_images/Wantok_namba_15_page-0002.jpg',
                    help='Location of the image file')

parser.add_argument('--credential_file', type=str, default='credentials.json',
                    help='Location of the Credentials file')

parser.add_argument('--output_folder_path', type=str, default='paddle_ocr',
                    help='Location of the output folder path')

args = parser.parse_args()

# Change this values to match your project
IMAGE_FILE = args.image_path
print(IMAGE_FILE)
CREDENTIALS_FILE = args.credential_file

# Connect to the Google Cloud-ML Service
credentials = GoogleCredentials.from_stream(CREDENTIALS_FILE)
service = googleapiclient.discovery.build('vision', 'v1', credentials=credentials)

# Read file and convert it to a base64 encoding
with open(IMAGE_FILE, "rb") as f:
    image_data = f.read()
    encoded_image_data = b64encode(image_data).decode('UTF-8')

# Create the request object for the Google Vision API
batch_request = [{
    'image': {
        'content': encoded_image_data
    },
    'features': [
        {
            'type': 'TEXT_DETECTION'
        }
    ]
}]
request = service.images().annotate(body={'requests': batch_request})

# Send the request to Google
response = request.execute()

# Check for errors
if 'error' in response:
    raise RuntimeError(response['error'])

# Print the results
extracted_texts = response['responses'][0]['textAnnotations']

# Print the first piece of text found in the image
extracted_text = extracted_texts[0]
print(extracted_text['description'])

with open(args.output_folder_path + "/" + "ocr_predicted.txt", "w") as file:
    file.write(extracted_text['description'])

# Print the location where the text was detected
print(extracted_text['boundingPoly'])
