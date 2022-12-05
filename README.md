## Optical Character Recognition 

This project is implemented to convert virtually any kind of image containing written text like images, documents, handwritten characters into machine-readable text data.

We will define various ways of predicting text.

#### Git Clone : 

```python
git clone https://github.com/umesh-gattem/OCR_NLP_Project.git
```

#### Requirements : 

There are few requirements needs to be installed to run the project.Once we clone the project we need to go to the git project path and needs to run the following command to install all the requirements.

```python
pip3 install -r requirements.txt
```

Optical Character Recognition can be done by many ways. Recent developments in Natural Language Processing models are very helpful for this kind of problems and solved with easier ways.

### DATASETS:

For this project I am trying to use the [Wantok Images](https://wantokniuspepa.com/index.php/archives) and trying to predict the text present in this dataset.

This data is of rare language and the format is complicated and its' difficult to predict the text with the layout and format of these images.

In addition to this dataset I am using few RESUME samples and some textual images to get the benchmark and evaluation metrics of different models.

Let's learn few ways of converting image data into text data.

1. Google API
2. Paddle OCR
3. Tesseract OCR
4. Layout Language Model.


### Google API:

Google, being one of the top tech giants has made our life easier by implementing all kinds of API required for our daily life like EMAIl, DRIVE, PHOTOS, MAPS, CLOUD Storage and many more. No wonder Google also implemented many projects in NLP and one of such project/API is Cloud Textual Recognition.

In this project I have designed one such Google API which takes the image as the Input and get the textual data present in the document.

So I have used the Google service of images().annotate() module which is used to convert image data into text data and boundary boxes. 
This API needs your Google Cloud access and credentials JSON which can be created in your Google Cloud

Before going through the API and python code for this I recommend you go through the following LinkedIn Learning Course which will give you quick heads up on how to connect to Google Cloud and generate the CREDENTIALS JSON which is required.

Once you go through the above course you will understand on how to create the CREDENTIALS JSON and then we can use this [code](https://github.com/umesh-gattem/OCR_NLP_Project/blob/master/GoogleAPI/google_cloud_text_recognition.py)

You can run the above code by changing file and give your own image file and get the results. You can see the result on the console and also the result will store in the text file.

### Paddle OCR : 

Paddle OCR is one of the open source library which is practical ultra-lightweight pre-trained model, support training and deployment among server, mobile, embedded and IoT devices.

Paddle OCR is mainly designed and trained on recognising the Chinese and English character recognition. But the proposed model is also verified in several language recognition tasks like French, Korean, Japanese and German.

As mentioned it is very light weighted and can be used with or without GPU. It returns the three output components which are 

1. Text detection
2. Detected Boxes 
3. Text recognition

You can run this [code](https://github.com/umesh-gattem/OCR_NLP_Project/blob/master/paddle_ocr/paddle_ocr.py) by changing the file_path and get the results. This file also returns the predicted text data and also saves into the text file.

Paddle-OCR Research Paper: - [PP-OCR](https://arxiv.org/abs/2009.09941)

Paddle-OCR Github : [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

### Tesseract Model:

Tesseract is an Optical Character Recognition Engine for various operating system.

To get acess to this engine in our operating system we need to install the following module in out system.

To install tesseract engine in our machine we need to run the following module based on the system environment.

#### For Linux/Ubuntu environment:

```python
sudo apt-get install tesseract-ocr
```

#### For MacOS environment:

For macOS users, we’ll be using Homebrew to install Tesseract
```python
brew install tesseract
```
If you just want to update tesseract without updating any other bre components. Use the following command.

```python
HOMEBREW_NO_AUTO_UPDATE=1 brew install tesseract
```

Once we install the above commands based on our environment, we will have access to tesseract engine in our machine. Also we need one of the Python library "Pytesseract" to run the tesseract model which we already installed through our requirements.txt.

Tesseract used the power of OCR with AI to capture data from structured and unstructured data. This module extracts text from images and documents without a text layer and outputs the document into a new searchable text file, PDF, or most other popular formats.

Like PaddleOCR, Tesseract model is also a light weight and can be run with or without GPU. Also, Tesseract give the extra abilities to predict the images from the BLUR background or the bright background.

You can run this [code](https://github.com/umesh-gattem/OCR_NLP_Project/blob/master/teserract_model/teserract_ocr.py) by changing the file_path and get the results. This file also returns the predicted text data and also saves into the text file.

### Layout Language Model

Layout Language models were introduced, inspired by the BERT model where input textual information is represented by text embeddings and position embeddings. 

LayoutLM further adds two types of input embeddings: 
1. a 2-D position embedding that denotes the relative position of a token within a document; 
2. an image embedding for scanned token images within a document. 

LayoutLM is the first model where text and layout are jointly learned in a single framework for document level pre-training. 

LayoutLM is a simple but effective multi-modal pre-training method of text, layout, and image for visually-rich document understanding and information extraction tasks, such as form understanding and receipt understanding. 

We have differnt versions of Layout LM like LayoutLM, LayoutLMv2 and LayoutLMv3 and all the models performs way better than the SOTA(State of the Art) Models results on multiple datasets. 

We can use the Layout language model from the transformers python module.

We can use the LayoutLMv3FeatureExtractor module to extract the features from the documents.
The features returned consists of three components. They are 

1. Words - Which contains the text data of the document
2. Boxes - Which the text boxes co-ordinates values.
3. Pixel Values - Pixel Values.

This model also uses the Tesseract engine and Tensorflow/Pytorch libraries in backend to run the models. 

You can run this [code]() by changing the file_path and get the results. This file also returns the predicted text data and also saves into the text file.








