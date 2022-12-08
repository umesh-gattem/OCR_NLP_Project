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

or you can run the below command to run this file with proper arguments

```python
python3 GoogleAPI/google_cloud_api.py --image_path data/wantok_images/Wantok_namba_15_page-0002.jpg --credential_file GoogleAPI/credentials.json --output_folder_path GoogleAPI
```

Make sure from which folder you are running and which path you are providing for the image data and output folder

Linked Course Reference : [Image Recognition LinkedIn Course](https://www.linkedin.com/learning/deep-learning-image-recognition/build-cutting-edge-image-recognition-systems?autoplay=true&u=87254282)

### Paddle OCR : 

Paddle OCR is one of the open source library which is practical ultra-lightweight pre-trained model, support training and deployment among server, mobile, embedded and IoT devices.

Paddle OCR is mainly designed and trained on recognising the Chinese and English character recognition. But the proposed model is also verified in several language recognition tasks like French, Korean, Japanese and German.

As mentioned it is very light weighted and can be used with or without GPU. It returns the three output components which are 

1. Text detection
2. Detected Boxes 
3. Text recognition

You can run this [code](https://github.com/umesh-gattem/OCR_NLP_Project/blob/master/paddle_ocr/paddle_ocr.py) by changing the file_path and get the results. This file also returns the predicted text data and also saves into the text file.

or you can run the below command to run this file with proper arguments

```python
python3 paddle_ocr/paddle_ocr_command_line.py --image_path data/wantok_images/Wantok_namba_15_page-0002.jpg --font_path .fonts/simfang.ttf --output_folder_path paddle_ocr
```

Make sure from which folder you are running and which path you are providing for the image data and output folder

Now let's see the paddle OCR model architecture

![paddle_ocr_model_architecture](https://raw.githubusercontent.com/umesh-gattem/OCR_NLP_Project/master/model_architectures/paddle_ocr_model.png)

Below are the few references for the PaddleOCR project

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

or you can run the below command to run this file with proper arguments

```python
python3 teserract_model/teserract_ocr_command_line.py --image_path data/wantok_images/Wantok_namba_15_page-0002.jpg  --output_folder_path teserract_model
```

Make sure from which folder you are running and which path you are providing for the image data and output folder

Now let's see the teserract OCR model architecture

![teserract_ocr_model_architecture](https://raw.githubusercontent.com/umesh-gattem/OCR_NLP_Project/master/model_architectures/teserract_model.png)
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

You can run this [code](https://github.com/umesh-gattem/OCR_NLP_Project/blob/master/layout_lm_ocr/layout_lm_ocr.py) by changing the file_path and get the results. This file also returns the predicted text data and also saves into the text file.

or you can run the below command to run this file with proper arguments

```python
python3 layout_lm_ocr/layout_lm_ocr_command_line.py --image_path data/wantok_images/Wantok_namba_15_page-0002.jpg --output_folder_path layout_lm_ocr
```

Make sure from which folder you are running and which path you are providing for the image data and output folder


Now let's see the Layout Language Model OCR model architecture

![layoutlm_ocr_model_architecture](https://raw.githubusercontent.com/umesh-gattem/OCR_NLP_Project/master/model_architectures/layout_lm_model.png)


### Google colab file

In addition to the above python files, I have actually created the google colab file. You can access that ipynb file from [here](https://github.com/umesh-gattem/OCR_NLP_Project/blob/master/OCR_Extraction.ipynb).

You can also refer HTML file from [here](https://github.com/umesh-gattem/OCR_NLP_Project/blob/master/OCR_Extraction.html)

You can download this file and upload to google colab and follow the instruction. You should be able to run the file with all the installations. 
**Note** : Please make sure you give input images path correctly. I have used my Google drive as storage and reading files from there. You need to change logic to read input


### Future Scope and Implementations: 

I am yet to finish two more things. 

1. Perform evaluation Metrics in the given dataset using predicted text and actual/ground truth values.
2. Finetune the Layout LM model using the input documents and also the ground truth values

But I haven't done those since I am stilling working on generating the ground truth labels of the dataset. I will be updating this repository with these tasks very soon.







