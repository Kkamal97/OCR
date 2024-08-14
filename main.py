from flask import Flask, request, render_template, redirect
import cv2
import pytesseract
import numpy as np

app = Flask(__name__)

# def ocr_core(img):
#     text = pytesseract.image_to_string(img)
#     return text


# for line by line extration
def ocr_core(img):
    # Use Tesseract with the layout that maintains the original line structure
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config)
    return text


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_removal(image):
    return cv2.medianBlur(image, 5)

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    if not file:
        return redirect('/')
    
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Image processing steps
    img = get_grayscale(img)
    img = thresholding(img)
    img = noise_removal(img)

    # Perform OCR
    extracted_text = ocr_core(img)

    # Return the extracted text to the webpage
    return render_template('result.html', extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)
