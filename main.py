import io
import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from torchvision import transforms as T
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
import random
import easyocr

model = YOLO('yolo_all_data.pt')
app = FastAPI()
reader = easyocr.Reader(['ru'],
                        model_storage_directory='./model',
                        user_network_directory='./user_network',
                        recog_network='custom_ocr',
                        detector=False,
                        gpu=False,
                        )
# Load the pre-trained YOLOv8 model

# Define the transformation to be applied to the input image

def random_valid_train_number():
    first_seven_digits = [random.randint(0, 9) for _ in range(7)]
    sum_of_seven_digits = sum(first_seven_digits)
    eighth_digit = (10 - (sum_of_seven_digits % 10)) % 10
    eight_digit_number = ''.join(map(str, first_seven_digits + [eighth_digit]))
    return eight_digit_number

def check_railcar_number(number_str):
    # Check if the input is exactly 8 digits
    if not number_str.isdigit() or len(number_str) != 8:
        return False

    # Define the list of multipliers
    multipliers = [2, 1, 2, 1, 2, 1, 2, 1]

    # Calculate the sum of products
    total = sum(int(digit) * multiplier for digit, multiplier in zip(number_str, multipliers))

    # Check if the total is divisible by 10
    return total % 10 == 0





class DetectionResult(BaseModel):
    datetime: str
    wagon_type: str
    number: str
    checksum: bool
    #image: bytes

@app.post("/predict", response_model=DetectionResult)
def predict(file: UploadFile = File(...)):
    # Read the image from the uploaded file
    #return {"message": f"Successfully uploaded {file.filename}"}
    #print('geere')
    img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    result = model.predict(img, save=True,imgsz=1280, conf=0.6)
    boxes_xyxy = result[0].boxes.xyxy.cpu().numpy()[0]
    new_box = img[int(boxes_xyxy[1]):int(boxes_xyxy[3]),int(boxes_xyxy[0]):int(boxes_xyxy[2])]
    ocr_result = reader.recognize(new_box)
    number = str(ocr_result[0][-2])
    checksum = check_railcar_number(number)
    
    # Return the detection results
    return DetectionResult(
            datetime=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            wagon_type='Грузовой вагон',
            number=number,
            checksum = checksum)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
