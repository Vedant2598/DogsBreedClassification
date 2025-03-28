from fastapi import routing
from fastapi import FastAPI, File, UploadFile

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
from PIL import Image
import json
import io
import numpy as np
import base64

from ultralytics import YOLO
yolo=YOLO("yolov8n.pt")

router=routing.APIRouter()

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

class_names=None
with open("./classes.json") as file:
    class_names = json.load(file)["classes"]

# Modify the final fully connected layer for num_classes
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)
checkpoint=torch.load("./best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])



def predict_image(image, model, class_names):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    # image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    # print(predicted.item())
    predicted_class = class_names[int(predicted.item())]
    return predicted_class





@router.post("/DogsBreedClassification")
async def predict__(file: UploadFile = File(...)):
    print("Working")
    try:
        image_data = await file.read()
        # Open the image using PIL from the byte data
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_array=np.array(image)
        # print(image.shape)
        results = yolo(image,device="cpu")

        dog_class_id = 16

        send_Images=[]
        send_Predictions=[]
        for i, box in enumerate(results[0].boxes.data):
            x1, y1, x2, y2, conf, class_id = box

            if int(class_id) == dog_class_id:
                # Crop and save the detected dog
                cropped_dog = image_array[int(y1):int(y2), int(x1):int(x2)]
                cropped_dog=Image.fromarray(cropped_dog)
                prediction=predict_image(cropped_dog,model,class_names)

                buffered = io.BytesIO()
                cropped_dog.save(buffered, format="JPEG")  # Save PIL image to buffer
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")  # Convert to base64 string
                send_Predictions.append(prediction)
                send_Images.append(img_str)

 
        # Respond with a success message
        return {"message": f"File {file.filename} uploaded and opened successfully!","image":send_Images,"predictions":send_Predictions}
    
    except Exception as e: 
        return {"message": f"Failed to process image. Error: {str(e)}"}
