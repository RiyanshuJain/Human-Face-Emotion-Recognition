import os

import torch
import torch.nn as nn

from torchvision.models import resnet18, mobilenet_v2, squeezenet1_0, shufflenet_v2_x1_0
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from PIL import Image

from flask import Flask, render_template, request


# torchvision transforms
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(MEAN, STD)
])


# initializing models
device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = resnet18()
resnet.fc = nn.Linear(resnet.fc.in_features, 7)
resnet.load_state_dict(torch.load("models/emotion_detection_model_resnet.pth", map_location=torch.device('cpu')))
resnet.to(device)
resnet.eval()

mobilenet = mobilenet_v2()
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 7)
mobilenet.load_state_dict(torch.load("models/emotion_detection_model_mobilenet.pth", map_location=torch.device('cpu')))
mobilenet.to(device)
mobilenet.eval()

squeezenet = squeezenet1_0()
squeezenet.classifier[1] = nn.Conv2d(squeezenet.classifier[1].in_channels, 7, 1, 1)
squeezenet.load_state_dict(torch.load("models/emotion_detection_model_squeezenet.pth", map_location=torch.device('cpu')))
squeezenet.to(device)
squeezenet.eval()

shufflenet = shufflenet_v2_x1_0()
shufflenet.fc = nn.Linear(shufflenet.fc.in_features, 7)
shufflenet.load_state_dict(torch.load("models/emotion_detection_model_shufflenet.pth", map_location=torch.device('cpu')))
shufflenet.to(device)
shufflenet.eval()


# to be used by flask
app = Flask(__name__)
UPLOAD_PATH = "static/temp/"
if not os.path.isdir(UPLOAD_PATH) :
    os.mkdir(UPLOAD_PATH)

LABEL_MAP = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def predict(image_path, model_name = "shufflenet") :
    # PIL Image load and predict
    with open(image_path, "rb") as fi :
        image = Image.open(fi)
        image = image.convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad() :
        if model_name == "resnet" :
            outs = resnet(image)
        elif model_name == "shufflenet" :
            outs = shufflenet(image)
        elif model_name == "squeezenet" :
            outs = squeezenet(image)
        elif model_name == "mobilenet" :
            outs = mobilenet(image)
    
    outs = torch.argmax(outs, dim = -1)[0]
    outs = LABEL_MAP[outs]
    
    return outs


@app.route("/", methods = ["GET", "POST"])
def hello() :
    if request.method == "POST" :
        model_name = request.form["models"]
        image_file = request.files["image"]

        if image_file :
            image_path = os.path.join(UPLOAD_PATH, image_file.filename)
            image_file.save(image_path)
            pred = predict(image_path, model_name)
            return render_template("index.html", prediction = pred, image_p = image_file.filename)
        
    return render_template("index.html", prediction = None, image_p = None)


if __name__ == "__main__" :
    app.run(debug = True)
