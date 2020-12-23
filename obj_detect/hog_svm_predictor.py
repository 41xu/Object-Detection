import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
Class = ['other', 'car']
device = torch.device('cuda:1')
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])


def predict(img_path):
    model = torch.load('models/CarData_model_25.pt')
    model = model.to(device)
    with torch.no_grad():
        img = Image.open(img_path).convert('RGB')
        plt.imshow(img)

        img = transform(img).unsqueeze(0)
        img = img.to(device)
        outputs = model(img)
        print(outputs)
        _, predicted = torch.max(outputs, 1)

        plt.title('tis picture maybe %s'%Class[predicted.item()])
        plt.show()


if __name__ == '__main__':
    predict('CarData/TrainImages/car/pos-2.pgm')

