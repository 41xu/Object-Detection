import torchvision.datasets as dset
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import os
from obj_detect.nn import Net
from torch.utils.data import DataLoader
from torchvision import transforms as T



if __name__ == '__main__':
    transform =T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),

    ])
    dataset = dset.ImageFolder('./CarData/TrainImages',transform=transform)
    trainloader = DataLoader(dataset, batch_size=8,shuffle=True, num_workers=4)

    model = Net()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.cuda(1)
    for epoch in range(50):
        model.train()
        runing_loss = 0.0
        for i,data in enumerate(trainloader,0):
            inputs, label = data
            print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs.cuda(1))
            loss = loss_fn(outputs.cpu(), label)
            loss.backward()
            optimizer.step()

            runing_loss +=loss.item()
        print('epoch : {}, loss: {}'.format(epoch,runing_loss/len(trainloader)))

    save_path = './trained_models'
    if not os.path.exists(save_path) :
        os.mkdir(save_path)
    torch.save({'state_dict':model.state_dict()}, save_path+'/50_model.pth.tar')
    print('Finished Training!')
