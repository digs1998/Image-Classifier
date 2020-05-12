import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
from collections import OrderedDict 
from PIL import Image
import torch.utils.data
import pandas as pd
import json

parser = argparse.ArguementParser(description = 'Training script guidelines')
parse.add_arguement('data_dir', help='Provide the data directory, Compulsory', type = str)
parse.add_arguement('save_dir',help= 'Save the data directory, Optional arguement', type = str)
parse.add_arguement('arch', help='Vgg13 can be used as a option if this is specified else Alexnet', type = str)
parse.add_arguement('lr', help='Set learning_rate to 0.001, mandatory', type = float)
parse.add_arguement('hidden_units', help='Hidden layer units in a classifeir default value is 2048', type = int)
parse.add_arguement('epochs', help='Number of iterations to be done', type = int)
parse.add_arguement('gpu', help='Use GPU for training', type = str)

#compiling all the above
args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#we define device from CPU and GPU
if args.gpu == 'GPU':
   device = 'cuda'
else:
    device = 'cpu'
#loading the datasets

if data_dir:
    training_data_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomSizedCrop(224),
                                     transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406], [0.229,0.224, 0.225])])

    valid_data_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406], [0.229,0.224, 0.225])])

    test_data_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406], [0.229,0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder(train_dir, transform = training_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform = valid_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform = test_data_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size = 64, shuffle = True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 64, shuffle = True)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = 64, shuffle = True)
    
#we then proceed to mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    len (cat_to_name)
cat_to_name

#defining model part
def loading(arch, hidden_units):
    if arch == 'vgg13': #setting model as per vggnet
        model = models.vgg13(pretrained = True)
        for param in model.parameters(): 
            param.requires_grad = False

            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, --hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (--hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: #no hidden units
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, --hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (--hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
    else: 
        #setting the model in terms of model used Alexnet in Part-1!!
        arch = 'alexnet' #setting model as per alexnet
        model = models.alexnet(pretrained = True)
        for param in model.parameters(): 
            param.requires_grad = False
            
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, --hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (--hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
         else:
            #hidden units maybe missing
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, --hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (--hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
    model.classifier = classifier
    return model, arch


#defining validation

def validation(model, valid_loader, criterion):
    model.to(device)
    
    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_dataloaders:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return valid_loss, accuracy
            
    #loading the model with the above specifics
    model, arch= loading(args.arch, args.hidden_units)
    
    #criterion and optimizer
    criterion = nn.NLLLoss()
    if args.lr:
        optim.Adam(model.parameters(), lr=args.lr)
    else:
        optim.Adam(model.parameters(), 0.001)
    
    #let's set the criteria of epochs
    if args.epochs: #assuming epochs are specified
        epochs = args.epochs
    else:
        epochs = 7
    #training part
    model.to(device)

    print_every = 40
    train_loss = 0
    steps = 0
    for e in range(epochs):
        running_loss = 0
        for ii,(inputs, labels) in enumerate(train_dataloaders):
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion (output, labels)
            loss.backward()
            optimizer.step ()  
    
            running_loss += loss.item () 
    
        if steps % print_every == 0:
            model.eval () #switching to evaluation mode so that dropout is turned off
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_dataloaders, criterion)
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(valid_dataloaders)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(valid_dataloaders)*100))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()
#Saving the training module
model.to('cpu')
model.class_to_idx = train_image_datasets.class_to_idx

checkpoint = {'Classifier': model.classifier,
              'State dict' : model.state_dict(),
              'arch' : arch,
              'Mapping' : model.class_to_idx}

if args.save_dir:
    torch.save(checkpoint , args.save_dir + '/checkpoint.pth')
else:
    torch.save(checkpoint , 'checkpoint.pth')
