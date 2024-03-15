from time import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Define the neural network model with Batch Normalization
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Increase number of channels
        self.bn1 = nn.BatchNorm2d(32)                # Add Batch Normalization
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # Increase number of channels
        self.bn2 = nn.BatchNorm2d(64)                # Add Batch Normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # Increase number of channels
        self.bn3 = nn.BatchNorm2d(128)                # Add Batch Normalization
        self.fc1 = nn.Linear(128 * 4 * 4, 512)        # Increase number of units
        self.fc2 = nn.Linear(512, 256)                # Decrease number of units
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Define the transform
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomRotation(10),      # Data augmentation
        transforms.RandomCrop(32, padding=4),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define batch size
    batch_size = 64

    # Load CIFAR-10 training and test datasets with augmented data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    # Define classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Initialize the network
    net = Net()
    # Move the network to GPU if available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    net.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)  # Add weight decay

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Step-wise LR decay

    # Train the network
    start_time = time()
    for epoch in range(10):  # Increase number of epochs
        net.train()  # Set the model to training mode

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:  # Print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        
        scheduler.step()  # Update learning rate
    end_time = time()
    duration = end_time - start_time

    print(f'Finished Training in {duration: .2f} seconds')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # Test the network
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    correct = 0
    total = 0
    start_time = time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time()
    duration = end_time - start_time

    print(f'Testing 10000 images completed in {duration: .2f} seconds')
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
