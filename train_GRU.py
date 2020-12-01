import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = torchvision.datasets.MNIST(root='./datasets',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

test_data = torchvision.datasets.MNIST(root='./datasets',
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       download=True)

# Hyper parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 10
num_classes = 10
batch_size = 100
num_epochs = 1
learning_rate = 0.001

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(device)

        # Forward propagate RNN
        GRU_out, hn = self.rnn(x, h0.detach())

        # Decode the hidden state of the last time step
        GRU_out = self.fc(GRU_out[:, -1, :])

        return GRU_out


GRU_model = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
GRU_optimizer = torch.optim.Adam(GRU_model.parameters(), lr=learning_rate)

####### Train #######
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):

        image = image.reshape(-1, sequence_length, input_size).requires_grad_().to(device)
        label = label.to(device)

        # Forward
        output = GRU_model(image)
        loss = criterion(output, label)

        # Backward and optimize
        GRU_optimizer.zero_grad()
        loss.backward()
        GRU_optimizer.step()

        if (i+1) % 400 == 0:
          print("Epoch [{}/{}], Step[{}/{}], Loss:{:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

torch.save(GRU_model.state_dict(), 'gru_20141393.pth')

test_GRU_model = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)
test_GRU_model.load_state_dict(torch.load('./gru_20141393.pth'))
test_GRU_model.eval()

######## TEST ########
with torch.no_grad():
  correct = 0

  for image, label in test_loader:
    image = image.reshape(-1, sequence_length, input_size).to(device)
    label = label.to(device)
    output = test_GRU_model(image)
    _, pred = torch.max(output.data, 1)
    correct += (pred == label).sum().item()

  print('Test Accuracy of GRU model on the {} test images: {}%'.format(len(test_data), 100 * correct / len(test_data)))