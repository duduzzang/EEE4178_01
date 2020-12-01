import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

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

class LSTMModel(nn.Module):
  def __init__(self, intput_size, hidden_size, num_layers, num_classes):
    super(LSTMModel, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    # set initial hidden states and cell states
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # torch.size([2, 50, 128])
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) # torch.size([2, 50, 128])

    #Forward propagate LSTM
    LSTM_out, _  = self.lstm(x, (h0, c0)) # output: tensor [batch_size, seq_length, hidden_size]

    #Decode the hidden state of the last time step
    LSTM_out = self.fc(LSTM_out[:,-1,:])

    return LSTM_out

LSTM_model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
LSTM_optimizer = torch.optim.Adam(LSTM_model.parameters(), lr=learning_rate)

####### Train #######
total_step = len(train_loader)
for epoch in range(num_epochs):
  for i, (image, label) in enumerate(train_loader):
    image = image.reshape(-1, sequence_length, input_size).to(device)
    label = label.to(device)

    # Forward
    output = LSTM_model(image)
    loss = criterion(output, label)

    # Backward and optimize
    LSTM_optimizer.zero_grad()
    loss.backward()
    LSTM_optimizer.step()

    if (i+1) % 400 == 0:
      print("Epoch [{}/{}], Step[{}/{}], Loss:{:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))

torch.save(LSTM_model.state_dict(), 'lstm_20141393.pth')

test_LSTM_model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
test_LSTM_model.load_state_dict(torch.load('./lstm_20141393.pth'))
test_LSTM_model.eval()

######## TEST ########
with torch.no_grad():
  correct = 0

  for image, label in test_loader:
    image = image.reshape(-1, sequence_length, input_size).to(device)
    label = label.to(device)
    output = test_LSTM_model(image)
    _, pred = torch.max(output.data, 1)
    correct += (pred == label).sum().item()

  print('Test Accuracy of LSTM model on the {} test images: {}%'.format(len(test_data), 100 * correct / len(test_data)))