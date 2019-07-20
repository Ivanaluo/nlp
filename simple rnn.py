import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable


class RNN(nn.Module):
	def __init__(self, input_size, output_size, hidden_dim, n_layers):
		super(RNN, self).__init__()

		self.hidden_dim = hidden_dim

		# batch_first means that the first dim of the input and output
		self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first = True)      
		self.fc = nn.Linear(hidden_dim, output_size)

	def forward(self, x, hidden):
		# x (batch_size, seq_length, input_size)
		# hidden (n_layers, batch_size, hidden_dim)
		# r_out (batch_size, time_steps, hidden_size)
		batch_size = x.size(0)

		r_out, hidden = self.rnn(x, hidden)   # get RNN outputs
		r_out = r_out.view(-1, self.hidden_dim)  # shape output to be (batch_size * seq_length, hidden_dim)

		output = self.fc(r_out)

		return output, hidden
 

############################# check the input and output dimensions ##################################
test_rnn = RNN(input_size = 1, output_size = 1, hidden_dim = 10, n_layers = 2)

seq_length = 20
time_steps = np.linspace(0, np.pi, seq_length)
data = np.sin(time_steps)
data.resize((seq_length,1))

test_input = torch.Tensor(data).unsqueeze(0)
print('input_size: ', test_input.size())     # input_size:  torch.Size([1, 20, 1])

test_out, test_h = test_rnn(test_input, None) 
print('out_size: ', test_out.size())         # out_size:  torch.Size([20, 1])
print('hidden state size: ', test_h.size())  # hidden state size:  torch.Size([2, 1, 10])
###################################################################################################### 


# training the rnn
input_size = 1
output_size = 1
hidden_dim = 32
n_layers = 1

rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)
'''
RNN(
  (rnn): RNN(1, 32, batch_first=True)
  (fc): Linear(in_features=32, out_features=1, bias=True)
)
'''

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = 0.01)

# define the training function
def train(rnn, n_steps, print_every):

	hidden = None      # initiaize the hidden state

	for batch_i, step in enumerate(range(n_steps)):
		# defining the training data
		time_steps = np.linspace(step * np.pi, (step+1) * np.pi, seq_length + 1)
		data = np.sin(time_steps)
		data.resize((seq_length+1, 1))              # input_size = 1

		x = data[:-1]
		y = data[1:]

		x_tensor, y_tensor = torch.Tensor(x).unsqueeze(0), torch.Tensor(y)
		prediction, hidden = rnn(x_tensor, hidden)  # outputs
		
		'''Representing Memory
		make a new variable for hidden and detach the hidden state from its history
		this way, we dont backpropagate though the entire history
		'''
		hidden = hidden.data

		loss = criterion(prediction, y_tensor)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch_i % print_every == 0:
			print('loss: ', loss.item())
			plt.plot(time_steps[1:], x, 'r.')   # input
			plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.')

	return rnn

n_steps = 75
print_every = 15
trained_rnn = train(rnn, n_steps, print_every)
'''
loss:  0.3922063112258911
loss:  0.029605086892843246
loss:  0.005197410471737385
loss:  0.00023165304446592927
loss:  0.0001293337845709175
'''






































