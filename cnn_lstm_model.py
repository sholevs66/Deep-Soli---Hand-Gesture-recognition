import torch
import torch.nn as nn
import argparse
import pprint as pp

def model_builder(args):

    Cin = int(args['Cin'])
    Cout1 = int(args['Cout1'])
    Cout2 = int(args['Cout2'])
    Cout3 = int(args['Cout3'])
    Lin_lstm = int(args['Lin_lstm'])
    lstm_hidden_size = int(args['lstm_hidden_size'])
    lstm_num_layers = int(args['lstm_num_layers'])
    batch_size = int(args['batch_size'])
    Nlabels = int(args['Nlabels'])
    torch.set_default_dtype(torch.float32)

    class pred_model(nn.Module):
        def __init__(self, bidirectional_flag):
            super(pred_model, self).__init__()
            self.bidirectional = bidirectional_flag
            self.cnn_layer_1 = nn.Conv2d(Cin, Cout1, kernel_size=3)
            self.cnn_layer_2 = nn.Conv2d(Cout1, Cout2, kernel_size=2)
            self.cnn_layer_3 = nn.Conv2d(Cout2, Cout3, kernel_size=2)
            Lin_fc = 6272        # Conv layers output size for 32x32 input
            self.FC = nn.Linear(Lin_fc, Lin_lstm)

            self.LSTM_1 = nn.LSTM(input_size=Lin_lstm, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, bidirectional=bidirectional_flag, batch_first=True)

            if bidirectional_flag ==  False:
                self.Linear_out = nn.Linear(lstm_hidden_size, Nlabels)
            else:
                self.Linear_out = nn.Linear(lstm_hidden_size*2, Nlabels)
            self.logprob = nn.LogSoftmax(dim=2)

            self.hidden_state, self.cell_state = self.init_hidden(batch=1)

        def init_hidden(self, batch=batch_size):
            if self.bidirectional == False:
                m=1
            else:
                m=2
            return torch.zeros(lstm_num_layers*m, batch, lstm_hidden_size), torch.zeros(lstm_num_layers*m, batch,
                                                                        lstm_hidden_size)  # (num_layers * num_directions, batch, hidden_size) = (1, batch_size, hidden_size)

        def forward(self, input, seq_len, batch_size_fwd, stateful_fwd):
            x = self.cnn_layer_1(input)
            x = nn.functional.leaky_relu(x)
            x = self.cnn_layer_2(x)
            x = nn.functional.leaky_relu(x)
            x = self.cnn_layer_3(x)
            x = nn.functional.leaky_relu(x)
            x = x.view(x.size(0), -1)
            x = self.FC(x)
            x = x.view(batch_size_fwd, seq_len, -1) # reshape to (batch, seq_len, input_size) for LSTM

            if stateful_fwd == True:
                output, (hn, cn) = self.LSTM_1(x, [self.hidden_state,
                                                           self.cell_state])  # output - (batch, seq_len, num_directions * hidden_size) hidden state in all time steps
                                                                              # input - (batch, seq_len, input_size) note that batch_first=True!
                self.hidden_state = hn  # update last hidden state to be used for initial hidden state at next iteration
                self.cell_state = cn  # # update last cell state to be used for initial hidden state at next iteration
            else:
                output, (hn, cn) = self.LSTM_1(x)
            '''
            if self.bidirectional ==  True:
            output = output.view(input.shape[0], input.shape[1], 2, Lstate).permute(0,1,3,2)
            output = self.bi_layer(output).view(input.shape[0],input.shape[1],-1)
            '''
            x = self.logprob(self.Linear_out(output))
            return x

    model = pred_model(False)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments')

    # simulation stuf
    parser.add_argument('--save_model', help='bool to indicate if to save NN model', default=1)
    parser.add_argument('--log_en', help='bool to save log of the script in the working folder', default=1)
    parser.add_argument('--plot_show', help='bool to indicate whether to show figures at the end of run', default=0)
    parser.add_argument('--Nlabels', help='Number of used labels from the data set', default=11)

    # model params
    parser.add_argument('--Cin', help='Input channels', default=1)
    parser.add_argument('--Cout1', help=' Channels Conv Layer 1', default=5)
    parser.add_argument('--Cout2', help='Channels Conv layer 2', default=2)
    parser.add_argument('--Cout3', help='Channels Conv layer 3', default=2)
    parser.add_argument('--Lin_lstm', help='Input size to LSTM after FC', default=50)
    parser.add_argument('--lstm_hidden_size', help='LSTM hidden size', default=50)
    parser.add_argument('--lstm_num_layers', help='Number of LSTM layers', default=1)
    parser.add_argument('--batch_size', help='Default batch size', default=1)


    args = vars(parser.parse_args())
    pp.pprint(args)
    a = model_builder(args)
