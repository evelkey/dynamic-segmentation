import torch.nn as nn
from torch.autograd import Variable
import torch

class Conv_1D_LSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int, length of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
      
    """
    def __init__(self, shape, input_chans, filter_size, num_features):
        super(Conv_1D_LSTM_cell, self).__init__()
        
        self.shape = shape #
        self.input_chans=input_chans
        self.filter_size=filter_size
        self.num_features = num_features
        self.padding= int((filter_size-1)/2) #in this way the output has the same size
 
        self.conv = nn.Conv1d(self.input_chans + self.num_features, 4*self.num_features, self.filter_size, 1, self.padding)
    
    def forward(self, input, hidden_state):
        hidden,c=hidden_state #hidden and c are images with several channels
        #print 'hidden ',hidden.size()
        #print 'input ',input.size()
        combined = torch.cat((input, hidden), 1) #concatenate in the channels
        #print 'combined',combined.size()
        A=self.conv(combined)
        (ai,af,ao,ag)=torch.split(A,self.num_features,dim=1)#it should return 4 tensors
        i=torch.sigmoid(ai)
        f=torch.sigmoid(af)
        o=torch.sigmoid(ao)
        g=torch.tanh(ag)
        
        next_c=f*c+i*g
        next_h=o*torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self,batch_size):
        return (Variable(torch.zeros(batch_size,self.num_features,self.shape)).cuda(),Variable(torch.zeros(batch_size,self.num_features,self.shape)).cuda())

    
class MultiConvLSTM(nn.Module):
    
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int thats the length of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size
      
    """
    def __init__(self, shape, input_chans, filter_sizes, num_features,num_layers):
        super(MultiConvLSTM, self).__init__()
        
        self.shape = shape
        self.input_chans=input_chans
        
        if not isinstance(filter_sizes, list):
            filter_sizes = [filter_sizes for _ in range(num_layers)]    
        assert len(filter_sizes) == num_layers
        self.filter_sizes=filter_sizes
        
        if not isinstance(num_features, list):
            num_features = [num_features for _ in range(num_layers)]    
        assert len(num_features) == num_layers
        self.num_features = num_features
        
        self.num_layers=num_layers
        cell_list=[]
        cell_list.append(Conv_1D_LSTM_cell(self.shape,
                                           self.input_chans,
                                           self.filter_sizes[0],
                                           self.num_features[0]).cuda())#the first
        #one has a different number of input channels
        
        for idcell in range(1,self.num_layers):
            cell_list.append(Conv_1D_LSTM_cell(self.shape,
                                               self.num_features[idcell-1],
                                               self.filter_sizes[idcell],
                                               self.num_features[idcell]).cuda())
        self.cell_list=nn.ModuleList(cell_list)      

    
    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape Batch,seq_len,Channels,length
        """

        current_input = input.transpose(0, 1)#now is seq_len,B,C,length
        #current_input=input
        next_hidden=[] #hidden states(h and c)
        seq_len=current_input.size(0)

        
        for idlayer in range(self.num_layers):#loop for every layer

            hidden_c=hidden_state[idlayer]#hidden and c are images with several channels
            all_output = []
            output_inner = []            
            for t in range(seq_len):#loop for every step
                hidden_c=self.cell_list[idlayer](current_input[t,...],hidden_c)#cell_list is a list with different conv_lstms 1 for every layer

                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            current_input = torch.cat(output_inner, 0).view(current_input.size(0),
                                                            *output_inner[0].size())#seq_len,B,chans,length


        return next_hidden, current_input

    def init_hidden(self,batch_size):
        init_states=[]#this is a list of tuples
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
    
    
if __name__ == "__main__":
    #TEST functionalty
  
    num_features=8
    filter_size=3 #only odd kernels are supported
    batch_size=10
    shape=25
    inp_chans=1
    nlayers=3
    seq_len=11

    #If using this format, then we need to transpose in CLSTM
    input = Variable(torch.rand(batch_size,seq_len,inp_chans,shape)).cuda()

    conv_lstm=MultiConvLSTM(shape, inp_chans, filter_size, num_features,nlayers)
    #conv_lstm.apply(weights_init)
    conv_lstm.cuda()

    print('convlstm module:',conv_lstm)


    print('params:')
    params=conv_lstm.parameters()
    for p in params:
       print ('param ',p.size())
       print ('mean ',torch.mean(p))


    hidden_state=conv_lstm.init_hidden(batch_size)
    print('hidden_h shape ',len(hidden_state))
    print ('hidden_h shape ',hidden_state[0][0].size())
    out=conv_lstm(input,hidden_state)
    print ('out shape',out[1].size())
    print ('len hidden ', len(out[0]))
    print ('next hidden',out[0][0][0].size())
    print ('convlstm dict',conv_lstm.state_dict().keys())


    L=torch.sum(out[1])
    L.backward()