## Library imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

## Read in pEdge
pEdge = pd.read_csv('connection-probs-full.csv')

## Initialize device
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda')


## Hyper-parameters
# bs = 100 ## batch size generally
# n_latent = 8 ## latent scaling dimension of RNNs
# pop_list = [8,1,1] ## Pyramidal neurons, SST neurons, VIP neurons in that order
# pop_list_depths = [1,1,1,1] ## Relative number of neurons at different imaging depths (L4, 2/3, 5, 6)


## Define circuit architecture
class celltypeRNN(nn.Module):
    def __init__(self, pEdge=pEdge, seq_len=128, n_features=1, latent_scaling=8, pop_list_types = [8,1,1], pop_list_depths = [1,1,1], bsize=100, device=device, manual_seed=0):
        super(celltypeRNN, self).__init__()

        self.seq_len = seq_len
        self.latent_scaling = latent_scaling
        self.bsize = bsize
        self.device = device
        self.n_features = n_features
        self.pop_types = pop_list_types
        self.pop_depths = pop_list_depths
        self.manual_seed = manual_seed

        ## Note that pEdge is a dictionary
        self.pEdge = pEdge

        ##ReLU
        self.relu = nn.ReLU()

        ##RNNs
        ## HOLD PLACE!!! Make sure input size is correct
        self.fullRNN = nn.RNN(input_size = 2*n_features*len(pop_list_types)*len(pop_list_depths), hidden_size=2*sum(pop_list_types)*sum(pop_list_depths)*latent_scaling, num_layers = 1, nonlinearity ='relu', batch_first = True)

        ##Recurrent backbones (within cell-type populations)

        ##V1 - L4
        pval = 0.1
        self.BBV1L4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.01
        self.BBV1L4SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.03
        self.BBV1L4VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##V1 - L23
        pval = 0.06
        self.BBV1L23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.05
        self.BBV1L23SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.01
        self.BBV1L23VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##V1 - L5
        pval = 0.04
        self.BBV1L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.03
        self.BBV1L5SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.03
        self.BBV1L5VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##LM - L4
        pval = 0.1
        self.BBLML4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.01
        self.BBLML4SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.03
        self.BBLML4VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##LM - L23
        pval = 0.06
        self.BBLML23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.05
        self.BBLML23SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.01
        self.BBLML23VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##LM - L5
        pval = 0.04
        self.BBLML5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.03
        self.BBLML5SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.03
        self.BBLML5VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

#########################################################################################################################################################################

        ##Inter-population backbones (across cell-type populations) (same area)

        ## V1 - V1

        ## L4 --> rest

        ## Pyr --> SST ## DONE
        pval = 0.04
        self.BBV1L4Pyr_L4SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.04
        self.BBV1L4Pyr_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.03
        self.BBV1L4Pyr_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)


        ## Pyr --> VIP ##DONE
        pval = 0.0
        self.BBV1L4Pyr_L4VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.02
        self.BBV1L4Pyr_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.11
        self.BBV1L4Pyr_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)


        ## Pyr --> Pyr ##DONE
        pval = 0.07
        self.BBV1L4Pyr_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.0
        self.BBV1L4Pyr_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr ##DONE
        pval = 0.22
        self.BBV1L4SST_L4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.33
        self.BBV1L4SST_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.08
        self.BBV1L4SST_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP ##DONE
        pval = 0.22
        self.BBV1L4SST_L4VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.21
        self.BBV1L4SST_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.07
        self.BBV1L4SST_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST ##DONE
        pval = 0.05
        self.BBV1L4SST_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.04
        self.BBV1L4SST_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr ##DONE
        pval = 0.0
        self.BBV1L4VIP_L4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L4VIP_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L4VIP_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST ##DONE
        pval = 0.14
        self.BBV1L4VIP_L4SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.25
        self.BBV1L4VIP_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.11
        self.BBV1L4VIP_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP ##DONE
        pval = 0.0
        self.BBV1L4VIP_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.04
        self.BBV1L4VIP_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ## L23 --> rest

        ## Pyr --> SST ##DONE
        pval = 0.17
        self.BBV1L23Pyr_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.3
        self.BBV1L23Pyr_L23SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.08
        self.BBV1L23Pyr_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP ##DONE
        pval = 0.0
        self.BBV1L23Pyr_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.16
        self.BBV1L23Pyr_L23VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.0
        self.BBV1L23Pyr_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr ##DONE
        pval = 0.05
        self.BBV1L23Pyr_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.08
        self.BBV1L23Pyr_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr ##DONE
        pval = 0.04
        self.BBV1L23SST_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.23
        self.BBV1L23SST_L23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L23SST_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP ##DONE
        pval = 0.13
        self.BBV1L23SST_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.3
        self.BBV1L23SST_L23VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L23SST_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST ##DONE
        pval = 0.0
        self.BBV1L23SST_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L23SST_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr ##DONE
        pval = 0.05
        self.BBV1L23VIP_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L23VIP_L23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L23VIP_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST ##DONE
        pval = 0.14
        self.BBV1L23VIP_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.14
        self.BBV1L23VIP_L23SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L23VIP_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP ##DONE
        pval = 0.05
        self.BBV1L23VIP_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.04
        self.BBV1L23VIP_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ## L5 --> rest

        ## Pyr --> SST ##DONE
        pval = 0.08
        self.BBV1L5Pyr_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.0
        self.BBV1L5Pyr_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.1
        self.BBV1L5Pyr_L5SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP ##DONE
        pval = 0.02
        self.BBV1L5Pyr_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.0
        self.BBV1L5Pyr_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.02
        self.BBV1L5Pyr_L5VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr ##DONE
        pval = 0.0
        self.BBV1L5Pyr_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.0
        self.BBV1L5Pyr_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr ##DONE
        pval = 0.19
        self.BBV1L5SST_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.15
        self.BBV1L5SST_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.15
        self.BBV1L5SST_L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP ##DONE
        pval = 0.14
        self.BBV1L5SST_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.13
        self.BBV1L5SST_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.1
        self.BBV1L5SST_L5VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST ##DONE
        pval = 0.05
        self.BBV1L5SST_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.04
        self.BBV1L5SST_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr ##DONE
        pval = 0.0
        self.BBV1L5VIP_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L5VIP_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L5VIP_L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST ##DONE
        pval = 0.2
        self.BBV1L5VIP_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L5VIP_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.05
        self.BBV1L5VIP_L5SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP ##DONE
        pval = 0.0
        self.BBV1L5VIP_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBV1L5VIP_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ############################################################################################################################################################
        ############################################################################################################################################################

        ## LM - LM

        ## L4 --> rest

        ## Pyr --> SST ## DONE
        pval = 0.04
        self.BBLML4Pyr_L4SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.04
        self.BBLML4Pyr_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.03
        self.BBLML4Pyr_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)


        ## Pyr --> VIP ##DONE
        pval = 0.0
        self.BBLML4Pyr_L4VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.02
        self.BBLML4Pyr_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.11
        self.BBLML4Pyr_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)


        ## Pyr --> Pyr ##DONE
        pval = 0.07
        self.BBLML4Pyr_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.0
        self.BBLML4Pyr_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr ##DONE
        pval = 0.22
        self.BBLML4SST_L4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.33
        self.BBLML4SST_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.08
        self.BBLML4SST_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP ##DONE
        pval = 0.22
        self.BBLML4SST_L4VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.21
        self.BBLML4SST_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.07
        self.BBLML4SST_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST ##DONE
        pval = 0.05
        self.BBLML4SST_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.04
        self.BBLML4SST_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr ##DONE
        pval = 0.0
        self.BBLML4VIP_L4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML4VIP_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML4VIP_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST ##DONE
        pval = 0.14
        self.BBLML4VIP_L4SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.25
        self.BBLML4VIP_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.11
        self.BBLML4VIP_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP ##DONE
        pval = 0.0
        self.BBLML4VIP_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.04
        self.BBLML4VIP_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ## L23 --> rest

        ## Pyr --> SST ##DONE
        pval = 0.17
        self.BBLML23Pyr_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.3
        self.BBLML23Pyr_L23SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.08
        self.BBLML23Pyr_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP ##DONE
        pval = 0.0
        self.BBLML23Pyr_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.16
        self.BBLML23Pyr_L23VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.0
        self.BBLML23Pyr_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr ##DONE
        pval = 0.05
        self.BBLML23Pyr_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.08
        self.BBLML23Pyr_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr ##DONE
        pval = 0.04
        self.BBLML23SST_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.23
        self.BBLML23SST_L23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML23SST_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP ##DONE
        pval = 0.13
        self.BBLML23SST_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.3
        self.BBLML23SST_L23VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML23SST_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST ##DONE
        pval = 0.0
        self.BBLML23SST_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML23SST_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr ##DONE
        pval = 0.05
        self.BBLML23VIP_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML23VIP_L23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML23VIP_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST ##DONE
        pval = 0.14
        self.BBLML23VIP_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.14
        self.BBLML23VIP_L23SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML23VIP_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP ##DONE
        pval = 0.05
        self.BBLML23VIP_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.04
        self.BBLML23VIP_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ## L5 --> rest

        ## Pyr --> SST ##DONE
        pval = 0.08
        self.BBLML5Pyr_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.0
        self.BBLML5Pyr_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.1
        self.BBLML5Pyr_L5SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP ##DONE
        pval = 0.02
        self.BBLML5Pyr_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.0
        self.BBLML5Pyr_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.02
        self.BBLML5Pyr_L5VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr ##DONE
        pval = 0.0
        self.BBLML5Pyr_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = 0.0
        self.BBLML5Pyr_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr ##DONE
        pval = 0.19
        self.BBLML5SST_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.15
        self.BBLML5SST_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.15
        self.BBLML5SST_L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP ##DONE
        pval = 0.14
        self.BBLML5SST_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.13
        self.BBLML5SST_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.1
        self.BBLML5SST_L5VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST ##DONE
        pval = 0.05
        self.BBLML5SST_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.04
        self.BBLML5SST_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr ##DONE
        pval = 0.0
        self.BBLML5VIP_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML5VIP_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML5VIP_L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST ##DONE
        pval = 0.2
        self.BBLML5VIP_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML5VIP_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.05
        self.BBLML5VIP_L5SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP ##DONE
        pval = 0.0
        self.BBLML5VIP_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = 0.0
        self.BBLML5VIP_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)


        ############################################################################################################################################################

        ##Output layer to project to original (input) dimension
        self.readout_V1_L4_Pyr = nn.Linear(in_features=(pop_list_depths[0]*pop_list_types[0])*latent_scaling, out_features=n_features)
        self.readout_V1_L4_SST = nn.Linear(in_features=(pop_list_depths[0]*pop_list_types[1])*latent_scaling, out_features=n_features)
        self.readout_V1_L4_VIP = nn.Linear(in_features=(pop_list_depths[0]*pop_list_types[2])*latent_scaling, out_features=n_features)

        self.readout_LM_L4_Pyr = nn.Linear(in_features=(pop_list_depths[0]*pop_list_types[0])*latent_scaling, out_features=n_features)
        self.readout_LM_L4_SST = nn.Linear(in_features=(pop_list_depths[0]*pop_list_types[1])*latent_scaling, out_features=n_features)
        self.readout_LM_L4_VIP = nn.Linear(in_features=(pop_list_depths[0]*pop_list_types[2])*latent_scaling, out_features=n_features)

        self.readout_V1_L23_Pyr = nn.Linear(in_features=(pop_list_depths[1]*pop_list_types[0])*latent_scaling, out_features=n_features)
        self.readout_V1_L23_SST = nn.Linear(in_features=(pop_list_depths[1]*pop_list_types[1])*latent_scaling, out_features=n_features)
        self.readout_V1_L23_VIP = nn.Linear(in_features=(pop_list_depths[1]*pop_list_types[2])*latent_scaling, out_features=n_features)

        self.readout_LM_L23_Pyr = nn.Linear(in_features=(pop_list_depths[1]*pop_list_types[0])*latent_scaling, out_features=n_features)
        self.readout_LM_L23_SST = nn.Linear(in_features=(pop_list_depths[1]*pop_list_types[1])*latent_scaling, out_features=n_features)
        self.readout_LM_L23_VIP = nn.Linear(in_features=(pop_list_depths[1]*pop_list_types[2])*latent_scaling, out_features=n_features)

        self.readout_V1_L5_Pyr = nn.Linear(in_features=(pop_list_depths[2]*pop_list_types[0])*latent_scaling, out_features=n_features)
        self.readout_V1_L5_SST = nn.Linear(in_features=(pop_list_depths[2]*pop_list_types[1])*latent_scaling, out_features=n_features)
        self.readout_V1_L5_VIP = nn.Linear(in_features=(pop_list_depths[2]*pop_list_types[2])*latent_scaling, out_features=n_features)

        self.readout_LM_L5_Pyr = nn.Linear(in_features=(pop_list_depths[2]*pop_list_types[0])*latent_scaling, out_features=n_features)
        self.readout_LM_L5_SST = nn.Linear(in_features=(pop_list_depths[2]*pop_list_types[1])*latent_scaling, out_features=n_features)
        self.readout_LM_L5_VIP = nn.Linear(in_features=(pop_list_depths[2]*pop_list_types[2])*latent_scaling, out_features=n_features)

    def forward(self, x):

        ## Indices for neurons by cell type, layer, and area
        ## Area hierarchy: V1, LM
        ## Layer hierarchy: L4, L2/3, L5, L6
        ## Cell type hierarchy: Pyr, SST, VIP

        ## V1, L4
        start_V1_L4_Pyr = 0
        end_V1_L4_Pyr = start_V1_L4_Pyr + self.pop_depths[0]*self.pop_types[0]*self.latent_scaling

        start_V1_L4_SST = end_V1_L4_Pyr
        end_V1_L4_SST = start_V1_L4_SST + self.pop_depths[0]*self.pop_types[1]*self.latent_scaling

        start_V1_L4_VIP = end_V1_L4_SST
        end_V1_L4_VIP = start_V1_L4_VIP + self.pop_depths[0]*self.pop_types[2]*self.latent_scaling

        ## V1, L2/3
        start_V1_L23_Pyr = end_V1_L4_VIP
        end_V1_L23_Pyr = start_V1_L23_Pyr + self.pop_depths[1]*self.pop_types[0]*self.latent_scaling

        start_V1_L23_SST = end_V1_L23_Pyr
        end_V1_L23_SST = start_V1_L23_SST + self.pop_depths[1]*self.pop_types[1]*self.latent_scaling

        start_V1_L23_VIP = end_V1_L23_SST
        end_V1_L23_VIP = start_V1_L23_VIP + self.pop_depths[1]*self.pop_types[2]*self.latent_scaling

        ## V1, L5
        start_V1_L5_Pyr = end_V1_L23_VIP
        end_V1_L5_Pyr = start_V1_L5_Pyr + self.pop_depths[2]*self.pop_types[0]*self.latent_scaling

        start_V1_L5_SST = end_V1_L5_Pyr
        end_V1_L5_SST = start_V1_L5_SST + self.pop_depths[2]*self.pop_types[1]*self.latent_scaling

        start_V1_L5_VIP = end_V1_L5_SST
        end_V1_L5_VIP = start_V1_L5_VIP + self.pop_depths[2]*self.pop_types[2]*self.latent_scaling

        ## LM, L4
        start_LM_L4_Pyr = end_V1_L5_VIP ## ending index would need to be changed if using L6 populations as well
        end_LM_L4_Pyr = start_LM_L4_Pyr + self.pop_depths[0]*self.pop_types[0]*self.latent_scaling

        start_LM_L4_SST = end_LM_L4_Pyr
        end_LM_L4_SST = start_LM_L4_SST + self.pop_depths[0]*self.pop_types[1]*self.latent_scaling

        start_LM_L4_VIP = end_LM_L4_SST
        end_LM_L4_VIP = start_LM_L4_VIP + self.pop_depths[0]*self.pop_types[2]*self.latent_scaling

        ## LM, L2/3
        start_LM_L23_Pyr = end_LM_L4_VIP
        end_LM_L23_Pyr = start_LM_L23_Pyr + self.pop_depths[1]*self.pop_types[0]*self.latent_scaling

        start_LM_L23_SST = end_LM_L23_Pyr
        end_LM_L23_SST = start_LM_L23_SST + self.pop_depths[1]*self.pop_types[1]*self.latent_scaling

        start_LM_L23_VIP = end_LM_L23_SST
        end_LM_L23_VIP = start_LM_L23_VIP + self.pop_depths[1]*self.pop_types[2]*self.latent_scaling

        ## LM, L5
        start_LM_L5_Pyr = end_LM_L23_VIP
        end_LM_L5_Pyr = start_LM_L5_Pyr + self.pop_depths[2]*self.pop_types[0]*self.latent_scaling

        start_LM_L5_SST = end_LM_L5_Pyr
        end_LM_L5_SST = start_LM_L5_SST + self.pop_depths[2]*self.pop_types[1]*self.latent_scaling

        start_LM_L5_VIP = end_LM_L5_SST
        end_LM_L5_VIP = start_LM_L5_VIP + self.pop_depths[2]*self.pop_types[2]*self.latent_scaling

        nSamp, nSteps, inDim = x.shape ## Make sure the inputs are concatenated in the correct order!!!

        ## Initialize tensor for saving reconstruction predictions for all steps of the sequence
        ## V1
        pred_V1_L4_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_V1_L4_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_V1_L4_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

        pred_V1_L23_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_V1_L23_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_V1_L23_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

        pred_V1_L5_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_V1_L5_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_V1_L5_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

        ## LM
        pred_LM_L4_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_LM_L4_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_LM_L4_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

        pred_LM_L23_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_LM_L23_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_LM_L23_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

        pred_LM_L5_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_LM_L5_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        pred_LM_L5_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

        ## Initialize hidden states (all random, sampled from [0,1) uniformly)
        h0 = torch.rand(1,nSamp,2*sum(self.pop_types)*sum(self.pop_depths)*self.latent_scaling,requires_grad=True).to(self.device)

        ## Initialize intermediate latent representations
        V1_L4_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[0]*self.pop_types[0]*self.latent_scaling).to(self.device)
        V1_L4_SST = torch.rand(nSamp,nSteps,self.pop_depths[0]*self.pop_types[1]*self.latent_scaling).to(self.device)
        V1_L4_VIP = torch.rand(nSamp,nSteps,self.pop_depths[0]*self.pop_types[2]*self.latent_scaling).to(self.device)

        V1_L23_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[1]*self.pop_types[0]*self.latent_scaling).to(self.device)
        V1_L23_SST = torch.rand(nSamp,nSteps,self.pop_depths[1]*self.pop_types[1]*self.latent_scaling).to(self.device)
        V1_L23_VIP = torch.rand(nSamp,nSteps,self.pop_depths[1]*self.pop_types[2]*self.latent_scaling).to(self.device)

        V1_L5_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[2]*self.pop_types[0]*self.latent_scaling).to(self.device)
        V1_L5_SST = torch.rand(nSamp,nSteps,self.pop_depths[2]*self.pop_types[1]*self.latent_scaling).to(self.device)
        V1_L5_VIP = torch.rand(nSamp,nSteps,self.pop_depths[2]*self.pop_types[2]*self.latent_scaling).to(self.device)

        LM_L4_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[0]*self.pop_types[0]*self.latent_scaling).to(self.device)
        LM_L4_SST = torch.rand(nSamp,nSteps,self.pop_depths[0]*self.pop_types[1]*self.latent_scaling).to(self.device)
        LM_L4_VIP = torch.rand(nSamp,nSteps,self.pop_depths[0]*self.pop_types[2]*self.latent_scaling).to(self.device)

        LM_L23_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[1]*self.pop_types[0]*self.latent_scaling).to(self.device)
        LM_L23_SST = torch.rand(nSamp,nSteps,self.pop_depths[1]*self.pop_types[1]*self.latent_scaling).to(self.device)
        LM_L23_VIP = torch.rand(nSamp,nSteps,self.pop_depths[1]*self.pop_types[2]*self.latent_scaling).to(self.device)

        LM_L5_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[2]*self.pop_types[0]*self.latent_scaling).to(self.device)
        LM_L5_SST = torch.rand(nSamp,nSteps,self.pop_depths[2]*self.pop_types[1]*self.latent_scaling).to(self.device)
        LM_L5_VIP = torch.rand(nSamp,nSteps,self.pop_depths[2]*self.pop_types[2]*self.latent_scaling).to(self.device)

        ## Mask (Input)
        ## HOLD PLACE!!! ## Keep zero any inputs going to other populations
        self.BBin = torch.zeros((2*self.n_features*len(self.pop_types)*len(self.pop_depths),2*sum(self.pop_depths)*sum(self.pop_types)*self.latent_scaling)).T.to(device)*(-1)

        ## Allow inputs to recurrent units corresponding to specific populations
        ## V1
        self.BBin[start_V1_L4_Pyr:end_V1_L4_Pyr, self.n_features*start_V1_L4_Pyr:self.n_features*end_V1_L4_Pyr] = 1.
        self.BBin[start_V1_L4_SST:end_V1_L4_SST, self.n_features*start_V1_L4_SST:self.n_features*end_V1_L4_SST] = 1.
        self.BBin[start_V1_L4_VIP:end_V1_L4_VIP, self.n_features*start_V1_L4_VIP:self.n_features*end_V1_L4_VIP] = 1.

        self.BBin[start_V1_L23_Pyr:end_V1_L23_Pyr, self.n_features*start_V1_L23_Pyr:self.n_features*end_V1_L23_Pyr] = 1.
        self.BBin[start_V1_L23_SST:end_V1_L23_SST, self.n_features*start_V1_L23_SST:self.n_features*end_V1_L23_SST] = 1.
        self.BBin[start_V1_L23_VIP:end_V1_L23_VIP, self.n_features*start_V1_L23_VIP:self.n_features*end_V1_L23_VIP] = 1.

        self.BBin[start_V1_L5_Pyr:end_V1_L5_Pyr, self.n_features*start_V1_L5_Pyr:self.n_features*end_V1_L5_Pyr] = 1.
        self.BBin[start_V1_L5_SST:end_V1_L5_SST, self.n_features*start_V1_L5_SST:self.n_features*end_V1_L5_SST] = 1.
        self.BBin[start_V1_L5_VIP:end_V1_L5_VIP, self.n_features*start_V1_L5_VIP:self.n_features*end_V1_L5_VIP] = 1.

        ## LM
        self.BBin[start_LM_L4_Pyr:end_LM_L4_Pyr, self.n_features*start_LM_L4_Pyr:self.n_features*end_LM_L4_Pyr] = 1.
        self.BBin[start_LM_L4_SST:end_LM_L4_SST, self.n_features*start_LM_L4_SST:self.n_features*end_LM_L4_SST] = 1.
        self.BBin[start_LM_L4_VIP:end_LM_L4_VIP, self.n_features*start_LM_L4_VIP:self.n_features*end_LM_L4_VIP] = 1.

        self.BBin[start_LM_L23_Pyr:end_LM_L23_Pyr, self.n_features*start_LM_L23_Pyr:self.n_features*end_LM_L23_Pyr] = 1.
        self.BBin[start_LM_L23_SST:end_LM_L23_SST, self.n_features*start_LM_L23_SST:self.n_features*end_LM_L23_SST] = 1.
        self.BBin[start_LM_L23_VIP:end_LM_L23_VIP, self.n_features*start_LM_L23_VIP:self.n_features*end_LM_L23_VIP] = 1.

        self.BBin[start_LM_L5_Pyr:end_LM_L5_Pyr, self.n_features*start_LM_L5_Pyr:self.n_features*end_LM_L5_Pyr] = 1.
        self.BBin[start_LM_L5_SST:end_LM_L5_SST, self.n_features*start_LM_L5_SST:self.n_features*end_LM_L5_SST] = 1.
        self.BBin[start_LM_L5_VIP:end_LM_L5_VIP, self.n_features*start_LM_L5_VIP:self.n_features*end_LM_L5_VIP] = 1.

        ## Steps in for loop
        for ii in range(nSteps):

            ## Make sure the inputs are concatenated in the correct order!!!
            ip = torch.unsqueeze(x[:,ii,:],1)

            ## HOLD PLACE!!!
            ## Mask input layer weights
            self.fullRNN._parameters['weight_ih_l0'].data.mul_(self.BBin)

            ## Mask self-recurrent weights
            ## V1 - L4
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L4VIP)

            ## V1 - L23
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L23VIP)

            ## V1 - L5
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L5Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L5SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L5VIP)

            ## LM - L4
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML4VIP)

            ## LM - L23
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML23VIP)

            ## LM - L5
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML5Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML5SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML5VIP)



            ## Mask inter - population weights
            ## V1 - V1

            ## L4 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L4Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L4Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L4Pyr_L5SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L4Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L4Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L4Pyr_L5VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L4Pyr_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L4Pyr_L5Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L4SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L4SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L4SST_L5Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L4SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L4SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L4SST_L5VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L4SST_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L4SST_L5SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L4VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L4VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L4VIP_L5Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L4VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L4VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L4VIP_L5SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L4VIP_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L4VIP_L5VIP)

            ##################################################################################################################################################

            ## L23 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L23Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L23Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L23Pyr_L5SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L23Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L23Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L23Pyr_L5VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L23Pyr_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L23Pyr_L5Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L23SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L23SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L23SST_L5Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L23SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L23SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L23SST_L5VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L23SST_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L23SST_L5SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L23VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L23VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L23VIP_L5Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L23VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L23VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L23VIP_L5SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L23VIP_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L23VIP_L5VIP)

            ##################################################################################################################################################

            ## L5 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L5Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L5Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L5Pyr_L5SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L5Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L5Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L5Pyr_L5VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L5Pyr_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L5Pyr_L23Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L5SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L5SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L5SST_L5Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L5SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L5SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L5SST_L5VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L5SST_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L5SST_L23SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L5VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L5VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L5VIP_L5Pyr)
            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L5VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L5VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L5VIP_L5SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L5VIP_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L5VIP_L23VIP)

            ##################################################################################################################################################
            ##################################################################################################################################################

            ## LM - ##LM
            ## L4 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML4Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML4Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML4Pyr_L5SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML4Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML4Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML4Pyr_L5VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML4Pyr_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML4Pyr_L5Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML4SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML4SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML4SST_L5Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML4SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML4SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML4SST_L5VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML4SST_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML4SST_L5SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML4VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML4VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML4VIP_L5Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML4VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML4VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML4VIP_L5SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML4VIP_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML4VIP_L5VIP)

            ##################################################################################################################################################

            ## L23 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML23Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML23Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML23Pyr_L5SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML23Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML23Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML23Pyr_L5VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML23Pyr_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML23Pyr_L5Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML23SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML23SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML23SST_L5Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML23SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML23SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML23SST_L5VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML23SST_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML23SST_L5SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML23VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML23VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML23VIP_L5Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML23VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML23VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML23VIP_L5SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML23VIP_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML23VIP_L5VIP)

            ##################################################################################################################################################

            ## L5 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML5Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML5Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML5Pyr_L5SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML5Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML5Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML5Pyr_L5VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML5Pyr_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML5Pyr_L23Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML5SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML5SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML5SST_L5Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML5SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML5SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML5SST_L5VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML5SST_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML5SST_L23SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML5VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML5VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML5VIP_L5Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML5VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML5VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML5VIP_L5SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML5VIP_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML5VIP_L23VIP)

            ##################################################################################################################################################


            ## Process data
            xFull, hiddenFull = self.fullRNN(ip,h0)
            h0 = hiddenFull

            ## Intermediate representations at V1
            V1_L4_Pyr[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L4_Pyr:end_V1_L4_Pyr])
            V1_L4_SST[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L4_SST:end_V1_L4_SST])
            V1_L4_VIP[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L4_VIP:end_V1_L4_VIP])

            V1_L23_Pyr[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L23_Pyr:end_V1_L23_Pyr])
            V1_L23_SST[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L23_SST:end_V1_L23_SST])
            V1_L23_VIP[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L23_VIP:end_V1_L23_VIP])

            V1_L5_Pyr[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L5_Pyr:end_V1_L5_Pyr])
            V1_L5_SST[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L5_SST:end_V1_L5_SST])
            V1_L5_VIP[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L5_VIP:end_V1_L5_VIP])

            ## Intermediate representations at LM
            LM_L4_Pyr[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L4_Pyr:end_LM_L4_Pyr])
            LM_L4_SST[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L4_SST:end_LM_L4_SST])
            LM_L4_VIP[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L4_VIP:end_LM_L4_VIP])

            LM_L23_Pyr[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L23_Pyr:end_LM_L23_Pyr])
            LM_L23_SST[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L23_SST:end_LM_L23_SST])
            LM_L23_VIP[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L23_VIP:end_LM_L23_VIP])

            LM_L5_Pyr[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L5_Pyr:end_LM_L5_Pyr])
            LM_L5_SST[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L5_SST:end_LM_L5_SST])
            LM_L5_VIP[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L5_VIP:end_LM_L5_VIP])

            ## Predictions at V1
            pred_V1_L4_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_V1_L4_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_V1_L4_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

            pred_V1_L23_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_V1_L23_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_V1_L23_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

            pred_V1_L5_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_V1_L5_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_V1_L5_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

            ## Predictions at LM
            pred_LM_L4_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_LM_L4_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_LM_L4_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

            pred_LM_L23_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_LM_L23_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_LM_L23_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

            pred_LM_L5_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_LM_L5_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
            pred_LM_L5_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

            ## Final projections at V1
            pred_V1_L4_Pyr[:,ii,:] = self.readout_V1_L4_Pyr(torch.squeeze(hiddenFull[:,:,start_V1_L4_Pyr:end_V1_L4_Pyr]))
            pred_V1_L4_SST[:,ii,:] = self.readout_V1_L4_SST(torch.squeeze(hiddenFull[:,:,start_V1_L4_SST:end_V1_L4_SST]))
            pred_V1_L4_VIP[:,ii,:] = self.readout_V1_L4_VIP(torch.squeeze(hiddenFull[:,:,start_V1_L4_VIP:end_V1_L4_VIP]))

            pred_V1_L23_Pyr[:,ii,:] = self.readout_V1_L23_Pyr(torch.squeeze(hiddenFull[:,:,start_V1_L23_Pyr:end_V1_L23_Pyr]))
            pred_V1_L23_SST[:,ii,:] = self.readout_V1_L23_SST(torch.squeeze(hiddenFull[:,:,start_V1_L23_SST:end_V1_L23_SST]))
            pred_V1_L23_VIP[:,ii,:] = self.readout_V1_L23_VIP(torch.squeeze(hiddenFull[:,:,start_V1_L23_VIP:end_V1_L23_VIP]))

            pred_V1_L5_Pyr[:,ii,:] = self.readout_V1_L5_Pyr(torch.squeeze(hiddenFull[:,:,start_V1_L5_Pyr:end_V1_L5_Pyr]))
            pred_V1_L5_SST[:,ii,:] = self.readout_V1_L5_SST(torch.squeeze(hiddenFull[:,:,start_V1_L5_SST:end_V1_L5_SST]))
            pred_V1_L5_VIP[:,ii,:] = self.readout_V1_L5_VIP(torch.squeeze(hiddenFull[:,:,start_V1_L5_VIP:end_V1_L5_VIP]))

            ## Final projections at LM
            pred_LM_L4_Pyr[:,ii,:] = self.readout_LM_L4_Pyr(torch.squeeze(hiddenFull[:,:,start_LM_L4_Pyr:end_LM_L4_Pyr]))
            pred_LM_L4_SST[:,ii,:] = self.readout_LM_L4_SST(torch.squeeze(hiddenFull[:,:,start_LM_L4_SST:end_LM_L4_SST]))
            pred_LM_L4_VIP[:,ii,:] = self.readout_LM_L4_VIP(torch.squeeze(hiddenFull[:,:,start_LM_L4_VIP:end_LM_L4_VIP]))

            pred_LM_L23_Pyr[:,ii,:] = self.readout_LM_L23_Pyr(torch.squeeze(hiddenFull[:,:,start_LM_L23_Pyr:end_LM_L23_Pyr]))
            pred_LM_L23_SST[:,ii,:] = self.readout_LM_L23_SST(torch.squeeze(hiddenFull[:,:,start_LM_L23_SST:end_LM_L23_SST]))
            pred_LM_L23_VIP[:,ii,:] = self.readout_LM_L23_VIP(torch.squeeze(hiddenFull[:,:,start_LM_L23_VIP:end_LM_L23_VIP]))

            pred_LM_L5_Pyr[:,ii,:] = self.readout_LM_L5_Pyr(torch.squeeze(hiddenFull[:,:,start_LM_L5_Pyr:end_LM_L5_Pyr]))
            pred_LM_L5_SST[:,ii,:] = self.readout_LM_L5_SST(torch.squeeze(hiddenFull[:,:,start_LM_L5_SST:end_LM_L5_SST]))
            pred_LM_L5_VIP[:,ii,:] = self.readout_LM_L5_VIP(torch.squeeze(hiddenFull[:,:,start_LM_L5_VIP:end_LM_L5_VIP]))


        ## would need to add tuples if including computations with L6
        return (pred_V1_L4_Pyr, pred_V1_L4_SST, pred_V1_L4_VIP), (pred_V1_L23_Pyr, pred_V1_L23_SST, pred_V1_L23_VIP), (pred_V1_L5_Pyr, pred_V1_L5_SST, pred_V1_L5_VIP), (pred_LM_L4_Pyr, pred_LM_L4_SST, pred_LM_L4_VIP), (pred_LM_L23_Pyr, pred_LM_L23_SST, pred_LM_L23_VIP), (pred_LM_L5_Pyr, pred_LM_L5_SST, pred_LM_L5_VIP), (V1_L4_Pyr, V1_L4_SST, V1_L4_VIP), (V1_L23_Pyr, V1_L23_SST, V1_L23_VIP), (V1_L5_Pyr, V1_L5_SST, V1_L5_VIP), (LM_L4_Pyr, LM_L4_SST, LM_L4_VIP), (LM_L23_Pyr, LM_L23_SST, LM_L23_VIP), (LM_L5_Pyr, LM_L5_SST, LM_L5_VIP)

#create the NN
model = celltypeRNN(pEdge = pEdge)
