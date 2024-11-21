## Library imports
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

## Initialize device
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda')


## Hyper-parameters
# bs = 100 ## batch size
# n_latent = 8 ## hidden dimension of RNNs
# pop_list = [8,1,1] ## Pyramidal neurons, SST neurons, VIP neurons in that order
# pop_list_depths = [1,1,1,1] ## Relative number of neurons at different imaging depths (L4, 2/3, 5, 6)


## Define circuit architecture
class celltypeRNN(nn.Module):
    def __init__(self, pEdge, seq_len=1, n_features=64, latent_scaling=8, pop_list_types = [8,1,1], pop_list_depths = [1,1,1], bsize=100, device=device, manual_seed=0):
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
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L4SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L4VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##V1 - L23
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L23SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L23VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##V1 - L5
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L5SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L5VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##V1 - L6 ## commented out because we're not using L6
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6Pyr = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6SST = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6VIP = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##LM - L4
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML4SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML4VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##LM - L23
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML23SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML23VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##LM - L5
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML5SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML5VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ##LM - L6 ## commented out because we're not using L6
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6Pyr = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6SST = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6VIP = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

#########################################################################################################################################################################

        ##Inter-population backbones (across cell-type populations) (same area)

        ## V1 - V1

        ## L4 --> rest

        ## Pyr --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L4Pyr_L4SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L4Pyr_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L4Pyr_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L4Pyr_L6SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L4Pyr_L4VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L4Pyr_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L4Pyr_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L4Pyr_L6VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L4Pyr_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L4Pyr_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L4Pyr_L6Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L4SST_L4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L4SST_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L4SST_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L4SST_L6Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L4SST_L4VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L4SST_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L4SST_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L4SST_L6VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L4SST_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L4SST_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L4SST_L6SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L4VIP_L4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L4VIP_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L4VIP_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L4VIP_L6Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L4VIP_L4SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L4VIP_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L4VIP_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L4VIP_L6SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L4VIP_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L4VIP_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L4VIP_L6VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ## L23 --> rest

        ## Pyr --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L23Pyr_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L23Pyr_L23SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L23Pyr_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L23Pyr_L6SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L23Pyr_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L23Pyr_L23VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L23Pyr_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L23Pyr_L6VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L23Pyr_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L23Pyr_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L23Pyr_L6Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L23SST_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L23SST_L23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L23SST_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L23SST_L6Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L23SST_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L23SST_L23VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L23SST_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L23SST_L6VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L23SST_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L23SST_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L23SST_L6SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L23VIP_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L23VIP_L23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L23VIP_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L23VIP_L6Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L23VIP_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L23VIP_L23SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L23VIP_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L23VIP_L6SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L23VIP_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L23VIP_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L23VIP_L6VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ## L5 --> rest

        ## Pyr --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L5Pyr_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L5Pyr_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L5Pyr_L5SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L5Pyr_L6SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L5Pyr_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L5Pyr_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L5Pyr_L5VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L5Pyr_L6VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L5Pyr_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L5Pyr_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L5Pyr_L6Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L5SST_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L5SST_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L5SST_L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L5SST_L6Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L5SST_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L5SST_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L5SST_L5VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L5SST_L6VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L5SST_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L5SST_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L5SST_L6SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L5VIP_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L5VIP_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBV1L5VIP_L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L5VIP_L6Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L5VIP_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L5VIP_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBV1L5VIP_L5SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L5VIP_L6SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L5VIP_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBV1L5VIP_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L5VIP_L6VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ## L6 --> rest

        ## Pyr --> SST
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6Pyr_L4SST = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6Pyr_L23SST = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6Pyr_L5SST = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6Pyr_L6SST = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6Pyr_L4VIP = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6Pyr_L23VIP = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6Pyr_L5VIP = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6Pyr_L6VIP = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6Pyr_L23Pyr = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6Pyr_L5Pyr = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6Pyr_L4Pyr = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6SST_L4Pyr = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6SST_L23Pyr = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6SST_L5Pyr = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6SST_L6Pyr = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6SST_L4VIP = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6SST_L23VIP = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6SST_L5VIP = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6SST_L6VIP = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6SST_L23SST = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6SST_L5SST = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6SST_L4SST = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6VIP_L4Pyr = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6VIP_L23Pyr = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6VIP_L5Pyr = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBV1L6VIP_L6Pyr = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6VIP_L4SST = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6VIP_L23SST = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6VIP_L5SST = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBV1L6VIP_L6SST = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6VIP_L23VIP = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6VIP_L5VIP = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBV1L6VIP_L4VIP = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################
        ############################################################################################################################################################

        ## LM - LM

        ## L4 --> rest

        ## Pyr --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML4Pyr_L4SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML4Pyr_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML4Pyr_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML4Pyr_L6SST = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML4Pyr_L4VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML4Pyr_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML4Pyr_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML4Pyr_L6VIP = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML4Pyr_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML4Pyr_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML4Pyr_L6Pyr = torch.ones((pop_list_depths[0]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML4SST_L4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML4SST_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML4SST_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML4SST_L6Pyr = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML4SST_L4VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML4SST_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML4SST_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML4SST_L6VIP = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML4SST_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML4SST_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML4SST_L6SST = torch.ones((pop_list_depths[0]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML4VIP_L4Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML4VIP_L23Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML4VIP_L5Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML4VIP_L6Pyr = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML4VIP_L4SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML4VIP_L23SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML4VIP_L5SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML4VIP_L6SST = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML4VIP_L23VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML4VIP_L5VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L4') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML4VIP_L6VIP = torch.ones((pop_list_depths[0]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ## L23 --> rest

        ## Pyr --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML23Pyr_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML23Pyr_L23SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML23Pyr_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML23Pyr_L6SST = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML23Pyr_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML23Pyr_L23VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML23Pyr_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML23Pyr_L6VIP = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML23Pyr_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML23Pyr_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML23Pyr_L6Pyr = torch.ones((pop_list_depths[1]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML23SST_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML23SST_L23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML23SST_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML23SST_L6Pyr = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML23SST_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML23SST_L23VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML23SST_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML23SST_L6VIP = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML23SST_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML23SST_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML23SST_L6SST = torch.ones((pop_list_depths[1]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML23VIP_L4Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML23VIP_L23Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML23VIP_L5Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML23VIP_L6Pyr = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML23VIP_L4SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML23VIP_L23SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML23VIP_L5SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML23VIP_L6SST = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML23VIP_L4VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML23VIP_L5VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L23') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML23VIP_L6VIP = torch.ones((pop_list_depths[1]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ## L5 --> rest

        ## Pyr --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML5Pyr_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML5Pyr_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML5Pyr_L5SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML5Pyr_L6SST = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML5Pyr_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML5Pyr_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML5Pyr_L5VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML5Pyr_L6VIP = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML5Pyr_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML5Pyr_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML5Pyr_L6Pyr = torch.ones((pop_list_depths[2]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML5SST_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML5SST_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML5SST_L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML5SST_L6Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML5SST_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML5SST_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML5SST_L5VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML5SST_L6VIP = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML5SST_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML5SST_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML5SST_L6SST = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML5VIP_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML5VIP_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        self.BBLML5VIP_L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML5VIP_L6Pyr = torch.ones((pop_list_depths[2]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML5VIP_L4SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML5VIP_L23SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        self.BBLML5VIP_L5SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML5VIP_L6SST = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP
        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML5VIP_L23VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        self.BBLML5VIP_L4VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        # pval = pEdge[(pEdge['pre-layer'] == 'L5') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML5VIP_L6VIP = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ############################################################################################################################################################

        ## L6 --> rest

        ## Pyr --> SST
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6Pyr_L4SST = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6Pyr_L23SST = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6Pyr_L5SST = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6Pyr_L6SST = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> VIP
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6Pyr_L4VIP = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6Pyr_L23VIP = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6Pyr_L5VIP = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6Pyr_L6VIP = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## Pyr --> Pyr
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6Pyr_L23Pyr = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6Pyr_L5Pyr = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'Pyr') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6Pyr_L4Pyr = torch.ones((pop_list_depths[3]*pop_list_types[0])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(+1)

        ## SST --> Pyr
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6SST_L4Pyr = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6SST_L23Pyr = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6SST_L5Pyr = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6SST_L6Pyr = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> VIP
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6SST_L4VIP = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6SST_L23VIP = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6SST_L5VIP = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6SST_L6VIP = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[3]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## SST --> SST
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6SST_L23SST = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6SST_L5SST = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'SST') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6SST_L4SST = torch.ones((pop_list_depths[3]*pop_list_types[1])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> Pyr
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6VIP_L4Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6VIP_L23Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6VIP_L5Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'Pyr')]
        # self.BBLML6VIP_L6Pyr = torch.ones((pop_list_depths[2]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[0])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> SST
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6VIP_L4SST = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6VIP_L23SST = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6VIP_L5SST = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L6') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'SST')]
        # self.BBLML6VIP_L6SST = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[3]*pop_list_types[1])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

        ## VIP --> VIP
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L23') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6VIP_L23VIP = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[1]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L5') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6VIP_L5VIP = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[2]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)
        #
        # pval = pEdge[(pEdge['pre-layer'] == 'L6') & (pEdge['post-layer'] == 'L4') & (pEdge['pre-cell-type'] == 'VIP') & (pEdge['post-cell-type'] == 'VIP')]
        # self.BBLML6VIP_L4VIP = torch.ones((pop_list_depths[3]*pop_list_types[2])*latent_scaling,(pop_list_depths[0]*pop_list_types[2])*latent_scaling).T.bernoulli_(p=pval,generator=torch.manual_seed(manual_seed)).to(device)*(-1)

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

        # self.readout_V1_L6_Pyr = nn.Linear(in_features=(pop_list_depths[3]*pop_list_types[0])*latent_scaling, out_features=n_features)
        # self.readout_V1_L6_SST = nn.Linear(in_features=(pop_list_depths[3]*pop_list_types[1])*latent_scaling, out_features=n_features)
        # self.readout_V1_L6_VIP = nn.Linear(in_features=(pop_list_depths[3]*pop_list_types[2])*latent_scaling, out_features=n_features)

        # self.readout_LM_L6_Pyr = nn.Linear(in_features=(pop_list_depths[3]*pop_list_types[0])*latent_scaling, out_features=n_features)
        # self.readout_LM_L6_SST = nn.Linear(in_features=(pop_list_depths[3]*pop_list_types[1])*latent_scaling, out_features=n_features)
        # self.readout_LM_L6_VIP = nn.Linear(in_features=(pop_list_depths[3]*pop_list_types[2])*latent_scaling, out_features=n_features)

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

        ## V1, L6
        # start_V1_L6_Pyr = end_V1_L5_VIP
        # end_V1_L6_Pyr = start_V1_L6_Pyr + self.pop_depths[3]*self.pop_types[0]*self.latent_scaling
        #
        # start_V1_L6_SST = end_V1_L6_Pyr
        # end_V1_L6_SST = start_V1_L6_SST + self.pop_depths[3]*self.pop_types[1]*self.latent_scaling
        #
        # start_V1_L6_VIP = end_V1_L6_SST
        # end_V1_L6_VIP = start_V1_L6_VIP + self.pop_depths[3]*self.pop_types[2]*self.latent_scaling

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

        ## LM, L6
        # start_LM_L6_Pyr = end_LM_L5_VIP
        # end_LM_L6_Pyr = start_LM_L6_Pyr + self.pop_depths[3]*self.pop_types[0]*self.latent_scaling
        #
        # start_LM_L6_SST = end_LM_L6_Pyr
        # end_LM_L6_SST = start_LM_L6_SST + self.pop_depths[3]*self.pop_types[1]*self.latent_scaling
        #
        # start_LM_L6_VIP = end_LM_L6_SST
        # end_LM_L6_VIP = start_LM_L6_VIP + self.pop_depths[3]*self.pop_types[2]*self.latent_scaling

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

        # pred_V1_L6_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        # pred_V1_L6_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        # pred_V1_L6_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

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

        # pred_LM_L6_Pyr = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        # pred_LM_L6_SST = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)
        # pred_LM_L6_VIP = torch.zeros(nSamp,nSteps,self.n_features,requires_grad=False).to(self.device)

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

        # V1_L6_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[3]*self.pop_types[0]*self.latent_scaling).to(self.device)
        # V1_L6_SST = torch.rand(nSamp,nSteps,self.pop_depths[3]*self.pop_types[1]*self.latent_scaling).to(self.device)
        # V1_L6_VIP = torch.rand(nSamp,nSteps,self.pop_depths[3]*self.pop_types[2]*self.latent_scaling).to(self.device)

        LM_L4_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[0]*self.pop_types[0]*self.latent_scaling).to(self.device)
        LM_L4_SST = torch.rand(nSamp,nSteps,self.pop_depths[0]*self.pop_types[1]*self.latent_scaling).to(self.device)
        LM_L4_VIP = torch.rand(nSamp,nSteps,self.pop_depths[0]*self.pop_types[2]*self.latent_scaling).to(self.device)

        LM_L23_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[1]*self.pop_types[0]*self.latent_scaling).to(self.device)
        LM_L23_SST = torch.rand(nSamp,nSteps,self.pop_depths[1]*self.pop_types[1]*self.latent_scaling).to(self.device)
        LM_L23_VIP = torch.rand(nSamp,nSteps,self.pop_depths[1]*self.pop_types[2]*self.latent_scaling).to(self.device)

        LM_L5_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[2]*self.pop_types[0]*self.latent_scaling).to(self.device)
        LM_L5_SST = torch.rand(nSamp,nSteps,self.pop_depths[2]*self.pop_types[1]*self.latent_scaling).to(self.device)
        LM_L5_VIP = torch.rand(nSamp,nSteps,self.pop_depths[2]*self.pop_types[2]*self.latent_scaling).to(self.device)

        # LM_L6_Pyr = torch.rand(nSamp,nSteps,self.pop_depths[3]*self.pop_types[0]*self.latent_scaling).to(self.device)
        # LM_L6_SST = torch.rand(nSamp,nSteps,self.pop_depths[3]*self.pop_types[1]*self.latent_scaling).to(self.device)
        # LM_L6_VIP = torch.rand(nSamp,nSteps,self.pop_depths[3]*self.pop_types[2]*self.latent_scaling).to(self.device)

        ## Mask (Input)
        ## HOLD PLACE!!! ## Keep zero any inputs going to other populations
        self.BBin = torch.zeros((2*n_features*len(pop_list_types)*len(pop_list_depths),2*sum(self.pop_depths)*sum(self.pop_types)*latent_scaling)).T.to(device)*(-1)

        ## Allow inputs to recurrent units corresponding to specific populations
        ## V1
        self.BBin[start_V1_L4_Pyr:end_V1_L4_Pyr, n_features*start_V1_L4_Pyr:n_features*end_V1_L4_Pyr] = 1.
        self.BBin[start_V1_L4_SST:end_V1_L4_SST, n_features*start_V1_L4_SST:n_features*end_V1_L4_SST] = 1.
        self.BBin[start_V1_L4_VIP:end_V1_L4_VIP, n_features*start_V1_L4_VIP:n_features*end_V1_L4_VIP] = 1.

        self.BBin[start_V1_L23_Pyr:end_V1_L23_Pyr, n_features*start_V1_L23_Pyr:n_features*end_V1_L23_Pyr] = 1.
        self.BBin[start_V1_L23_SST:end_V1_L23_SST, n_features*start_V1_L23_SST:n_features*end_V1_L23_SST] = 1.
        self.BBin[start_V1_L23_VIP:end_V1_L23_VIP, n_features*start_V1_L23_VIP:n_features*end_V1_L23_VIP] = 1.

        self.BBin[start_V1_L5_Pyr:end_V1_L5_Pyr, n_features*start_V1_L5_Pyr:n_features*end_V1_L5_Pyr] = 1.
        self.BBin[start_V1_L5_SST:end_V1_L5_SST, n_features*start_V1_L5_SST:n_features*end_V1_L5_SST] = 1.
        self.BBin[start_V1_L5_VIP:end_V1_L5_VIP, n_features*start_V1_L5_VIP:n_features*end_V1_L5_VIP] = 1.

        # self.BBin[start_V1_L6_Pyr:end_V1_L6_Pyr, n_features*start_V1_L6_Pyr:n_features*end_V1_L6_Pyr] = 1.
        # self.BBin[start_V1_L6_SST:end_V1_L6_SST, n_features*start_V1_L6_SST:n_features*end_V1_L6_SST] = 1.
        # self.BBin[start_V1_L6_VIP:end_V1_L6_VIP, n_features*start_V1_L6_VIP:n_features*end_V1_L6_VIP] = 1.

        ## LM
        self.BBin[start_LM_L4_Pyr:end_LM_L4_Pyr, n_features*start_LM_L4_Pyr:n_features*end_LM_L4_Pyr] = 1.
        self.BBin[start_LM_L4_SST:end_LM_L4_SST, n_features*start_LM_L4_SST:n_features*end_LM_L4_SST] = 1.
        self.BBin[start_LM_L4_VIP:end_LM_L4_VIP, n_features*start_LM_L4_VIP:n_features*end_LM_L4_VIP] = 1.

        self.BBin[start_LM_L23_Pyr:end_LM_L23_Pyr, n_features*start_LM_L23_Pyr:n_features*end_LM_L23_Pyr] = 1.
        self.BBin[start_LM_L23_SST:end_LM_L23_SST, n_features*start_LM_L23_SST:n_features*end_LM_L23_SST] = 1.
        self.BBin[start_LM_L23_VIP:end_LM_L23_VIP, n_features*start_LM_L23_VIP:n_features*end_LM_L23_VIP] = 1.

        self.BBin[start_LM_L5_Pyr:end_LM_L5_Pyr, n_features*start_LM_L5_Pyr:n_features*end_LM_L5_Pyr] = 1.
        self.BBin[start_LM_L5_SST:end_LM_L5_SST, n_features*start_LM_L5_SST:n_features*end_LM_L5_SST] = 1.
        self.BBin[start_LM_L5_VIP:end_LM_L5_VIP, n_features*start_LM_L5_VIP:n_features*end_LM_L5_VIP] = 1.

        # self.BBin[start_LM_L6_Pyr:end_LM_L6_Pyr, n_features*start_LM_L6_Pyr:n_features*end_LM_L6_Pyr] = 1.
        # self.BBin[start_LM_L6_SST:end_LM_L6_SST, n_features*start_LM_L6_SST:n_features*end_LM_L6_SST] = 1.
        # self.BBin[start_LM_L6_VIP:end_LM_L6_VIP, n_features*start_LM_L6_VIP:n_features*end_LM_L6_VIP] = 1.

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

            ## V1 - L6
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L6Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L6SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L6VIP)

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

            ## LM - L6
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML6Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML6SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML6VIP)

            ## Mask inter - population weights
            ## V1 - V1

            ## L4 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L4Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L4Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L4Pyr_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L4Pyr_L6SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L4Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L4Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L4Pyr_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L4Pyr_L6VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L4Pyr_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L4Pyr_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_Pyr:end_V1_L4_Pyr,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L4Pyr_L6Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L4SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L4SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L4SST_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L4SST_L6Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L4SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L4SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L4SST_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L4SST_L6VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L4SST_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L4SST_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_SST:end_V1_L4_SST,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L4SST_L6SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L4VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L4VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L4VIP_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L4VIP_L6Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L4VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L4VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L4VIP_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L4VIP_L6SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L4VIP_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L4VIP_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L4_VIP:end_V1_L4_VIP,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L4VIP_L6VIP)

            ##################################################################################################################################################

            ## L23 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L23Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L23Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L23Pyr_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L23Pyr_L6SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L23Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L23Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L23Pyr_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L23Pyr_L6VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L23Pyr_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L23Pyr_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_Pyr:end_V1_L23_Pyr,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L23Pyr_L6Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L23SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L23SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L23SST_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L23SST_L6Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L23SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L23SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L23SST_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L23SST_L6VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L23SST_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L23SST_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_SST:end_V1_L23_SST,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L23SST_L6SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L23VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L23VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L23VIP_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L23VIP_L6Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L23VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L23VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L23VIP_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L23VIP_L6SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L23VIP_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L23VIP_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L23_VIP:end_V1_L23_VIP,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L23VIP_L6VIP)

            ##################################################################################################################################################

            ## L5 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L5Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L5Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L5Pyr_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L5Pyr_L6SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L5Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L5Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L5Pyr_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L5Pyr_L6VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L5Pyr_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L5Pyr_L23Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_Pyr:end_V1_L5_Pyr,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L5Pyr_L6Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L5SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L5SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L5SST_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L5SST_L6Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L5SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L5SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L5SST_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L5SST_L6VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L5SST_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L5SST_L23SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_SST:end_V1_L5_SST,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L5SST_L6SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L5VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L5VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L5VIP_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L5VIP_L6Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L5VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L5VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L5VIP_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L5VIP_L6SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L5VIP_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L5VIP_L23VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L5_VIP:end_V1_L5_VIP,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L5VIP_L6VIP)

            ##################################################################################################################################################

            ## L6 --> rest
            ## Pyr --> SST
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L6Pyr_L4SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L6Pyr_L23SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L6Pyr_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L6Pyr_L6SST)

            ## Pyr --> VIP
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L6Pyr_L4VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L6Pyr_L23VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L6Pyr_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L6Pyr_L6VIP)

            ## Pyr --> Pyr
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L6Pyr_L4Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L6Pyr_L23Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_Pyr:end_V1_L6_Pyr,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L6Pyr_L5Pyr)

            ## SST --> Pyr
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L6SST_L4Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L6SST_L23Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L6SST_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L6SST_L6Pyr)

            ## SST --> VIP
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L6SST_L4VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L6SST_L23VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L6SST_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L6_VIP:end_V1_L6_VIP].T.mul_(self.BBV1L6SST_L6VIP)

            ## SST --> SST
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L6SST_L4SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L6SST_L23SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_SST:end_V1_L6_SST,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L6SST_L5SST)

            ## VIP --> Pyr
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L4_Pyr:end_V1_L4_Pyr].T.mul_(self.BBV1L6VIP_L4Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L23_Pyr:end_V1_L23_Pyr].T.mul_(self.BBV1L6VIP_L23Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L5_Pyr:end_V1_L5_Pyr].T.mul_(self.BBV1L6VIP_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L6_Pyr:end_V1_L6_Pyr].T.mul_(self.BBV1L6VIP_L6Pyr)
            #
            # ## VIP --> SST
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L4_SST:end_V1_L4_SST].T.mul_(self.BBV1L6VIP_L4SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L23_SST:end_V1_L23_SST].T.mul_(self.BBV1L6VIP_L23SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L5_SST:end_V1_L5_SST].T.mul_(self.BBV1L6VIP_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L6_SST:end_V1_L6_SST].T.mul_(self.BBV1L6VIP_L6SST)
            #
            # ## VIP --> VIP
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L4_VIP:end_V1_L4_VIP].T.mul_(self.BBV1L6VIP_L4VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L23_VIP:end_V1_L23_VIP].T.mul_(self.BBV1L6VIP_L23VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_V1_L6_VIP:end_V1_L6_VIP,start_V1_L5_VIP:end_V1_L5_VIP].T.mul_(self.BBV1L6VIP_L5VIP)

            ##################################################################################################################################################
            ##################################################################################################################################################

            ## LM - ##LM
            ## L4 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML4Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML4Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML4Pyr_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML4Pyr_L6SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML4Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML4Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML4Pyr_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML4Pyr_L6VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML4Pyr_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML4Pyr_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_Pyr:end_LM_L4_Pyr,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML4Pyr_L6Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML4SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML4SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML4SST_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML4SST_L6Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML4SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML4SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML4SST_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML4SST_L6VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML4SST_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML4SST_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_SST:end_LM_L4_SST,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML4SST_L6SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML4VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML4VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML4VIP_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML4VIP_L6Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML4VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML4VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML4VIP_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML4VIP_L6SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML4VIP_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML4VIP_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L4_VIP:end_LM_L4_VIP,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML4VIP_L6VIP)

            ##################################################################################################################################################

            ## L23 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML23Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML23Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML23Pyr_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML23Pyr_L6SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML23Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML23Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML23Pyr_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML23Pyr_L6VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML23Pyr_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML23Pyr_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_Pyr:end_LM_L23_Pyr,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML23Pyr_L6Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML23SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML23SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML23SST_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML23SST_L6Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML23SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML23SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML23SST_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML23SST_L6VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML23SST_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML23SST_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_SST:end_LM_L23_SST,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML23SST_L6SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML23VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML23VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML23VIP_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML23VIP_L6Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML23VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML23VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML23VIP_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML23VIP_L6SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML23VIP_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML23VIP_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L23_VIP:end_LM_L23_VIP,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML23VIP_L6VIP)

            ##################################################################################################################################################

            ## L5 --> rest
            ## Pyr --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML5Pyr_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML5Pyr_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML5Pyr_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML5Pyr_L6SST)

            ## Pyr --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML5Pyr_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML5Pyr_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML5Pyr_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML5Pyr_L6VIP)

            ## Pyr --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML5Pyr_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML5Pyr_L23Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_Pyr:end_LM_L5_Pyr,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML5Pyr_L6Pyr)

            ## SST --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML5SST_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML5SST_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML5SST_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML5SST_L6Pyr)

            ## SST --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML5SST_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML5SST_L23VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML5SST_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML5SST_L6VIP)

            ## SST --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML5SST_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML5SST_L23SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_SST:end_LM_L5_SST,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML5SST_L6SST)

            ## VIP --> Pyr
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML5VIP_L4Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML5VIP_L23Pyr)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML5VIP_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML5VIP_L6Pyr)

            ## VIP --> SST
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML5VIP_L4SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML5VIP_L23SST)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML5VIP_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML5VIP_L6SST)

            ## VIP --> VIP
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML5VIP_L4VIP)
            self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML5VIP_L23VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L5_VIP:end_LM_L5_VIP,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML5VIP_L6VIP)

            ##################################################################################################################################################

            ## L6 --> rest
            ## Pyr --> SST
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML6Pyr_L4SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML6Pyr_L23SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML6Pyr_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML6Pyr_L6SST)

            ## Pyr --> VIP
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML6Pyr_L4VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML6Pyr_L23VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML6Pyr_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML6Pyr_L6VIP)

            ## Pyr --> Pyr
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML6Pyr_L4Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML6Pyr_L23Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_Pyr:end_LM_L6_Pyr,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML6Pyr_L5Pyr)

            ## SST --> Pyr
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML6SST_L4Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML6SST_L23Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML6SST_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML6SST_L6Pyr)

            ## SST --> VIP
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML6SST_L4VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML6SST_L23VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML6SST_L5VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L6_VIP:end_LM_L6_VIP].T.mul_(self.BBLML6SST_L6VIP)

            ## SST --> SST
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML6SST_L4SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML6SST_L23SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_SST:end_LM_L6_SST,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML6SST_L5SST)

            ## VIP --> Pyr
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L4_Pyr:end_LM_L4_Pyr].T.mul_(self.BBLML6VIP_L4Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L23_Pyr:end_LM_L23_Pyr].T.mul_(self.BBLML6VIP_L23Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L5_Pyr:end_LM_L5_Pyr].T.mul_(self.BBLML6VIP_L5Pyr)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L6_Pyr:end_LM_L6_Pyr].T.mul_(self.BBLML6VIP_L6Pyr)
            #
            # ## VIP --> SST
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L4_SST:end_LM_L4_SST].T.mul_(self.BBLML6VIP_L4SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L23_SST:end_LM_L23_SST].T.mul_(self.BBLML6VIP_L23SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L5_SST:end_LM_L5_SST].T.mul_(self.BBLML6VIP_L5SST)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L6_SST:end_LM_L6_SST].T.mul_(self.BBLML6VIP_L6SST)
            #
            # ## VIP --> VIP
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L4_VIP:end_LM_L4_VIP].T.mul_(self.BBLML6VIP_L4VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L23_VIP:end_LM_L23_VIP].T.mul_(self.BBLML6VIP_L23VIP)
            # self.fullRNN._parameters['weight_hh_l0'].data[start_LM_L6_VIP:end_LM_L6_VIP,start_LM_L5_VIP:end_LM_L5_VIP].T.mul_(self.BBLML6VIP_L5VIP)

            #########################################################################################

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

            # V1_L6_Pyr[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L6_Pyr:end_V1_L6_Pyr])
            # V1_L6_SST[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L6_SST:end_V1_L6_SST])
            # V1_L6_VIP[:,ii,:] = torch.squeeze(xFull[:,:,start_V1_L6_VIP:end_V1_L6_VIP])

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

            # LM_L6_Pyr[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L6_Pyr:end_LM_L6_Pyr])
            # LM_L6_SST[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L6_SST:end_LM_L6_SST])
            # LM_L6_VIP[:,ii,:] = torch.squeeze(xFull[:,:,start_LM_L6_VIP:end_LM_L6_VIP])

            ## Final projections at V1
            pred_V1_L4_Pyr[:,ii,:] = self.pred_V1_L4_Pyr(torch.squeeze(hiddenFull[:,:,start_V1_L4_Pyr:end_V1_L4_Pyr]))
            pred_V1_L4_SST[:,ii,:] = self.pred_V1_L4_SST(torch.squeeze(hiddenFull[:,:,start_V1_L4_SST:end_V1_L4_SST]))
            pred_V1_L4_VIP[:,ii,:] = self.pred_V1_L4_VIP(torch.squeeze(hiddenFull[:,:,start_V1_L4_VIP:end_V1_L4_VIP]))

            pred_V1_L23_Pyr[:,ii,:] = self.pred_V1_L23_Pyr(torch.squeeze(hiddenFull[:,:,start_V1_L23_Pyr:end_V1_L23_Pyr]))
            pred_V1_L23_SST[:,ii,:] = self.pred_V1_L23_SST(torch.squeeze(hiddenFull[:,:,start_V1_L23_SST:end_V1_L23_SST]))
            pred_V1_L23_VIP[:,ii,:] = self.pred_V1_L23_VIP(torch.squeeze(hiddenFull[:,:,start_V1_L23_VIP:end_V1_L23_VIP]))

            pred_V1_L5_Pyr[:,ii,:] = self.pred_V1_L5_Pyr(torch.squeeze(hiddenFull[:,:,start_V1_L5_Pyr:end_V1_L5_Pyr]))
            pred_V1_L5_SST[:,ii,:] = self.pred_V1_L5_SST(torch.squeeze(hiddenFull[:,:,start_V1_L5_SST:end_V1_L5_SST]))
            pred_V1_L5_VIP[:,ii,:] = self.pred_V1_L5_VIP(torch.squeeze(hiddenFull[:,:,start_V1_L5_VIP:end_V1_L5_VIP]))

            # pred_V1_L6_Pyr[:,ii,:] = self.pred_V1_L6_Pyr(torch.squeeze(hiddenFull[:,:,start_V1_L6_Pyr:end_V1_L6_Pyr]))
            # pred_V1_L6_SST[:,ii,:] = self.pred_V1_L6_SST(torch.squeeze(hiddenFull[:,:,start_V1_L6_SST:end_V1_L6_SST]))
            # pred_V1_L6_VIP[:,ii,:] = self.pred_V1_L6_VIP(torch.squeeze(hiddenFull[:,:,start_V1_L6_VIP:end_V1_L6_VIP]))

            ## Final projections at LM
            pred_LM_L4_Pyr[:,ii,:] = self.pred_LM_L4_Pyr(torch.squeeze(hiddenFull[:,:,start_LM_L4_Pyr:end_LM_L4_Pyr]))
            pred_LM_L4_SST[:,ii,:] = self.pred_LM_L4_SST(torch.squeeze(hiddenFull[:,:,start_LM_L4_SST:end_LM_L4_SST]))
            pred_LM_L4_VIP[:,ii,:] = self.pred_LM_L4_VIP(torch.squeeze(hiddenFull[:,:,start_LM_L4_VIP:end_LM_L4_VIP]))

            pred_LM_L23_Pyr[:,ii,:] = self.pred_LM_L23_Pyr(torch.squeeze(hiddenFull[:,:,start_LM_L23_Pyr:end_LM_L23_Pyr]))
            pred_LM_L23_SST[:,ii,:] = self.pred_LM_L23_SST(torch.squeeze(hiddenFull[:,:,start_LM_L23_SST:end_LM_L23_SST]))
            pred_LM_L23_VIP[:,ii,:] = self.pred_LM_L23_VIP(torch.squeeze(hiddenFull[:,:,start_LM_L23_VIP:end_LM_L23_VIP]))

            pred_LM_L5_Pyr[:,ii,:] = self.pred_LM_L5_Pyr(torch.squeeze(hiddenFull[:,:,start_LM_L5_Pyr:end_LM_L5_Pyr]))
            pred_LM_L5_SST[:,ii,:] = self.pred_LM_L5_SST(torch.squeeze(hiddenFull[:,:,start_LM_L5_SST:end_LM_L5_SST]))
            pred_LM_L5_VIP[:,ii,:] = self.pred_LM_L5_VIP(torch.squeeze(hiddenFull[:,:,start_LM_L5_VIP:end_LM_L5_VIP]))

            # pred_LM_L6_Pyr[:,ii,:] = self.pred_LM_L6_Pyr(torch.squeeze(hiddenFull[:,:,start_LM_L6_Pyr:end_LM_L6_Pyr]))
            # pred_LM_L6_SST[:,ii,:] = self.pred_LM_L6_SST(torch.squeeze(hiddenFull[:,:,start_LM_L6_SST:end_LM_L6_SST]))
            # pred_LM_L6_VIP[:,ii,:] = self.pred_LM_L6_VIP(torch.squeeze(hiddenFull[:,:,start_LM_L6_VIP:end_LM_L6_VIP]))

        ## would need to add tuples if including computations with L6
        return (pred_V1_L4_Pyr, pred_V1_L4_SST, pred_V1_L4_VIP), (pred_V1_L23_Pyr, pred_V1_L23_SST, pred_V1_L23_VIP), (pred_V1_L5_Pyr, pred_V1_L5_SST, pred_V1_L5_VIP), (pred_LM_L4_Pyr, pred_LM_L4_SST, pred_LM_L4_VIP), (pred_LM_L23_Pyr, pred_LM_L23_SST, pred_LM_L23_VIP), (pred_LM_L5_Pyr, pred_LM_L5_SST, pred_LM_L5_VIP),
        (V1_L4_Pyr, V1_L4_SST, V1_L4_VIP), (V1_L23_Pyr, V1_L23_SST, V1_L23_VIP), (V1_L5_Pyr, V1_L5_SST, V1_L5_VIP), (V1_L6_Pyr, V1_L6_SST, V1_L6_VIP), (LM_L4_Pyr, LM_L4_SST, LM_L4_VIP), (LM_L23_Pyr, LM_L23_SST, LM_L23_VIP), (LM_L5_Pyr, LM_L5_SST, LM_L5_VIP)

#create the NN
model = celltypeRNN()
