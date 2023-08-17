import torch
from clonaltrans.bootstrap import Bootstrapping, ProfileLikelihood
import sys 

if __name__ == '__main__':
    if False:
        print (f'***** Bootstrapping trails *****')
        model_ori = torch.load('./data/V5_Mingze_BG/models/K2Pro/const_var.pt')
        print (f'Starting index is {sys.argv[1]}')
        boots = Bootstrapping(model_ori, int(sys.argv[1]))
        boots.num_gpus = 20
        boots.bootstart(1000)

    else:
        print (f'***** Profile likelihood trails *****')
        model_ori = torch.load('./data/V5_Mingze_BG/models/K2Pro/const_var.pt')
        boots = ProfileLikelihood(model_ori, './data/V5_Mingze_BG/models/K2Pro/const_var.pt')
        boots.num_gpus = int(sys.argv[1])
        boots.profilestart()