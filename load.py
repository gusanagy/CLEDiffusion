import os
import torch

pretrained_path = ckpt/lol.pt

#Erro ao carregar o state dict e o modelo
    #Testar com os checkpoints dados pelo cara
        ckpt = torch.load(os.path.join(
                pretrained_path), map_location='cpu')
        checkpoint = torch.load({k.replace('module.', ''): v for k, v in ckpt.items()})
        state_dict = checkpoint['state_dict']
        state_dict['conv1.weight'][:, :10, :, :] = state_dict['conv1.weight']
        state_dict['conv1.weight'][:, 10, :, :] = torch.zeros_like(state_dict['conv1.weight'][:, 0, :, :])
        state_dict['conv1.in_channels'] = 11