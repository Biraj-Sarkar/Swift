import torch
from torch.autograd import Variable
import torch.nn as nn

def prepare_batch(batch):
    return batch - 0.5

def prepare_inputs(crops, args):
    data_arr = []
    for crop_idx, data in enumerate(crops):
        patches = Variable(data.cuda())
        res = prepare_batch(patches)
        data_arr.append(res)
    return torch.cat(data_arr, dim=0)

def init_lstm(batch_size, height, width, args):
    encoder_h_1 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda(),
                   Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda())
    encoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda(),
                   Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda())
    encoder_h_3 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda(),
                   Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda())

    return (encoder_h_1, encoder_h_2, encoder_h_3, None, None, None, None)

def init_decoder_states(batch_size, height, width, args):
    decoder_h_1 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda(),
                   Variable(torch.zeros(batch_size, 512, height // 16, width // 16)).cuda())
    decoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda(),
                   Variable(torch.zeros(batch_size, 512, height // 8, width // 8)).cuda())
    decoder_h_3 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda(),
                   Variable(torch.zeros(batch_size, 256, height // 4, width // 4)).cuda())
    decoder_h_4 = (Variable(torch.zeros(batch_size, 128, height // 2, width // 2)).cuda(),
                   Variable(torch.zeros(batch_size, 128, height // 2, width // 2)).cuda())

    return (decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
