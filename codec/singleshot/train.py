import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from torch.autograd import Variable

from dataset import get_loader
from train_options import parser
import model
from util import init_lstm, init_decoder_states, prepare_inputs

args = parser.parse_args()
print(args)

############### Data ###############
# Assuming standard single-shot data loading
train_loader = get_loader(
    is_train=True,
    root=args.train, mv_dir=args.train_mv,
    args=args
)

############### Model ###############
encoder = network.EncoderCell().cuda()
binarizer = network.Binarizer(args.bits).cuda()
decoder = network.DecoderCell(
    bits=args.bits,
    iterations=args.iterations,
    num_heads=args.decoder_heads).cuda()

nets = [encoder, binarizer, decoder]
params = [{'params': net.parameters()} for net in nets]

solver = optim.Adam(params, lr=args.lr)
milestones = [int(s) for s in args.schedule.split(',')]
scheduler = LS.MultiStepLR(solver, milestones=milestones, gamma=args.gamma)

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

############### Training ###############
train_iter = 0

while True:
    for batch, (crops, _, _) in enumerate(train_loader):
        scheduler.step()
        train_iter += 1

        if train_iter > args.max_train_iters:
            break

        solver.zero_grad()

        # Init encoder states
        (encoder_h_1, encoder_h_2, encoder_h_3, _, _, _, _) = init_lstm(
            batch_size=(crops[0].size(0) * args.num_crops), height=crops[0].size(2),
            width=crops[0].size(3), args=args)

        in_img = prepare_inputs(crops, args)
        batch_size, _, height, width = in_img.size()
        h, w = height // 16, width // 16

        code_arr = []
        for i in range(args.iterations):
            # Encode
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                in_img, encoder_h_1, encoder_h_2, encoder_h_3)

            # Binarize
            codes = binarizer(encoded)
            code_arr.append(codes)

        # Stack codes for decoder: [batch, iterations * bits, h, w]
        stacked_codes = torch.cat(code_arr, dim=1)

        total_loss = 0

        # Train with multi-headed decoder and early exits (5 quality levels)
        # We can optimize all quality levels simultaneously
        for q in range(1, 6):
            (decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_decoder_states(
                batch_size, height, width, args)

            decoded_frame, _, _, _, _ = decoder(
                stacked_codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4, quality_level=q)

            loss = (in_img - decoded_frame).abs().mean()
            total_loss += loss

        total_loss.backward()

        for net in nets:
            torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)

        solver.step()

        if train_iter % 10 == 0:
            print('[TRAIN] Iter[{}]; LR: {}; Loss: {:.6f}'.format(
                train_iter, scheduler.get_lr()[0], total_loss.item() / 5.0))

        if train_iter % args.checkpoint_iters == 0:
            for i, name in enumerate(['encoder', 'binarizer', 'decoder']):
                torch.save(nets[i].state_dict(),
                           '{}/{}_{}_{:08d}.pth'.format(
                               args.model_dir, args.save_model_name, name, train_iter))

    if train_iter > args.max_train_iters:
        break
