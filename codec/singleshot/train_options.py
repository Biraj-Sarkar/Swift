import argparse

parser = argparse.ArgumentParser()

######## Data ########
parser.add_argument('--train', required=True, type=str, help='Path to training data.')
parser.add_argument('--patch', default=64, type=int, help='Patch size.')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1, help='Batch size for evaluation.')
parser.add_argument('--num-crops', type=int, default=2, help='# training crops per example.')
parser.add_argument('--train-mv', type=str, help='Path to motion vectors (unused in this version).')

######## Model ########
parser.add_argument('--iterations', type=int, default=10, help='# iterations of progressive encoding.')
parser.add_argument('--bits', default=16, type=int, help='Bottle neck size.')
parser.add_argument('--decoder-heads', default=5, type=int, help='Number of decoder heads (quality levels).')

######## Learning ########
parser.add_argument('--max-train-iters', type=int, default=100000, help='Max training iterations.')
parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate.')
parser.add_argument('--clip', type=float, default=0.5, help='Gradient clipping.')
parser.add_argument('--schedule', default='50000,75000', type=str, help='Schedule milestones.')
parser.add_argument('--gamma', type=float, default=0.5, help='LR decay factor.')
parser.add_argument('--checkpoint-iters', type=int, default=5000, help='Model checkpoint period.')

######## Experiment ########
parser.add_argument('--model-dir', type=str, default='model_new_singleshot', help='Path to model folder.')
parser.add_argument('--save-model-name', type=str, default='new_ss', help='Checkpoint name to save.')
