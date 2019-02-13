import argparse

from os.path import join, exists
from os import makedirs, environ
from time import localtime, strftime

time = strftime("%y_%m_%d-%H_%M_%S", localtime())
parser = argparse.ArgumentParser(description="Capsule Network")

####################################
#       Environment Setting        #
####################################
parser.add_argument('-d', '--dataset', default='org-dataset', type=str,
                    help="The directory path where the original dataset is stored")
parser.add_argument('--root_dir', default='.', type=str,
                    help="The directory path where all the computed data is stored")
parser.add_argument('--num_threads', default=8, type=int,
                    help="The number of threads for enqueueing examples")
parser.add_argument('--gpu_id', default=2, type=int,
                    help="The gpu  id on which the program should run")
parser.add_argument('-r', '--restore_model', action='store_true',
                    help="Restore a model from previous version")
parser.add_argument('-w', '--weights', default='model_19_01_25-18_10_09.h5', type=str,
                    help="Load specific weights to the model")
parser.add_argument('--debug', action='store_true', help="Debug the program")

####################################
#     Hyper Parameter Setting      #
####################################
parser.add_argument('--epochs', default=200, type=int,
                    help="The number of epochs to train the network")
parser.add_argument('--batch_size', default=1, type=int,
                    help="The maximum number of examples used to train at a time")
parser.add_argument('--lr', default=0.0001, type=float,
                    help="Initial learning rate")
parser.add_argument('--recon_wei', type=float, default=0.5,
                    help="The coefficient (weighting) for the loss of decoder")
parser.add_argument('--lr_decay', default=0.9, type=float,
                    help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
parser.add_argument('--m_plus', default=1.0, type=float, help="M Plus")
parser.add_argument('--m_minus', default=0.9, type=float, help="M Minus")
parser.add_argument('--lam', default=0.5, type=float, help="Lambda for Down Weighting")
parser.add_argument('--routings', default=2, type=int,
                    help="Number of iterations used in routing algorithm. Should > 0")
parser.add_argument('--shift_fraction', default=0.1, type=float,
                    help="Fraction of pixels to shift at most in each direction")
parser.add_argument('--patch_size', default=10, type=int, help="The size of each patch of the image to break into")
parser.add_argument('--num_class', default=2, type=int, help="The number of classes to segment into")

args = parser.parse_args()


args.time = time
args.logs_dir = join(args.root_dir, 'logs')  # Path to store various logs during training stage
args.models_dir = join(args.root_dir, 'models')  # Path to store the saved models
args.weights_dir = join(args.root_dir, 'weights')  # Path to store weights calculated during training
args.layers_dir = join(args.root_dir, 'layers')  # Path to store the output of each layers
args.tests_dir = join(args.root_dir, 'tests')  # Path to store predicted and reconstructed values
args.patch_dataset_dir = join(args.root_dir, 'patch-dataset')  # Path to store the patched dataset

directories = ["models_dir", "logs_dir", "weights_dir", "layers_dir", "patch_dataset_dir", "tests_dir"]
for d in directories:
    if not exists(str(getattr(args, d))):
        makedirs(str(getattr(args, d)))

if args.debug:
    print("=" * 42)
    print("ARGUMENTS".ljust(20) + "  " + "VALUES")
    print("=" * 20 + "  " + "=" * 20)
    for arg in vars(args):
        print(str(arg).ljust(20) + " |" + str(getattr(args, arg)))
    print("=" * 42)
