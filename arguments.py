import argparse
parser = argparse.ArgumentParser(description='Proposed_AutoEncoder')
#Change self.test = 0 if you want to train

parser.add_argument('--feature-size', type=int, default=1024, help='size of pre-extracted feature (default = 1024)')
parser.add_argument('--rgb_list', type=str, default='list/shanghai-i3d-train-10crop.list', help='list of train files')
parser.add_argument('--test_rgb_list', type=str, default='list/shanghai-i3d-test-10crop.list', help='list of test files')
parser.add_argument('--gt', type=str, default='list/gt-sh.npy', help='ground truth file')
parser.add_argument('--batch_size', type=int, default=60, help='batch size for training')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--dataset', type=str, default='shanghai', help='dataset name')
parser.add_argument('--datafolder', type=str, default='data', help='data folder')
parser.add_argument('--error_analysis', type=int, default=0, help='save the result wrong prediction while testing')
parser.add_argument('--modelpath', type=str, default='best_proposed_model.pkl', help='path to trained model which use for testing')
parser.add_argument('--max-epoch', type=int, default=10000, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--test', type=int, default=0, help='use for just testing')

