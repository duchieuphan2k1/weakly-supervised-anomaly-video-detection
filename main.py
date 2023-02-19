import torch
from torch.utils.data import DataLoader
from config import Config
from data_loader import Dataset
from model import AutoEncoder
import arguments
from testing import test

if __name__ == '__main__':
    args = arguments.parser.parse_args()
    config = Config(args)
    
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    model = AutoEncoder(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if args.test == 1:
        model.load_state_dict(torch.load(args.modelpath))
        auc = test(test_loader, model, args, None, device)