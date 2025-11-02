from args import get_args
import os
import pandas as pd
from datasets import knee_xray_dataset
from torch.utils.data import DataLoader
from models import MyModel
from trainer import train_model

def main():
    # 1. we need some arguments
    args = get_args()

    # 2. iterate among the folds
    for fold in range(5):
        print('Training fold: ', fold)

        train_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_train.csv'.format(str(fold))))
        val_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_val.csv'.format(str(fold))))

        # 3. prepare datasets
        train_dataset = knee_xray_dataset(train_set)
        val_dataset =  knee_xray_dataset(val_set)

        # 4. create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # 5. initialize the model
        model = MyModel(args.backbone)

        # 6. train the model
        train_model(model, train_loader, val_loader, fold)

        print()



if __name__ == '__main__':
    main()