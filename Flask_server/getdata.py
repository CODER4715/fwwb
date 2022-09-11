import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class get_data(Dataset):

    def __init__(self, path, filetype='.csv'):
        super(get_data, self).__init__()

        self.file = path
        self.is_ok = True
        try:
            self.data, self.infos = self.load_file(filetype)
        except:
            self.is_ok = False

    def load_file(self, filetype):
        try:
            with open(self.file, errors='ignore') as f:
                if filetype == '.csv':
                    dataset = pd.read_csv(f, sep=',')
                elif filetype == '.tsv':
                    dataset = pd.read_csv(f, sep='\t')
                elif filetype == '.xlsx':
                    dataset = pd.read_excel()
            data = dataset[['ITEM_PRICE', 'ITEM_SALES_VOLUME', 'NAME_LEN', 'CATE_NAME_LV1', 'EVAL/VOLUME',
                            'FAV/VOLUME', 'SHOP_SALES_VOLUME', 'SHOP_SALES_AMOUNT', 'SHOP_DELIVERY_PROVINCE'
                , 'SCORE', 'OPEN_YEAR', 'byxs']]
        except:

            return 'error'

        data = np.array(data).tolist()
        infos = dataset[['USER_ID', 'ITEM_ID']]
        infos = np.array(infos).tolist()

        del dataset
        assert len(data) == len(infos)

        return data, infos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # idx~[0~len(data)]
        dat, info = self.data[idx], self.infos[idx]

        dat = torch.tensor(dat)
        dat = dat.reshape(1, 12)

        return dat, info
