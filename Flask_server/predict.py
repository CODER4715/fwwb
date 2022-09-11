import pandas as pd
import torch
import time
from torch.utils.data import DataLoader
from getdata import get_data
from resnet import ResNet
import gc



batchsz = 50000
device = torch.device('cuda')


def evaluate(model, loader):
    model.eval()
    res = pd.DataFrame()
    for x, y in loader:
        torch.cuda.empty_cache()
        x = x.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)

        pred = pred.cpu()
        result = pd.DataFrame(columns=['PREDICT'], data=pred)
        usr = pd.DataFrame(columns=['USER_ID'], data=y[0])
        id = pd.DataFrame(columns=['ITEM_ID'], data=y[1])
        info = pd.concat([usr, id], axis=1)
        result = pd.concat([info, result], axis=1)
        res = res.append(result)
        # res = pd.concat([res,result],axis=0)
    return res


def main():
    sta = time.time()
    model = ResNet(3).to(device)
    model.load_state_dict(torch.load('best.pt'))
    for month in range(6, 10):
        st = time.time()
        db = get_data(month)
        dat_loader = DataLoader(db, batch_size=batchsz, shuffle=False, pin_memory=True, num_workers=1)

        res = evaluate(model, dat_loader)
        price = res[res['PREDICT'] == 1]
        volume = res[res['PREDICT'] == 2]
        ed = time.time()
        print('处理%d月共耗时(秒):' % month + str(ed - st))
        print(price.info())
        print(volume.info())
        price.to_csv(path_or_buf='./price_error/price' + str(month) + '.csv', index=False)
        volume.to_csv(path_or_buf='./volume_error/volume' + str(month) + '.csv', index=False)
        del db, dat_loader
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)
    ed = time.time()
    print('处理共耗时(秒):' + str(ed - sta))


if __name__ == '__main__':
    main()
