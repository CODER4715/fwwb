from flask import Flask, render_template, request, send_from_directory
import torch
from resnet import ResNet
import os
import random
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from getdata import get_data
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

def load_model():
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global model
    model = ResNet(3)
    model.load_state_dict(torch.load('best.pt', map_location=torch.device('cpu')))

    model.eval()
    if device == 'cuda':
        model.to(device)

def evaluate(model, loader):
    model.eval()
    res = pd.DataFrame()
    for x, y in loader:
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
    return res

batchsz = 1000

app = Flask(__name__)
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.xlsx', '.tsv']
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024**3
app.config['UPLOAD_FOLDER'] = './upload_file'

load_model()

@app.route('/', methods=['Get', 'Post'])
def index():
    return render_template('index.html')




@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    json = {}
    result = pd.DataFrame()
    uploaded_file = request.files['file']
    filename = uploaded_file.filename
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return "bad file extension"

    # Ensure an image was properly uploaded to our endpoint.
    if request.method == 'POST':
        if request.files.get('file'):
            # Read the image in PIL format

            id = random.randint(0, 999999)
            while id in os.listdir(app.config['UPLOAD_FOLDER']):
                id = random.randint(0, 999999)
            path = app.config['UPLOAD_FOLDER'] + '/' + str(id)
            if not os.path.exists(path):  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(path)

            request.files['file'].save(os.path.join(path, uploaded_file.filename))
            db = get_data(os.path.join(path, uploaded_file.filename), filetype=file_ext)
            if not db.is_ok:
                return "bad input"

            dat_loader = DataLoader(db, batch_size=batchsz, shuffle=False, pin_memory=True)
            res = evaluate(model, dat_loader)
            price = res[res['PREDICT'] == 1]
            volume = res[res['PREDICT'] == 2]
            result = pd.concat([price, volume])
            # result['PREDICT'] = result['PREDICT'].map(class_code)
            result[['USER_ID','ITEM_ID']]=result[['USER_ID','ITEM_ID']].astype('int64')
            result.to_csv(path_or_buf=os.path.join(path, 'result_'+str(id)+'.csv'), index=False)
            result = shuffle(result)
            result = result.reset_index(drop=True)
            del db, dat_loader, price, volume
            
    result['index'] = result.index + 1
    result = result.iloc[0:500]
    return render_template('result.html', data=result.values, pid=id)

@app.route("/download/<path:id>/", methods=["POST","Get"])
def download(id):
    path = app.config['UPLOAD_FOLDER'] + '/' + str(id)+'/result_'+str(id)+'.csv'
    name = 'result_'+str(id)+'.csv'
    return send_from_directory(directory='./upload_file/'+str(id), path=path, filename=name, as_attachment=True)

if __name__ == '__main__':
    load_model()
    app.run(port=5050)
