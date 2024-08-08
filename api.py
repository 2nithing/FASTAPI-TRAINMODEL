from fastapi import FastAPI,UploadFile
from typing import List
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import numpy as np
import re
import training


app = FastAPI()

@app.post('/')
async def train(files:List[UploadFile]):
    # files = await files.read()
    data=[]
    for file in files:
        if file.filename!='labels.txt':
            img = await file.read()
            img = Image.open(io.BytesIO(img))
            img = img.resize((160,160))
            img = img_to_array(img)
            img = img/255
            data.append(img)
        else:
            label = await file.read()                
            label = re.findall(r'\d+', str(label))
    if data:
        model = training.train(data,label)      
        print("model trained")
    return {'result':np.array(data).shape,'labels':label}