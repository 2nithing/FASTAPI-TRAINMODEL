from fastapi import FastAPI,UploadFile
from typing import List
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.saving import load_model
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
        model.save('model.keras')
    return {'result':'model trained'}

@app.post('/test')
async def testing(file:UploadFile):
    img = await file.read()
    img = Image.open(io.BytesIO(img))
    img = img.resize((160,160))
    img = np.array(img)
    img = img/255
    img = img.reshape(1,160,160,3)
    model = load_model('model.keras')
    result = model.predict(img)
    print (result)
    if np.argmax(result)==0:
        response='cat'
    elif np.argmax(result)==1:
        response = 'dog'
    return {'result':response}
