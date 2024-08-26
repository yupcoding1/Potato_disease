from fastapi import FastAPI,UploadFile,File
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = FastAPI()
MODEL = tf.keras.models.load_model('../models/1.keras')
CLASS_NAMES=["Early Blight", "Late Blight","Healthy"]
def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image


@app.get("/ping")
async def ping():
    return "this is testing"

@app.post("/predict")
async def predict(
        file : UploadFile = File(...)
):
     image = read_file_as_image(await file.read())
     image_expanded=np.expand_dims(image,0)
     predicted=MODEL.predict(image_expanded)
     dis=CLASS_NAMES[np.argmax(predicted[0])]
     confidence=max(predicted[0])
     return{
         'class':dis,
         'confidence':float(confidence)
     }


if __name__=="__main__":
  uvicorn.run(app,host='localhost',port=8000)
