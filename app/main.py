import torch
import uvicorn
from uvicorn import Config, Server
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel# o validate the input and output of the RESTful API, we can define the schema in FastAPI with pydantic, which will be used to generate the OpenAPI docs and ReDoc automatically.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from PIL import Image
import io
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import numpy as np



app = FastAPI(title='Deploying a ML Model with FastAP')


# Define class for image requests
#me: If a user passes something that is incorrect, a response is returned to the user letting them know that the request could not be processed due to a data validation error
# # Expected input
class ImageRequest(BaseModel):
    file_name: str
    file_content: UploadFile







@app.get("/")  #this part only from  https://colab.research.google.com/github/aasimsani/model-quick-deploy/blob/main/Model_Quick_Deploy.ipynb#scrollTo=EIQ3BGWgNAe6
async def main():
    """Create a basic home page to upload a file

    :return: HTML for homepage
    :rtype: HTMLResponse
    """

    content = """<body>
          <h3>Upload an image to get ....</h3>
          <form action="/predict" enctype="multipart/form-data" method="post">
              <input name="files" type="file" multiple>
              <input type="submit">
          </form>
      </body>
      """
    return HTMLResponse(content=content)




from PIL import Image
import io
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors #me: annotator uses opencv like cv2.rectangle, cv2.draw.text,......
model = YOLO("yolov8n-seg.pt")  # load an official model (seg+box)

#?? loading model before receiving an image???  https://github.com/chaklam-silpasuwanchai/Python-for-Data-Science/tree/master/Code/Appendix%20-%20Deployment/01-FastAPI%2BDocker
#@app.on_event("startup")  #To add a function that should be run before the application starts, declare it with the event "startup":
 #      async def startup_event():  #things like connecting to a database, loading model or configuration settings, or setting up background tasks.   




#1- async type:
# Define API endpoint for image classification
@app.post("/predict/")
async def predict(files: UploadFile = File(...)):  
        # first, VALIDATE INPUT FILE
         filename = files.filename
         fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
         if not fileExtension:
             raise HTTPException(status_code=415, detail="Unsupported file provided.")   
        image = await files.read()
        image = Image.open(io.BytesIO(image)).convert('RGB')
        image=image.resize((640,448), Image.LANCZOS)#resize it to be the same size of the model's output

        results = model(image)
        idx=((results[0].to("cpu").numpy().boxes.cls).astype(int)) #labels' idx
        masks=np.array(results[0].masks.data)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image = np.asarray(image)
        annotator = Annotator(image)#$$$$ we should annotator again due to the last change:   annotator=Image.fromarray(annotator.result())
        img_gpu = torch.as_tensor(image, dtype=torch.float16, device=results[0].masks.data.device).permute(
                2, 0, 1).flip(0).contiguous() / 255
        annotator.masks(results[0].masks.data, colors=[colors(x) for x in range(len(idx))], im_gpu=img_gpu)
        #annotator.masks(results[0].masks.data, colors=[colors(x, True) for x in range(len(idx))], im_gpu=img_gpu)
        #                                                         True: i think for bgr or rgb <-- bgr=False
        #annotator=Image.fromarray(annotator.result())# result() is different than results[0]
        #annotator.show()


        cv2.imwrite("cv_sil.jpg", cv2.cvtColor(annotator.result(), cv2.COLOR_RGB2BGR)) #????cv2.imencode
        new_image=open("cv_sil.jpg",mode="rb")#read the image as a binary
        return StreamingResponse(new_image,media_type="image/jpeg") #send binay image to the site
        

        






 

# Start the app in normal deployment
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

#  !uvicorn main:app --port 8000 --host 0.0.0.0






