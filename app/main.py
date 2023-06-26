import torch
import uvicorn
from uvicorn import Config, Server
from fastapi import FastAPI, File, UploadFile, HTTPException
import pydantic
from pydantic import BaseModel# 
from PIL import Image
import io
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors #me: annotator uses opencv like cv2.rectangle, cv2.draw.text,......

app = FastAPI(title='Deploying a ML Model with FastAP')


# Define class for image requests
class ImageRequest(BaseModel):
    file_name: str
    file_content: UploadFile







@app.get("/")  
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


model = YOLO("yolov8n-seg.pt") 

@app.post("/predict/")
async def predict(files: UploadFile = File(...)):  
        # # first, VALIDATE INPUT FILE
        filename = files.filename
        fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
        if not fileExtension:
            raise HTTPException(status_code=415, detail="Unsupported file provided.")  
        image = await files.read()
        image = Image.open(io.BytesIO(image)).convert('RGB')
        image=image.resize((640,448), Image.LANCZOS)#resize it to be the same size of the model's output

        results = model(image)
        idx=((results[0].to("cpu").numpy().boxes.cls).astype(int)) #labels' idx
        masks=np.array(results[0].masks.data)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image = np.array(image)
        annotator = Annotator(image)
        img_gpu = torch.as_tensor(image, dtype=torch.float16, device=results[0].masks.data.device).permute(
                2, 0, 1).flip(0).contiguous() / 255
        annotator.masks(results[0].masks.data, colors=[colors(x) for x in range(len(idx))], im_gpu=img_gpu)
      


        cv2.imwrite("cv_sil.jpg", cv2.cvtColor(annotator.result(), cv2.COLOR_RGB2BGR)) 
        new_image=open("cv_sil.jpg",mode="rb")
        return StreamingResponse(new_image,media_type="image/jpeg") 
        

        






 

# Start the app in normal deployment
if __name__ == "__main__":
    uvicorn.run("main:app", host="0,0,0,0", port=8000)

#  !uvicorn main:app --port 8000 --host 0.0.0.0






