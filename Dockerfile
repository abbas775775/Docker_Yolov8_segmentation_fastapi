
FROM python:3.8.5-slim

WORKDIR /app

COPY ./app/requirements.txt /app/requirements.txt

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \ 
    libglib2.0-0 -y


RUN pip install --upgrade pip

RUN apt install -y libgl1-mesa-glx
RUN pip install python-multipart
RUN pip install -r requirements.txt


COPY ./app /app


EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



