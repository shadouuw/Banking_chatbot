FROM python:3.9
WORKDIR /flaskProject
COPY requirements.txt .
RUN pip install cmake==3.22.4
RUN pip install dlib==19.20
RUN apt-get update
RUN apt-get install libasound-dev libportaudio2 libportaudiocpp0 portaudio19-dev -y
RUN pip install pyaudio
RUN pip3 install -r requirements.txt
RUN pip3 install speechrecognition
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install build-essential
RUN apt-get -y install lsb-release
RUN apt-get -y install vim
COPY ./app ./app
CMD ["python","./app/app.py"]