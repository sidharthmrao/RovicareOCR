FROM ubuntu:latest

WORKDIR /ocr  
VOLUME /ocr

RUN apt-get -y update   
RUN apt-get -y install poppler-utils
RUN apt-get -y install tesseract-ocr
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install default-jre

COPY requirements.txt /src/requirements.txt

RUN pip install cachetools==4.2.2
RUN pip install certifi==2021.5.30
RUN pip install charset-normalizer==2.0.4
RUN pip install ci-info==0.2.0
RUN pip install click==8.0.1
RUN pip install distro==1.6.0
RUN pip install etelemetry==0.2.2
RUN pip install filelock==3.0.12
RUN pip install google-api-core==2.0.1
RUN pip install google-auth==2.0.2
RUN pip install google-cloud==0.34.0
RUN pip install google-cloud-vision==2.4.2
RUN pip install googleapis-common-protos==1.53.0
RUN pip install grpcio==1.39.0
RUN pip install idna==3.2
RUN pip install isodate==0.6.0
RUN pip install lxml==4.6.3
RUN pip install networkx==2.6.2
RUN pip install numpy==1.21.2
RUN pip install opencv-python
RUN pip install packaging==21.0
RUN pip install pandas==1.3.2
RUN pip install pdf2image==1.16.0
RUN pip install Pillow==8.3.1
RUN pip install proto-plus==1.19.0
RUN pip install protobuf==3.17.3
RUN pip install prov==2.0.0
RUN pip install pyasn1==0.4.8
RUN pip install pyasn1-modules==0.2.8
RUN pip install pydot==1.4.2
RUN pip install pyparsing==2.4.7
RUN pip install pytesseract==0.3.8
RUN pip install python-dateutil==2.8.2
RUN pip install pytz==2021.1
RUN pip install rdflib==6.0.0
RUN pip install requests==2.26.0
RUN pip install rsa==4.7.2
RUN pip install scipy==1.7.1
RUN pip install simplejson==3.17.5
RUN pip install six==1.16.0
RUN pip install tabula-py==2.3.0
RUN pip install traits==6.2.0
RUN pip install urllib3==1.26.6
RUN pip install flask
RUN pip install google-cloud-storage
RUN pip install tqdm



COPY conversionprgm.py /ocr/conversionprgm.py
COPY testapi.py /ocr/testapi.py
COPY key.json /ocr/key.json
COPY storagekey.json /ocr/storagekey.json
COPY data/filler.txt /ocr/data/filler.txt
COPY config.ini /ocr/config.ini

EXPOSE 5000
CMD ["python3", "/ocr/testapi.py"]

##Created by Sidharth Rao -- github.com/sidharthmrao

  