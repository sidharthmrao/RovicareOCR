import flask
from flask import request, jsonify
import logging
from google.cloud import storage
from google.oauth2 import service_account
import sys
import subprocess
import json

import configparser

config = configparser.ConfigParser()
config.read("config.ini")
keys = config["KEYS"]
store = config["STORAGE"]

credentials = service_account.Credentials.from_service_account_file(keys["storagesa"])
storage_client = storage.Client('vision-6964', credentials=credentials)
bucket = storage_client.get_bucket(store["bucketname"])

app = flask.Flask(__name__)
app.config["DEBUG"] = False

@app.route('/', methods=['GET'])
def home():
    return "<h1>MediText/ConvertMed API</h1><p>This API converts medical forms into a json format for easy form->database transfers.</p><p>Created by Sidharth Rao</p>"

@app.route('/convert/', methods=['GET'])
def api_all():
    if 'path' in request.args:
        file_path = request.args.get('path')
        blob = bucket.blob(file_path)
        blob.download_to_filename(f'data/{file_path}')
        convert = subprocess.Popen(f"python3 conversionprgm.py {file_path}", shell=True, stdout=subprocess.PIPE)
        convert.wait()
        out, err = convert.communicate()
        return str(out).replace("b'", '').replace('NaN', '""').replace('\\n', '')
    else:
        return "<h1>ConvertMed API</h1><p>Please reload page with {/?path='file-path'} at the end.</p>"
        

app.run(host='0.0.0.0')

##Created by Sidharth Rao -- github.com/sidharthmrao