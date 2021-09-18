from google.cloud import storage
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file("vision-6964-cec0e32a1768.json")

CLIENT_ID = "GOOG1ELM7PJRRII5V3WQZJDFPLMVLU7BWMX3CPYOIF4QWXQGHG37DSZDCVYSY"
CLIENT_SECRET = "Nk/1HI0zI00gx208y4Sm+ZiK/dP8sqpt7i+QoIWZ"

storage_client = storage.Client('vision-6964', credentials=credentials)

bucket = storage_client.get_bucket('convertmed-form-bucket')

file_name = 'admissionRecord2.xhtml.pdf'

blob = bucket.blob(file_name)

blob.download_to_filename(f'data/{file_name}')