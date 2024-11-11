REGION = 'us-east-1'
ENDPOINT_NAME='<COPY PASTE SAGEMAKER ENDPOINT NAME FROM CONSOLE HERE>'

HOST = f'runtime.sagemaker.{REGION}.amazonaws.com'
# replace the url below with the sagemaker endpoint you are load testing
SAGEMAKER_ENDPOINT_URL = f'https://{HOST}/endpoints/{ENDPOINT_NAME}/invocations'
ACCESS_KEY = '<COPY PASTE YOUR AWS ACCESS KEY HERE>'
SECRET_KEY = '<COPY PASTE YOUR AWS SECRET KEY HERE>'
# replace the context type below as per your requirements
CONTENT_TYPE = 'text/csv'
# CONTENT_TYPE = 'application/json'
# TARGET_MODEL = 'model1.tar.gz'
METHOD = 'POST'
SERVICE = 'sagemaker'
SIGNED_HEADERS = 'content-type;host;x-amz-date'
CANONICAL_QUERY_STRING = ''
ALGORITHM = 'AWS4-HMAC-SHA256'
