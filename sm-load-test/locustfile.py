from locust import HttpUser, task
from authorizer import authorize, get_payload_hash
import config as conf
import json
import cv2
import base64
import config as conf

SIZE=640
test_file = './bus.jpg'
image = cv2.imread(test_file)
image = cv2.resize(image, (SIZE, SIZE))
image = cv2.imencode('.jpg', image)[1]
# Serialize the jpg using base 64
payload = base64.b64encode(image).decode('utf-8')

batch_size = 50
batch_payload = json.dumps([{'image': payload} for _ in range(batch_size)]).encode('utf-8')
payload_hash = get_payload_hash(batch_payload, type='bytes')


class WebsiteUser(HttpUser):
    min_wait = 1
    max_wait = 5  # time in ms
    
    @task
    def test_post(self):
        global batch_payload, payload_hash
        """
        Load Test SageMaker Endpoint (POST request)
        """
        headers = authorize(batch_payload, payload_hash)
        resp = self.client.post(conf.SAGEMAKER_ENDPOINT_URL, data=batch_payload, headers=headers, name='Post Request')
        if resp.status_code != 200:
            print(resp)