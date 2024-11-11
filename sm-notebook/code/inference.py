import json
import time
import numpy as np
import torch, os, json, base64, cv2, time
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    env = os.environ
    model = YOLO(os.path.join(model_dir, env['YOLOV8_MODEL']))
    model.to(device)
    return model

def input_fn(request_body, request_content_type):
    start = time.time()
    print("Executing input_fn from inference.py ...")
    print('request_content_type:', request_content_type)
    if request_content_type == 'text/csv':
        jpg_original = base64.b64decode(request_body)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=-1)
    elif request_content_type == 'application/json':
        requests = json.loads(request_body)
        images = []

        def process_single_image(jpg_original):
            """处理单张图片的辅助函数"""
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            img = cv2.imdecode(jpg_as_np, flags=cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 转换为float32并归一化到0-1
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            return torch.from_numpy(img).float()

        if isinstance(requests, list):
            # 处理批量图片
            for request in requests:
                jpg_original = base64.b64decode(request['image'])
                img_tensor = process_single_image(jpg_original)
                images.append(img_tensor)
        else:
            # 处理单张图片
            jpg_original = base64.b64decode(requests['image'])
            img_tensor = process_single_image(jpg_original)
            images.append(img_tensor)

        batch_tensor = torch.stack(images)  # .to(device)  # BCHW format with RGB channels float32 (0.0-1.0).
        
        end = time.time()
        print('intput_fn time:', end-start)
        return batch_tensor
    else:
        raise Exception("Unsupported content type: " + request_content_type)
    return img

def predict_fn(input_data, model):
    start = time.time()
    print("Executing predict_fn from inference.py ...")
    with torch.no_grad():
        result = model(input_data)  # , verbose=False
    end = time.time()
    print('predict_fn time:', end-start)
    return result

def output_fn(prediction_output, content_type):
    print("Executing output_fn from inference.py ...")
    results = []
    # print('prediction_output:', len(prediction_output))
    for output in prediction_output:
        infer = {}
        for result in output:
            if 'boxes' in result._keys and result.boxes is not None:
                infer['boxes'] = result.boxes.cpu().numpy().data.tolist()
            if 'masks' in result._keys and result.masks is not None:
                infer['masks'] = result.masks.cpu().numpy().data.tolist()
            if 'keypoints' in result._keys and result.keypoints is not None:
                infer['keypoints'] = result.keypoints.cpu().numpy().data.tolist()
            if 'probs' in result._keys and result.probs is not None:
                infer['probs'] = result.probs.cpu().numpy().data.tolist()
        results.append(infer)
    return json.dumps(results)

if __name__ == '__main__':
    os.environ['YOLOV8_MODEL'] = 'yolov8l.pt'
    model = model_fn('../')

    img_path = '../bus.jpg'

    model_height, model_width = 640, 640
    orig_image = cv2.imread(img_path)
    resized_image = cv2.resize(orig_image, (model_height, model_width))
    resized_jpeg = cv2.imencode('.jpg', resized_image)[1]
    payload = base64.b64encode(resized_jpeg).decode('utf-8')

    start = time.time()
    # input_data = input_fn(payload, 'text/csv')
    # input_data = input_fn(json.dumps({'image': payload}), 'application/json')
    batch_size = 50
    input_data = input_fn(json.dumps([{'image': payload} for i in range(batch_size)]), 'application/json')
    end1 = time.time()
    print('intput_fn time:', end1-start)

    result = predict_fn(input_data, model)
    # print(result)
    end2 = time.time()
    print('predict_fn time:', end2-end1)

    output = output_fn(result, 'application/json')
    # print(output)
    end = time.time()
    print('output_fn time:', end-end2)
    print('time:', end-start)
    
    for i in range(10):
        start = time.time()
        result = predict_fn(input_data, model)
        end = time.time()
        print('predict_fn time:', end-start)