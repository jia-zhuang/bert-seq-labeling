import requests

url = 'http://localhost:8501/v1/models/tf_debug:predict'
data = {'inputs': {
            'input_ids': [[1] * 256],
            'input_mask': [[1] * 256],
            'segment_ids': [[1] * 256]
            }
       }
       
resp = requests.post(url, json=data)
print(resp)
       
