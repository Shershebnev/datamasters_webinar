import glob
import requests

from tqdm import tqdm


URL = "http://localhost:5000/predict"

# read files and prepare batch
files = glob.glob("data/test/*/*JPG")
content_type = 'image/jpeg'
headers = {'content-type': content_type}
for i in tqdm(range(len(files)), total=len(files)):
    img = open(files[i], "rb").read()
    response = requests.post(URL, data=img, headers=headers)
