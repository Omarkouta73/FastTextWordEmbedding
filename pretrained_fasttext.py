import requests
import gzip
import shutil
import os

model_path = "cc.en.300.bin"
compressed_model_path = model_path + ".gz"

if not os.path.exists(model_path):
    if not os.path.exists(compressed_model_path):
        print("Downloading FastText pretrained model...")
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
        response = requests.get(url, stream=True)
        with open(compressed_model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
    # Decompress the file
    print("Decompressing model file...")
    with gzip.open(compressed_model_path, 'rb') as f_in:
        with open(model_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print("Model ready!")