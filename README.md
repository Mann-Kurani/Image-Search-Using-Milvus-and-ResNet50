# Image Search Using Milvus and ResNet50

## Overview
This project demonstrates an image search system using Milvus as a vector database and ResNet50 for generating image embeddings. The system enables efficient similarity searches by indexing image features and retrieving visually similar images based on L2 distance.

## Features
- **Milvus VectorDB** for storing and searching image embeddings
- **ResNet50** for generating high-quality image embeddings
- **Batch processing** for efficient embedding and insertion
- **L2 distance metric** for similarity search
- **Visualization** of search results with Matplotlib

## Installation
Ensure you have Python installed, then install the required dependencies:

```bash
pip install pymilvus torch gdown torchvision tqdm
```

## Setting Up Milvus
1. Define global parameters:
   ```python
   COLLECTION_NAME = 'image_search'
   DIMENSION = 2048
   BATCH_SIZE = 128
   TOP_K = 3
   ```
2. Connect to Milvus:
   ```python
   from pymilvus import MilvusClient
   client = MilvusClient("milvus1.db")
   ```
3. Define the schema and create the collection:
   ```python
   schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
   schema.add_field("id", datatype=DataType.INT64, is_primary=True, auto_id=True)
   schema.add_field("filepath", datatype=DataType.VARCHAR, max_length=200)
   schema.add_field("image_embedding", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
   client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
   ```

## Indexing and Searching
4. Define index parameters:
   ```python
   index_params = client.prepare_index_params()
   index_params.add_index(field_name="image_embedding", index_type="IVF_FLAT", metric_type="L2", params={"nlist": 16384})
   client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
   client.load_collection(collection_name=COLLECTION_NAME)
   ```

## Generating Image Embeddings
5. Load ResNet50:
   ```python
   import torch
   model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
   model = torch.nn.Sequential(*(list(model.children())[:-1]))
   model.eval()
   ```
6. Define preprocessing:
   ```python
   from torchvision import transforms
   preprocess = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
   ])
   ```

## Inserting Data
7. Download and extract dataset:
   ```python
   import gdown, zipfile
   url = 'https://drive.google.com/uc?id=1OYDHLEy992qu5C4C8HV5uDIkOWRTAR1_'
   gdown.download(url, './paintings.zip')
   with zipfile.ZipFile("./paintings.zip","r") as zip_ref:
       zip_ref.extractall("./paintings")
   ```
8. Embed and insert images into Milvus:
   ```python
   from PIL import Image
   from tqdm import tqdm
   def embed(data):
       with torch.no_grad():
           output = model(torch.stack(data[0])).squeeze()
           embd_data = output.tolist()
           data = [{"filepath": data[1][i], "image_embedding": embd_data[i]} for i in range(len(embd_data))]
           client.insert(collection_name=COLLECTION_NAME, data=data)
   ```

## Searching for Similar Images
9. Generate embeddings for test images:
   ```python
   def test_embed(data):
       with torch.no_grad():
           ret = model(torch.stack(data))
           return ret.squeeze().tolist()
   ```
10. Search images using L2 metric:
    ```python
    res = client.search(
        collection_name=COLLECTION_NAME,
        anns_field='image_embedding',
        data=test_embed(data_batch[0]),
        limit=TOP_K,
        output_fields=["filepath"]
    )
    ```

## Visualizing Results
11. Plot search results:
    ```python
    from matplotlib import pyplot as plt
    f, axarr = plt.subplots(len(data_batch[1]), TOP_K + 1, figsize=(20, 10), squeeze=False)
    for hits_i, hits in enumerate(res):
        axarr[hits_i][0].imshow(Image.open(data_batch[1][hits_i]))
        axarr[hits_i][0].set_title('Test Image')
        for hit_i, hit in enumerate(hits):
            axarr[hits_i][hit_i + 1].imshow(Image.open(hit['entity'].get('filepath')))
            axarr[hits_i][hit_i + 1].set_title(f'Result Image (Distance: {hit["distance"]})')
    plt.savefig('search_result.png')
    ```

## Randomized Search
12. Perform a random search:
    ```python
    data = torch.rand(size=[5, DIMENSION]).tolist()
    res = client.search(collection_name=COLLECTION_NAME, anns_field='image_embedding', data=data, limit=TOP_K, output_fields=["filepath"])
    ```

## Results
- The system successfully indexes and retrieves images based on visual similarity.
- Search results are displayed with matched images and similarity scores.
- Performance is optimized through batch processing and efficient indexing.


---
This README provides a structured overview of your project, making it easy for users to understand and replicate your work.

