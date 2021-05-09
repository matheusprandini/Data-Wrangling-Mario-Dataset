# Data-Wrangling-Mario-Dataset

Very simple project for data wrangling of Mario Bros Datasets.

## Dependencies

The following dependencies must be installed to run this project:

- Python3: sudo apt install python3-pip
- Numpy: pip3 install numpy
- Pandas: pip3 install pandas
- Tensorflow: pip3 install tensorflow
- Keras: pip3 install keras
- Cv2: pip3 install opencv-python (needs to upgrade pip: pip3 install --upgrade pip)

## Build and Run

### Create Config File

The `config.json` configuration file (under conf/ directory) has the following structure:

```
{
    "datasetInfo": {
        "inputDataset": "/path/to/source/dataset/files/",
        "outputDataset": "/path/to/output/dataset/files/",
        "classes": ["Class1", "Class2", ..., "ClassN"],
        "chunkSize": chunk_size, (Recommended Value: 6)
        "imageSize": image_size (Recommended Value: 224)
    },
    "featureExtractor": {
        "name": "model_name" (MobileNetV2, ResNet50V2 or VGG16)
    }
}
```

- **datasetInfo/inputDataset:** directory to load video data.
- **datasetInfo/outputDataset:** directory to save image data.
- **datasetInfo/classes:** classes to extract frames.
- **datasetInfo/chunkSize:** number of frames that compose the chunk.
- **datasetInfo/imageSize:** size of frames to use in the execution.
- **featureExtractor/name:** name of the feature extractor model to use in the execution.

### Execution

Steps to execute this code:

- Gather data.
- Create `config.json`.
- Execute the following command under **src/** directory: ```python3 Main.py```
