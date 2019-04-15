# keras-yolo3

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).


## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).

```
wget https://pjreddie.com/media/files/yolov3.weights
```

2. Convert the Darknet YOLO model to a Keras model.

```
python utils/convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```

3. Run YOLO detection.

```
python yolo_predict.py [OPTIONS...] --input_type, for image detection mode, OR
python yolo_predict.py --input_type video [video_path] [output_path (optional)]
```

For example if you want to train for an image run:

```
python yolo_predict.py --input_type image
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.

    - One row for one image;  
    - Row format: `image_file_path box1 box2 ... boxN`;  
    - Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  

    For VOC dataset, try `python preprocessing/voc_annotation.py`

    Here there is an example of input file:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. create a file that maps the `class_id`s to the class name and save it into the folder `model_data`

3. Make sure you have converted the weights from yolo. The file model_data/yolo_weights.h5 is used to load pretrained weights.

4. run the `train.py` script to train the model

```
Usage: train.py [OPTIONS]

Options:
  --annotation_path TEXT          the path of the file with the image paths
                                  and annotations  [required]
  --log_dir TEXT                  the directory for the results and log
  --classes_path TEXT             the path with the classes for the model
                                  [required]
  --anchors_path TEXT             the path with the paths for the model
                                  [required]
  --weights_path TEXT             the path of the model to use as starting
                                  point,
                                  specify None to start from scratch
                                  [required]
  --mode [only_dense|fine_tuning]
```

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.
