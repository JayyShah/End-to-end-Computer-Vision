# SSD (Single Shot Detector) Algorithm for Custom Dataset

## Overview

This repository provides an implementation of the Single Shot Detector (SSD) algorithm for object detection on a custom dataset. To use this code, follow the instructions below to set up your dataset and configure the necessary parameters.

## Dataset Preparation

1. **VOC2007 Directory:**
    - Add your dataset images to `VOC2007/JPEGImages/`.
    - Include annotations in PascalVOC format in `VOC2007/Annotations/`.

2. **Run Dataset Splitting Script:**
    - Execute `split_train_test.py` in the root directory to create 'train.txt' and 'test.txt' in `VOC2007/ImageSets/`.

3. **XML to TFRecords Conversion:**
    - Modify `xmltotfrecords_conversion.py`:
        - Update `VOC_LABELS` based on your number of classes.
        - Set correct paths for `dataset_dir` and `output_directory`.
    - Run the script to convert XML annotations to TFRecords format.

## Configuration Files

1. **datasets/pascalvoc_2007.py:**
    - Adjust train and test statistics based on your dataset.
    - Specify the number of classes.
    - Update class-specific information (e.g., 'none', 'class_1', 'class_2').

2. **datasets/pascalvoc_common.py:**
    - Update `VOC_LABELS` based on your dataset classes.

## Training Configuration

1. **Train SSD Network:**
    - Edit `train_ssd_network.py`:
        - Set `TRAIN_DIR` and `CHECKPOINT_PATH`.
        - Specify the number of classes in `num_classes`.
        - Adjust `batch_size` using `tf.app.flags.DEFINE_integer`.
    - Execute the training command:

    ```bash
    python train_ssd_network.py --dataset_name=pascalvoc_2007 --dataset_split_name=train --model_name=ssd_300_vgg --save_summaries_secs=60 --save_interval_secs=600 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.001 --batch_size=6 --gpu_memory_fraction=0.9 --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box
    ```

Ensure that you have made all the above changes correctly before initiating the training process.

Feel free to customize other parameters and configurations based on your specific requirements.