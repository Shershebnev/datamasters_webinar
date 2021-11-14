# datamasters_webinar
Materials for the webinar on ML model life cycle

Master branch contains the base code which is "productionized" during the webinar (and in separate branches in git)

# Base code (master branch)
Contains of 4 main scripts:
* prepare_data.py - used for preparing dataset for further training. Example usage:  
`python3 prepare_data.py --data_path /data --verbose 1`  

Original structure of the dataset is expected to follow the following structure:
```
    args.data_path/
        class_1/
            image1.jpg
            image2.jpg
        ...
        class_2/
            image1.jpg
            image2.jpg
        ...
```
Final structure of the dataset:
```
    args.data_path/
        train/
            class_1/
                image1.jpg
                image2.jpg
                ...
            ...
        val/
            class_1/
                image1.jpg
                image2.jpg
                ...
            ...
        test/
            class_1/
                image1.jpg
                image2.jpg
                ...
            ...
```
* train.py - used for training the classification network. Example usage:  
`python3 train.py --batch_size 512 --image_shape 224 --model_type resnet18 --epochs 2 --data_path /data/ --verbose 1`
* eval.py - used to evaluating trained network on test set. Example usage:  
`python3 eval.py --batch_size 512 --image_shape 224 --model_type resnet18 --weights_path resnet18.h5 --data_path /data/ --verbose 1`
* predict.py - sample script to run predictions on other dataset or image. Example usage:  
`python3 predict.py --batch_size 512 --image_shape 224 --model_type resnet18 --weights_path resnet18.h5 --data_path /data/test/люди --output_file pred.csv --verbose 1`
