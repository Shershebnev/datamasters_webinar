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


# 1. Experiment tracking and data versioning
I will be exploring two options here:
* Completely free and open-source option - [mlflow](https://mlflow.org/) for experiment tracking + [dvc](https://dvc.org/) for versioning  
* Proprietary solution with free tier option - [wandb](https://wandb.ai/site)

## 1.1 MLFlow and DVC
Available in the branch [mlflow_dvc](https://github.com/Shershebnev/datamasters_webinar/tree/mlflow_dvc). Installing both of them is easy:  
`pip install mlflow dvc[s3]`  
I will be using dvc with s3 support (hence [s3]) for data storage. As S3-like storage I will be using [minio](https://min.io/).
For model registry - [postgresql](https://www.postgresql.org/)
### Preliminary setup
Before using MLFlow and DVC let's first setup storage, database and MLFlow server.
#### Minio
As mentioned for S3-like storage I will be using Minio
which can be run on a host machine or in Docker. [Installation instructions](https://min.io/download) are available on the website.
Example on running minio locally:  
```shell
MINIO_ROOT_USER=admin MINIO_ROOT_PASSWORD=password minio server $HOME/minio_tmp --console-address ":9001"
mc alias set myminio http://localhost:9000 admin password
mc mb myminio/dvc-demo  # create bucket for dvc
mc mb myminio/mlflow-bucket  # create bucket for mlflow
```
#### Postgresql
Postgresql ([installation instructions](https://www.postgresql.org/download/)) can also be run locally or in docker like this:
```shell
initdb -D $HOME/psql_tmp
pg_ctl -D psql_tmp start
```
#### MLFlow server
To start MLFlow server and connect it to minio and postgresql here's a sample command
`MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000 AWS_ACCESS_KEY_ID=admin AWS_SECRET_ACCESS_KEY=password mlflow server --backend-store-uri postgresql://postgres:password@localhost:5432 --default-artifact-root s3://mlflow-bucket/ --host 0.0.0.0`  
After that you can access MLFlow UI in the web browser at http://localhost:5000

#### Docker-compose
Alternatively, there is a sample docker-compose file available in `mlflow_dvc` branch that will prepare everything for you using
`docker-compose up -d`

### DVC
Using DVC is interweaved with using git. Here's some sample usage:
* First initialize dvc in your repository:
    ```shell
    dvc init  # this will automatically run git add for all new files
    git commit -m "Initialize DVC"  # commit them into git
    ```
* Next add some data you want to version:
    ```shell
    dvc add -R data/*  #  this will create .dvc file for each existing file inside data/*
    ```
* Now we need to add those newly created .dvc files into git. If there a lot of files git can file because of `argument list too long`, so instead
we can use:
  ```shell
  # for macos
  find data -type f -name "*.dvc" | tr \\n \\0  | xargs -0 git add
  # for linux
  find data -type f -name "*.dvc" | xargs -d "\n" git add
  ```
* And commit this:  
`git commit -m "First version of dataset"`
* To also push this data into remote storage we first need to set it up:
    ```shell
    dvc remote add -d minio s3://dvc-demo/
    dvc remote modify minio endpointurl http://127.0.0.1:9000
    dvc remote modify minio access_key_id admin  # as set up in example above
    dvc remote modify minio secret_access_key password  # as set up in example above
    git add .dvc/config
    git commit -m "Configure remote storage"
    ```
* Now let's push the data to remote storage:  
`dvc push` and that's it :)
  

### MLFlow client
Using MLFlow inside your code is pretty easy for most of the popular frameworks, especially using autologging capabilities.
Some examples can be found in [train.py in mlflow_dvc branch](https://github.com/Shershebnev/datamasters_webinar/blob/mlflow_dvc/train.py)
or on [MLFlow website](https://mlflow.org/docs/latest/tracking.html#automatic-logging).
After setting up mlflow tracking in the code we can run the training script as
`MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000 AWS_ACCESS_KEY_ID=admin AWS_SECRET_ACCESS_KEY=password python3 train.py --batch_size 128 --image_shape 224 --model_type resnet18 --epochs 2 --data_path data/ --verbose 1`

Apart from metrics MLFlow also stores artifacts like model files which can then be put into MLFlow Registry.
