# WSI-Classification

The purpose of this repo is to extend the functionality of TF-Slim for use in other projects.
This is derived from  commit `70c86f2` of (tensorflow/models)[https://github.com/tensorflow/models].
#Set ENV variable
MODELS=/data/models
export PYTHONPATH=$PWD/:/data/models/research/slim/

0. Convert images to TFRecords
# Create `scripts/build_image_data.py` to convert a directory with subdirectories of images to TFrecords and labels
```
python scripts/create_tf_records.py \
    --input_directory /projects/deepLearning/mayo/data/processed/Training-Images-Set-01/ \
    --num_shards 5 \
    --validation_count 10000 \
    --prefix spitz \
    --output_directory /data/images/
```


1. Copy the `models/research/slim/datasets/dataset_factory.py` to `datasets/dataset_factory.py`
# Add in the SPITZ mapping and remove the rest

2. Create a `spitz.py` dataset.dataProvider to teach TF how to decode your data.
#This is where I need to add my decoder to tell TF how to extract my images and labels from TFRecords


3. Download and unzip the `inception_v3.ckpt` file into the `checkpoints` directory
```
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar xvzf inception_v3_2016_08_28.tar.gz
mv inception_v3.ckpt checkpoints/
mv inception_v3_2016_08_28.tar.gz checkpoints/
```

4. Create a preprocessing script for SPITZ


# verify it works on the flowers dataset
 python scripts/train_image_classifier.py
 	--train_dir=${TRAIN_DIR} \
 	--dataset_name=flowers    \
 	--dataset_split_name=train \
 	--dataset_dir $DATASET_DIR \
 	--model_name=inception_v3  \
 	--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
 	--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
 	--checkpoint_path checkpoints/inception_v3.ckpt \
 	--moving_average_decay 0.05 \
 	--log_every_n_steps 1000



5. Run the pretrained model on the SPITZ dataset.
```
DATASET_DIR=/data/images/
TRAIN_DIR=/tmp/from_checkpoint
python scripts/train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=spitz \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3 \
    --checkpoint_path checkpoints/inception_v3.ckpt \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --preprocessing_name spitz 
```

6. Run the naive model on Spitz
```
TRAIN_DIR=/tmp/from_scratch
for DA in '--DA' ''
do
 for model_name in inception_v4 inception_v3
 do
 for optimizer in adadelta adagrad adam ftrl momentum sgd rmsprop
 do
 for lr in 0.01 0.05 0.001
 do
time python scripts/train_image_classifier.py  \
    --train_dir=${TRAIN_DIR}/${model_name}_${DA//-/}_${optimizer}_${lr} \
    --dataset_name=spitz     \
    --train_image_size 299 \
    --dataset_split_name=train     \
    --dataset_dir=${DATASET_DIR}    \
     --model_name=${model_name} \
     --preprocessing_name spitz  \
     --log_every_n_steps 1000 \
     --num_clones 4 \
     --optimizer ${optimizer} \
     --max_number_of_steps 20000 \
     ${DA}
     echo ${model_name}_${DA//-}_${optimizer}_${lr}
    done
  done
 done
done
```