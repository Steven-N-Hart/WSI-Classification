# WSI-Classification

The purpose of this repo is to extend the functionality of TF-Slim for use in other projects.
This is derived from  commit `70c86f2` of (tensorflow/models)[https://github.com/tensorflow/models].


## Installation
to use

```
# Build the docker image
docker build -t stevenhart/wsi-classification .

# initialize the docker container and map volumes
# here I map my current working directory to `/data` so I can get my output
# I also map `/path/to/img/` so I know where to get my whole slide images
docker run -it --rm -v $PWD:/data -v /path/to/img:/img stevenhart/wsi-classification

```
Once inside the container, I can call scripts like:
```
python WSI-Classification/scripts/patch_extraction.py -i img/163538.svs -p 56 -o data/ -b 0.8 -v DEBUG

#which will output the results into the current directory on the host machine.
```


## The full process

## 0. Initialize variables
```
# Get tensorflow models
git clone https://github.com/tensorflow/models
cd models
git checkout --detach 70c86f2
cd ..

# set ENV
export MODELS=$PWD/models
export PYTHONPATH=$PWD/:$MODELS/research/slim/
```

## 1. Parse WSI into image patches or split up large hand-crafted jpegs to be 299x299
```
mkdir -p patches/spitz patches/conventional patches/other

# For each WSI, extract image patches. Since 0-50 are Spitz, and I have 150 slides, 
# I apply the following regex to get 119 of them
# find /data/temp/DL-[0-2][0-4][0-7]_HE.svs| xargs -I {} python scripts/patch_extraction.py -o patches/spitz -i {}
# Do the same for Conventional
# Note: this only applies when not using pre-selected regions of interest

# or parse hand crafted image chunks of different sizes
# find /data/temp/Deep_Learning_Set_01-Training_images/Used_Images/Other/ -name "*jpg"|
# xargs -I {} python scripts/parse_jpg.py -o patches/other -i {}
# Note: this only applies when using pre-selected regions of interest in unequally sized jpg files



# or just copy the 299x299 images to the appropriate folder
# patches/other/ # 38,734
# patches/spitz/ # 21,468
# patches/conventional/ # 15,868
# Note: Use this if you already have 299 x 299 image patches

```

## 2. Convert images to TFRecords
Create `scripts/build_image_data.py` to convert a directory with subdirectories of images to TFrecords and labels.
```
mkdir records
python scripts/create_tf_records.py \
    --input_directory patches/ \
    --num_shards 4 \
    --prefix spitz \
    --output_directory records
```

## 3. Download and unzip the `inception_v3.ckpt` file into the `checkpoints` directory
```
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar xvzf inception_v3_2016_08_28.tar.gz
mv inception_v3.ckpt checkpoints/
mv inception_v3_2016_08_28.tar.gz checkpoints/
```

## 4. Create a preprocessing script for SPITZ
This preprocessing step will ensure that during training, the images are randomly flipped to increase training diversity.
```
# see preprocessing/spitz_preprocessing.py & preprocessing/preprocessing_factory.py
```

## 5. Run the pretrained model on the SPITZ dataset.
```
# where the TFRecords directory is
DATASET_DIR=records/

for step in $(seq 50000 50000 250001)
do
    # Run pretrained model
    TRAIN_DIR=results/pretrained/inception_v3_rmsprop_0.01
    python scripts/train_image_classifier.py \
        --train_dir=${TRAIN_DIR} \
        --dataset_name=spitz \
        --dataset_split_name=train \
        --dataset_dir=${DATASET_DIR} \
        --model_name=inception_v3 \
        --log_every_n_steps 10000 \
        --num_clones 4 \
        --max_number_of_steps ${step} \
        --batch_size 32 \
        --checkpoint_path checkpoints/inception_v3.ckpt \
        --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
        --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
        --preprocessing_name spitz \
        --optimizer rmsprop \
        --learning_rate 0.01 

    python scripts/eval_image_classifier.py \
        --checkpoint_path ${TRAIN_DIR} \
        --eval_dir ${TRAIN_DIR}/${step} \
        --dataset_name=spitz \
        --dataset_split_name=validation \
        --dataset_dir=${DATASET_DIR} \
        --model_name=inception_v3 \
        --batch_size 32 \
        --model_name inception_v3 \
        --preprocessing_name spitz

    # Run naive model
    TRAIN_DIR=results/from_scratch/inception_v3_rmsprop_0.01
    python scripts/train_image_classifier.py \
        --train_dir=${TRAIN_DIR} \
        --dataset_name=spitz \
        --dataset_split_name=train \
        --dataset_dir=${DATASET_DIR} \
        --model_name=inception_v3 \
        --log_every_n_steps 10000 \
        --num_clones 4 \
        --max_number_of_steps ${step} \
        --batch_size 32 \
        --preprocessing_name spitz \
        --optimizer rmsprop \
        --learning_rate 0.01 

    python scripts/eval_image_classifier.py \
        --checkpoint_path ${TRAIN_DIR} \
        --eval_dir ${TRAIN_DIR}/${step} \
        --dataset_name=spitz \
        --dataset_split_name=validation \
        --dataset_dir=${DATASET_DIR} \
        --model_name=inception_v3 \
        --batch_size 32 \
        --model_name inception_v3 \
        --preprocessing_name spitz
done
```

## 6. Evaluate the model using WSIs that were not used for training.
```
label_file=records/labels.txt
chekckpoint_path=results/from_scratch/inception_v3_rmsprop_0.01

find /data/temp/DL-[1-2][0-4][0-9]_HE.svs| \
    xargs -I {} python scripts/classify_WSI.py \
        --wsi {} \
        --label_file ${label_file} \
        --preprocessing_name spitz \
        --checkpoint_path ${chekckpoint_path} \
        --dict_file spitz.dict

find /data/temp/DL-[1-2][5-9][0-9]_HE.svs| \
    xargs -I {} python scripts/classify_WSI.py \
        --wsi {} \
        --label_file ${label_file} \
        --preprocessing_name spitz \
        --checkpoint_path ${chekckpoint_path} \
        --dict_file conventional.dict
```

## 7. While that's running, let's try to export all image patches from the first 100 slides, but limit to 500k patches
```
OUT_DIR=/data/naiveTesting/patches
for x in /data/temp/DL-0[0-4][0-9]_HE.svs
do
    COUNT=$(ls ${OUT_DIR}/spitz|wc -l)
    if [[ "$COUNT" -lt 500000 ]];then 
        python scripts/patch_extraction.py -i ${x} -o ${OUT_DIR}/spitz
        echo -ne 'Completed $COUNT patches up to sample $x'\\r
    fi
done

for x in /data/temp/DL-0[5-9][0-9]_HE.svs
do
    COUNT=$(ls ${OUT_DIR}/spitz|wc -l)
    if [[ "$COUNT" -lt 500000 ]];then 
        python scripts/patch_extraction.py -i ${x} -o ${OUT_DIR}/conventional
        echo -ne 'Completed $COUNT patches up to sample $x'\\r
    fi
done

```

## 8. Convert those image patches to TFRecords
```
python scripts/create_tf_records.py \
    --input_directory ${OUT_DIR} \
    --num_shards 4 \
    --prefix spitz \
    --output_directory /data/naiveTesting/TFRecords/
```

## 9. Now, train against the non-curated image patches.
```
# where the TFRecords directory is
DATASET_DIR=/data/naiveTesting/TFRecords/

for step in $(seq 50000 50000 250001)
do
    # Run naive model
    TRAIN_DIR=results/non_curated/inception_v3_rmsprop_0.01
    python scripts/train_image_classifier.py \
        --train_dir=${TRAIN_DIR} \
        --dataset_name=spitz \
        --dataset_split_name=train \
        --dataset_dir=${DATASET_DIR} \
        --model_name=inception_v3 \
        --log_every_n_steps 10000 \
        --num_clones 4 \
        --max_number_of_steps ${step} \
        --batch_size 30 \
        --preprocessing_name spitz \
        --optimizer rmsprop \
        --learning_rate 0.01 

    if [ $? -eq 0 ]; then
        python scripts/eval_image_classifier.py \
            --checkpoint_path ${TRAIN_DIR} \
            --eval_dir ${TRAIN_DIR}/${step} \
            --dataset_name=spitz \
            --dataset_split_name=validation \
            --dataset_dir=${DATASET_DIR} \
            --model_name=inception_v3 \
            --batch_size 30 \
            --model_name inception_v3 \
            --preprocessing_name spitz
    fi
done
```