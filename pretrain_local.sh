#!/bin/bash
if [ $# -eq 0 ]
then
    echo "No arguments supplied. Exiting."
    exit 1
else
    epochs=$1
    run_uuid=$2
    random_seed=$3
    dataset_name="${4:-gb1}"
fi

#echo "Copying ${dataset_name} dataset"
# cp "/staging/iaross/processed-${dataset_name}.tar.gz" .
echo "Untarring ${dataset_name} dataset"
mkdir -p ${dataset_name}
tar -xvzf processed-${dataset_name}.tar.gz -C "${dataset_name}" --strip-components=1

#unzip cleaned_data_test.zip -d precleaned
# rm "processed-${dataset_name}.tar.gz"

ln -s /workspace/metl/data/

parent_dir=$(realpath "${PWD}/${dataset_name}"/splits/*/)
splits_dir=$(basename "${parent_dir}")

pwd
env

mkdir wandb
mkdir wandb_data
export WANDB_DIR=$PWD/wandb
export WANDB_DATA_DIR=$PWD/wandb_data
export WANDB_CACHE_DIR=$PWD/wandb/.cache
export WANDB_CONFIG_DIR=$PWD/wandb/.config
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

python /workspace/metl/code/train_source_model.py @/workspace/metl/args/pretrain_$dataset_name\_local.txt \
    --ds_fn "$PWD/${dataset_name}/${dataset_name}.db"   \
    --split_dir "$PWD/${dataset_name}/splits/${splits_dir}" \
    --max_epochs $epochs --uuid=$run_uuid  \
    --random_seed $random_seed

