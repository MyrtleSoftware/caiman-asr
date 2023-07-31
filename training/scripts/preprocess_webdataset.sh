: ${DATA_DIR:=}
: ${TRAIN_TAR_FILES:=}
: ${DATASET_NAME_LOWER_CASE:="librispeech"}
: ${MAX_DURATION_SECS:=16.7}
: ${SPM_SIZE:=8703}
: ${CONFIG_NAME:=base-8703sp}


SPM_NAME=$DATASET_NAME_LOWER_CASE$SPM_SIZE
CONFIG=configs/${CONFIG_NAME}.yaml
RUN_CONFIG=configs/${CONFIG_NAME}_run.yaml


# populate run config
cat $CONFIG | sed s/SENTENCEPIECE/$SPM_NAME/g | sed s/MAX_DURATION/$MAX_DURATION_SECS/g > $RUN_CONFIG

# build spm and move to /datasets/sentencepieces
python rnnt_train/common/data/webdataset_spm.py \
    --dataset_dir $DATA_DIR \
    --train_tar_files $TRAIN_TAR_FILES \
    --spm_size $SPM_SIZE \
    --spm_name $SPM_NAME \
    --model_config $RUN_CONFIG

mkdir -p /datasets/sentencepieces
mv $SPM_NAME.* /datasets/sentencepieces/
