# -----------------------------------------
# Sync From Local to S3
CHECKPOINST_LISTING="src/configs/list_checkpoints.py"
CHECKPOINT_ROOT_DIR="checkpoints"

# Create folder checkpoints if not exists
mkdir -p $CHECKPOINT_ROOT_DIR

python $CHECKPOINST_LISTING 1>&1 |
  while IFS=' ' read -r -a array
  do
    profile=${array[0]}
    bucketName=${array[1]}
    modelName=${array[2]}
    prefix=${array[3]}
    isDir=${array[4]}

    echo "syncing $modelName"

    if [ "${array[$IS_DIR]}" = "True" ]; then
      aws --profile  $profile s3 sync $CHECKPOINT_ROOT_DIR/$modelName/$prefix s3://$bucketName/$modelName/$prefix --exact-timestamps
    else
      aws --profile  $profile s3 sync $CHECKPOINT_ROOT_DIR/$modelName s3://$bucketName/$modelName --exclude "*" --include "$prefix*" --exact-timestamps
    fi
  done