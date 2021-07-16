#!/bin/bash

# saner programming env: these switches turn some bugs into errors
set -o errexit -o pipefail -o noclobber -o nounset

! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=c:s:
LONGOPTS=debug,force,output:,verbose,train-path:,test-path:,dev-path:,config:,serialization-dir:,num-epochs:,bert-vocab:,bert-model:,bert-weights:,vocab:,cuda-device:,recover,dataset-type:,lazy,include-venue,batch-size:,num-train-instances:,max-seq-len:

# -use ! and PIPESTATUS to get exit code with errexit set
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

#d=n f=n v=n outFile=- num-epochs=30
# set default args
NUM_EPOCHS=2
lazy="true"
batch_size=2
# bert_requires_grad='all'  #'pooler,11,10,9,8,7,6,5,4,3,2,1,0'

# finetune last 2 layers
bert_requires_grad='pooler,11,10' #'all' or 'pooler,11,10,9,8,7,6,5,4,3,2,1,0'
default_lr=2e-5

num_train_instances=3000
max_seq_len=256

# ---------------

TRAIN_PATH="data/training-data/train.pkl"
DEV_PATH="data/training-data/val.pkl"

BERT_MODEL="bert-pretrained"
BERT_VARIANT=$BERT_VARIANT
# BERT_VARIANT="legal-bert-base-uncased" # for nlpaueb-legal-bert
# BERT_VARIANT="custom-legalbert" # for custom-legal-bert
BERT_VOCAB="$ELECTER_DIR/pre-experiment/$BERT_VARIANT/vocab.txt"
BERT_WEIGHTS="$ELECTER_DIR/pre-experiment/$BERT_VARIANT/$BERT_VARIANT.tar.gz"

VOCAB_DIR="$LEGAL_DATA_DIR/legal-data-vocab/"
INCLUDE_VENUE="false"

# --------------

while true; do
    case "$1" in
        --train-path)
            TRAIN_PATH="$2"
            shift 2
            ;;
        --dev-path)
            DEV_PATH="$2"
            shift 2
            ;;
        -c|--config)
            config_file="$2"
            shift 2
            ;;
        -s|--serialization-dir)
            serialization_dir="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --vocab)
            VOCAB_DIR="$2"
            shift 2
            ;;
        --bert-vocab)
            BERT_VOCAB="$2"
            shift 2
            ;;
        --bert-weights)
            BERT_WEIGHTS="$2"
            shift 2
            ;;
        --bert-model)
            BERT_MODEL="$2"
            shift 2
            ;;
        --cuda-device)
            gpu="$2"
            shift 2
            ;;
        --recover)
            recover=true
            shift
            ;;
        --include-venue)
            INCLUDE_VENUE="true"
            shift
            ;;
        --batch-size)
            batch_size="$2"
            shift 2
            ;;
        --num-train-instances)
            num_train_instances="$2"
            shift 2
            ;;
        --max-seq-len)
            max_seq_len="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

#echo "running experiment: $config_file, train_path: $TRAIN_PATH, coviews: $COVIEWS, cocites: $COCITES, copdfs: $COPDFS, epochs: $NUM_EPOCHS, vocab: $vocab, limit-training: $RatioTrainingSamples"

export TRAIN_PATH=$TRAIN_PATH
export DEV_PATH=$DEV_PATH
export VOCAB_DIR=$VOCAB_DIR
export NUM_EPOCHS=$NUM_EPOCHS
export BATCH_SIZE=$batch_size
export CUDA_DEVICE=$gpu
export LAZY=$lazy
export TRAINING_DATA_INSTANCES=$num_train_instances
export BERT_REQUIRES_GRAD=$bert_requires_grad
export BERT_MODEL=$BERT_MODEL
export MAX_SEQ_LEN=$MAX_SEQ_LENGTH
export INCLUDE_VENUE=$INCLUDE_VENUE
if [ -z "${BERT_VOCAB+x}" ]
then
    echo "Bert Weights Not Set"
else
    export BERT_VOCAB=$BERT_VOCAB
    export BERT_WEIGHTS=$BERT_WEIGHTS
fi
if [ -z "${recover+x}" ]
then
    python3 -m allennlp.run train $config_file  --include-package $EMB_MODEL -s $serialization_dir
else
    python3 -m allennlp.run train $config_file  --include-package $EMB_MODEL -s $serialization_dir --recover
fi
