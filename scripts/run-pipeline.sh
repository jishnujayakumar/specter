export ELECTER_DIR=`pwd`
chmod a+x pre-experiment/preprocess-legal-data.sh
export LEGAL_DATA_DIR=$1
export MAX_SEQ_LENGTH=$4

# Get legal-bert
cd $ELECTER_DIR/pre-experiment/
python get-legal-bert.py 

cd $ELECTER_DIR

# Preprocess Data
./pre-experiment/preprocess-legal-data.sh $LEGAL_DATA_DIR $2 $3

# Create triplets files
python $ELECTER_DIR/specter/data_utils/create_training_files.py \
--data-dir $LEGAL_DATA_DIR/preProcessedData \
--metadata $LEGAL_DATA_DIR/preProcessedData/metadata.json \
--outdir $LEGAL_DATA_DIR/preProcessedData/experimentData \
--bert_vocab $ELECTER_DIR/pre-experiment/legal-bert-base-uncased/vocab.txt \
--included-text-fields title \
--ratio_hard_negatives 0.4

NUM_TRAIN_INSTANCES=`grep 'train' $LEGAL_DATA_DIR/preProcessedData/experimentData/data-metrics.json | sed -r 's/^[^:]*:(.*)$/\1/' | sed 's/ //g' | sed 's/,//g'`

model_out_dir="$LEGAL_DATA_DIR/model-output-$2-$3-$MAX_SEQ_LENGTH"

# Perform training
rm -rf $model_out_dir && $ELECTER_DIR/scripts/run-exp-simple.sh -c $ELECTER_DIR/experiment_configs/simple.jsonnet \
-s $model_out_dir --num-epochs 10 --batch-size 32 \
--train-path $LEGAL_DATA_DIR/preProcessedData/experimentData/data-train.p \
--dev-path $LEGAL_DATA_DIR/preProcessedData/experimentData/data-val.p \
--num-train-instances $NUM_TRAIN_INSTANCES --cuda-device -1 --max-seq-len $MAX_SEQ_LENGTH 

# Move model artifacts to appropriate tar.gz file
cd $model_out_dir
mv best.th weights.th
mkdir -p model
mv weights.th config.json model/
mv vocabulary/ model/vocabulary/
tar czC model . --transform='s,^\./,,' >| model.tar.gz

# Run inference on gold-docs
cd $ELECTER_DIR
CUDA_VISIBLE_DEVICES=1 \
python $ELECTER_DIR/scripts/embed.py \
--ids $LEGAL_DATA_DIR/Gold-Score-Docs/gold-docs.txt \
--model $model_out_dir/model.tar.gz \
--metadata $LEGAL_DATA_DIR/Gold-Score-Docs/metadata.json \
--cuda-device -1 \
--batch-size 32 \
--output-file $LEGAL_DATA_DIR/embeddings-output-gold-docs.jsonl \
--vocab-dir $LEGAL_DATA_DIR/legal-data-vocab/ \
--included-text-fields title

python3 $ELECTER_DIR/scripts/result-analysis.py $LEGAL_DATA_DIR


