export ELECTER_DIR=`pwd`
chmod a+x pre-experiment/preprocess-legal-data.sh
export LEGAL_DATA_DIR=$1
export MAX_SEQ_LENGTH=$4
export BERT_VARIANT=$5

# Get legal-bert
cd $ELECTER_DIR/pre-experiment/
python get-legal-bert.py 

cd $ELECTER_DIR

# Preprocess Data
./pre-experiment/preprocess-legal-data.sh $LEGAL_DATA_DIR 0.9 1

# export EXPERIMENT_DATA_DIR="$LEGAL_DATA_DIR/preProcessedData/experimentData"
# export EXPERIMENT_DATA_DIR="$ELECTER_HULK_DIR/legal-data-dsdr-summarized/preProcessedData/experimentData-correct-custom-legal-bert"
export EXPERIMENT_DATA_DIR="$LEGAL_DATA_DIR/preProcessedData/experimentData-NLPAUEB-LEGAL-BERT"

# Create triplets files
python $ELECTER_DIR/specter/data_utils/create_training_files.py \
--data-dir $LEGAL_DATA_DIR/preProcessedData \
--metadata $LEGAL_DATA_DIR/preProcessedData/metadata.json \
--outdir $EXPERIMENT_DATA_DIR \
--included-text-fields title \
--ratio_hard_negatives 0.4 \
--bert_vocab $ELECTER_DIR/pre-experiment/custom-legalbert/vocab.txt
# --bert_vocab $ELECTER_DIR/pre-experiment/legal-bert-base-uncased/vocab.txt

NUM_TRAIN_INSTANCES=`grep 'train' $EXPERIMENT_DATA_DIR/data-metrics.json | sed -r 's/^[^:]*:(.*)$/\1/' | sed 's/ //g' | sed 's/,//g'`

model_out_dir="$ELECTER_HUL_DIR/$LEGAL_DATA_DIR/model-<code>-output-$MAX_SEQ_LENGTH"

# Perform training
rm -rf $model_out_dir && $ELECTER_DIR/scripts/run-exp-simple.sh -c $ELECTER_DIR/experiment_configs/simple.jsonnet \
-s $model_out_dir --num-epochs 100 --batch-size 1 \
--train-path $EXPERIMENT_DATA_DIR/data-train.p \
--dev-path $EXPERIMENT_DATA_DIR/data-val.p \
--num-train-instances $NUM_TRAIN_INSTANCES --cuda-device 0 --max-seq-len $MAX_SEQ_LENGTH \
--vocab $LEGAL_DATA_DIR/legal-data-vocab/

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
--cuda-device 0 \
--batch-size 32 \
--output-file $LEGAL_DATA_DIR/embeddings-output-gold-docs.jsonl \
--vocab-dir $LEGAL_DATA_DIR/legal-data-vocab/ \
--included-text-fields title

python3 $ELECTER_DIR/scripts/result-analysis.py $LEGAL_DATA_DIR


