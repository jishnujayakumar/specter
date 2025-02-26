# Setup necessary environment vars 
export ELECTER_DIR=`pwd`
export ELECTER_HULK_DIR=`/home/jishnu-paheli/HULK_JISHNU/ELECTER`
export LEGAL_DATA_DIR=$1
export MAX_SEQ_LENGTH=512
export BERT_VARIANT=$4
export SHB_BERT_VARIANT="distilbert-base-uncased"
export SPQ=$5
export MARGIN_FRACTION=$6
export OUT_FILE="$BERT_VARIANT-SPQ-$SPQ-MF-$MARGIN_FRACTION"
export EXPERIMENT_DATA_DIR="$LEGAL_DATA_DIR/preProcessedData/experimentData-$OUT_FILE"
export EMB_MODEL=$7 # {specter, hecter}

# Get legal-bert
cd $ELECTER_DIR/pre-experiment/
python get-legal-bert.py 

cd $ELECTER_DIR

# Preprocess casetext dataset to generate vocab, t
chmod a+x pre-experiment/preprocess-legal-data.sh
./pre-experiment/preprocess-legal-data.sh $LEGAL_DATA_DIR 0.9 1 true

# Create triplets files
python $ELECTER_DIR/$EMB_MODEL/data_utils/create_training_files.py \
--data-dir $LEGAL_DATA_DIR/preProcessedData \
--metadata $LEGAL_DATA_DIR/preProcessedData/metadata.json \
--outdir $EXPERIMENT_DATA_DIR \
--included-text-fields title \
--ratio_hard_negatives 0.5 \
--samples_per_query $SPQ \
--margin_fraction $MARGIN_FRACTION \
--bert_vocab $ELECTER_DIR/pre-experiment/custom-legalbert/vocab.txt
# --bert_vocab $ELECTER_DIR/pre-experiment/legal-bert-base-uncased/vocab.txt

NUM_TRAIN_INSTANCES=`grep 'train' $EXPERIMENT_DATA_DIR/data-metrics.json | sed -r 's/^[^:]*:(.*)$/\1/' | sed 's/ //g' | sed 's/,//g'`

model_out_dir="$ELECTER_HULK_DIR/$LEGAL_DATA_DIR/model-output-$OUT_FILE"

# Perform training
rm -rf $model_out_dir && $ELECTER_DIR/scripts/run-exp-simple.sh \
-c $ELECTER_DIR/experiment_configs/$EMB_MODEL.jsonnet \
-s $model_out_dir --num-epochs 100 --batch-size 4 \
--train-path $EXPERIMENT_DATA_DIR/data-train.p \    
--dev-path $EXPERIMENT_DATA_DIR/data-val.p \
--num-train-instances $NUM_TRAIN_INSTANCES \
--cuda-device 0 --max-seq-len $MAX_SEQ_LENGTH \
--vocab $LEGAL_DATA_DIR/legal-data-vocab

# Move model artifacts to appropriate tar.gz file
cd $model_out_dir
mv best.th weights.th
mkdir -p model
mv weights.th config.json model/
mv vocabulary/ model/vocabulary/
tar cvzC model . --transform='s,^\./,,' >| model.tar.gz

# Run inference on gold-docs
cd $ELECTER_DIR
CUDA_VISIBLE_DEVICES=1 \
python $ELECTER_DIR/scripts/embed.py \
--ids $LEGAL_DATA_DIR/Gold-Score-Docs/gold-docs.txt \
--model $model_out_dir/model.tar.gz \
--model_type $EMB_MODEL \
--metadata $LEGAL_DATA_DIR/Gold-Score-Docs/metadata.json \
--cuda-device -1 \
--batch-size 32 \
--output-file $LEGAL_DATA_DIR/embeddings-output-gold-docs.jsonl \
--vocab-dir $LEGAL_DATA_DIR/legal-data-vocab/ \
--included-text-fields title

python3 $ELECTER_DIR/scripts/result-analysis.py $LEGAL_DATA_DIR


