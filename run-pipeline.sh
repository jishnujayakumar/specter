export ELECTER_DIR=`pwd`
chmod a+x pre-experiment/preprocess-legal-data.sh
export LEGAL_DATA_DIR=$1

# Preprocess Data
./pre-experiment/preprocess-legal-data.sh $LEGAL_DATA_DIR $2 $3

# Create triplets files
python specter/data_utils/create_training_files.py \
--data-dir $LEGAL_DATA_DIR/preProcessedData \
--metadata $LEGAL_DATA_DIR/preProcessedData/metadata.json \
--outdir $LEGAL_DATA_DIR/preProcessedData/experimentData \
--bert_vocab pre-experiment/legal-bert-base-uncased/vocab.txt \
--included-text-fields title \
--ratio_hard_negatives 0.4

# Perform training
rm -rf $LEGAL_DATA_DIR-model-output/ && ./scripts/run-exp-simple.sh -c experiment_configs/simple.jsonnet \
-s $LEGAL_DATA_DIR-model-output/ --num-epochs 10 --batch-size 16 \
--train-path $LEGAL_DATA_DIR/preProcessedData/experimentData/data-train.p \
--dev-path $LEGAL_DATA_DIR/preProcessedData/experimentData/data-val.p \
--num-train-instances 535 --cuda-device -1

# Move model artifacts to appropriate tar.gz file
cd $LEGAL_DATA_DIR-model-output/
mv best.th weights.th
mkdir -p model
mv weights.th config.json model/
mv vocabulary/ model/vocabulary/
tar czC model . --transform='s,^\./,,' >| model.tar.gz

# Run inference on gold-docs
cd $ELECTER_DIR
CUDA_VISIBLE_DEVICES=1 \
python scripts/embed.py \
--ids $LEGAL_DATA_DIR/Gold-Score-Docs/gold-docs.txt \
--model $LEGAL_DATA_DIR-model-output/model.tar.gz \
--metadata $LEGAL_DATA_DIR/Gold-Score-Docs/metadata.json \
--cuda-device -1 \
--batch-size 32 \
--output-file $LEGAL_DATA_DIR/embeddings-output-gold-docs.jsonl \
--vocab-dir $LEGAL_DATA_DIR/legal-data-vocab/


