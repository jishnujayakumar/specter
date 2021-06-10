import os
import tarfile
import logging
from transformers import AutoTokenizer, AutoModel


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def saveToDisk(transformerObj, filename):
    """
    Saves transformer objects like model, tokenizer to disk
    """
    transformerObj.save_pretrained(filename)


def prepareLegalBertArtifacts(model):
    """
    Prepare compressed legalBert artifacts as present in
    scibert.tar.gz (from: archive.tar.gz)
    """
    logging.info(f"Retrieving {model}'s tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model, force_download=True)

    logging.info(f"Retrieving {model}'s model")
    mdl = AutoModel.from_pretrained(model, force_download=True)

    filename = f"{model.split('/')[-1]}"

    logging.info("Saving tokenizer and model to disk")
    for artifact in [mdl, tokenizer]:
        saveToDisk(artifact, filename)

    logging.info(f"Saving to {filename}.tar.gz")
    os.system(f"mv ./{filename}/config.json ./{filename}/bert_config.json && \
        tar czC {filename} . --transform='s,^\./,,' >| {filename}.tar.gz")

    logging.info("Placing files in proper directory")
    os.system(f"find {filename} -type f ! -name 'vocab.txt' -delete && \
        mv {filename}.tar.gz {filename}/")


if __name__ == "__main__":
    for model in ["nlpaueb/legal-bert-small-uncased",
                  "nlpaueb/legal-bert-base-uncased"]:
        prepareLegalBertArtifacts(model)
