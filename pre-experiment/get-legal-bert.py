import os
import tarfile
import logging
from transformers import AutoTokenizer, AutoModel


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def saveTarGZ(filename):
    """
    Save to tar.gz
    """
    tar = tarfile.open(f"{filename}.tar.gz", "w:gz")
    for name in ["config.json", "pytorch_model.bin", "vocab.txt"]:
        tar.add(f"./{filename}/{name}")
    tar.close()


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
    logging.info(f"Retrieving {model}'s' tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model)

    logging.info(f"Retrieving {model}'s model")
    mdl = AutoModel.from_pretrained(model)

    filename = f"{model.replace('/', '-')}"

    logging.info("Saving tokenizer and model to disk")
    for artifact in [mdl, tokenizer]:
        saveToDisk(artifact, filename)

    """
    Uncomment to try out raw download of vocab.txt
    """
    # for category in ["small", "base"]:
    #     logging.info(f"Downloading {category}: vocab.txt")
    #     os.system(f"mkdir -p {filename}-vocab && wget -O {filename}-vocab/vocab.txt \
    #         https://huggingface.co/nlpaueb/legal-bert-{category}-uncased/raw/main/vocab.txt")

    logging.info(f"Saving to {filename}.tar.gz")
    saveTarGZ(filename)

    logging.info("Deleting {filename} directory")
    os.system(f"rm -rf {filename}/")


if __name__ == "__main__":
    for model in ["nlpaueb/legal-bert-small-uncased",
                  "nlpaueb/legal-bert-base-uncased"]:
        prepareLegalBertArtifacts(model)
