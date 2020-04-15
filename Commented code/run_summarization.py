#! /usr/bin/python3
import argparse
import logging
import os
import sys
from collections import namedtuple

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from modeling_bertabs import BertAbs, build_predictor
from transformers import BertTokenizer

from .utils_summarization import (
    CNNDMDataset,
    build_mask,
    compute_token_type_ids,
    encode_for_summarization,
    truncate_or_pad,
)


logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


Batch = namedtuple("Batch", ["document_names", "batch_size", "src", "segs", "mask_src", "tgt_str"])

# store the vocabulary for each model and provide methods for encoding/decoding strings in list of token embeddings indices to be fed to a model
def evaluate(args):
    #  store the vocabulary for each model and provide methods for encoding/decoding strings in list of token embeddings indices to be fed to a model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    # runs bertabs-finetuned-cnndm-extractive-abstractive-summarization model
    model = BertAbs.from_pretrained("bertabs-finetuned-cnndm")
    model.to(args.device)
    model.eval()

    symbols = {
        "BOS": tokenizer.vocab["[unused0]"],
        "EOS": tokenizer.vocab["[unused1]"],
        "PAD": tokenizer.vocab["[PAD]"],
    }

    if args.compute_rouge:
        reference_summaries = []
        generated_summaries = []

        import rouge
        import nltk

        nltk.download("punkt")
        # creates rouge evaluator for model evaluation
        rouge_evaluator = rouge.Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=args.beam_size,
            length_limit_type="words",
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
            stemming=True,
        )

    # these (unused) arguments are defined to keep the compatibility
    # with the legacy code and will be deleted in a next iteration.
    args.result_path = ""
    args.temp_dir = ""

    # creates embeddings
    data_iterator = build_data_iterator(args, tokenizer)
    # creates score for each sentence within document using GNMTGlobalScorer class in modeling_bertabs.py
    predictor = build_predictor(args, tokenizer, symbols, model)

    # store model summary within log files that are stored every n steps
    logger.info("***** Running evaluation *****")
    logger.info("  Number examples = %d", len(data_iterator.dataset))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("")
    logger.info("***** Beam Search parameters *****")
    logger.info("  Beam size = %d", args.beam_size)
    logger.info("  Minimum length = %d", args.min_length)
    logger.info("  Maximum length = %d", args.max_length)
    logger.info("  Alpha (length penalty) = %.2f", args.alpha)
    logger.info("  Trigrams %s be blocked", ("will" if args.block_trigram else "will NOT"))

    
    for batch in tqdm(data_iterator):
        # Generates summaries from one batch of data.
        batch_data = predictor.translate_batch(batch)
        # compare translations to pretrained translations
        translations = predictor.from_batch(batch_data)
        # transforms the output of the `from_batch` function into nicely formatted summaries.
        summaries = [format_summary(t) for t in translations]
        save_summaries(summaries, args.summaries_output_dir, batch.document_names)

        # calculate rouge score for predicted vs actual summaries
        if args.compute_rouge:
            reference_summaries += batch.tgt_str
            generated_summaries += summaries
    
    # calculate rouge score for predicted vs actual summaries
    if args.compute_rouge:
        scores = rouge_evaluator.get_scores(generated_summaries, reference_summaries)
        str_scores = format_rouge_scores(scores)
        save_rouge_scores(str_scores)
        print(str_scores)


def save_summaries(summaries, path, original_document_name):
    """ Write the summaries in fies that are prefixed by the original
    files' name with the `_summary` appended.

    Attributes:
        original_document_names: List[string]
            Name of the document that was summarized.
        path: string
            Path were the summaries will be written
        summaries: List[string]
            The summaries that we produced.
    """
    for summary, document_name in zip(summaries, original_document_name):
        # Prepare the summary file's name
        if "." in document_name:
            bare_document_name = ".".join(document_name.split(".")[:-1])
            extension = document_name.split(".")[-1]
            name = bare_document_name + "_summary." + extension
        else:
            name = document_name + "_summary"

        file_path = os.path.join(path, name)
        with open(file_path, "w") as output:
            output.write(summary)


def format_summary(translation):
    """ Transforms the output of the `from_batch` function
    into nicely formatted summaries.
    """
    raw_summary, _, _ = translation
    summary = (
        raw_summary.replace("[unused0]", "")
        .replace("[unused3]", "")
        .replace("[PAD]", "")
        .replace("[unused1]", "")
        .replace(r" +", " ")
        .replace(" [unused2] ", ". ")
        .replace("[unused2]", "")
        .strip()
    )

    return summary

# create print friendly format for rouge scores
def format_rouge_scores(scores):
    return """\n
****** ROUGE SCORES ******

** ROUGE 1
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}

** ROUGE 2
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}

** ROUGE L
F1        >> {:.3f}
Precision >> {:.3f}
Recall    >> {:.3f}""".format(
        scores["rouge-1"]["f"],
        scores["rouge-1"]["p"],
        scores["rouge-1"]["r"],
        scores["rouge-2"]["f"],
        scores["rouge-2"]["p"],
        scores["rouge-2"]["r"],
        scores["rouge-l"]["f"],
        scores["rouge-l"]["p"],
        scores["rouge-l"]["r"],
    )


# save rouge scores
def save_rouge_scores(str_scores):
    with open("rouge_scores.txt", "w") as output:
        output.write(str_scores)


#
# LOAD the dataset
#


def build_data_iterator(args, tokenizer):
    # get dataset
    dataset = load_and_cache_examples(args, tokenizer)
    # Samples elements sequentially, always in the same order.
    sampler = SequentialSampler(dataset)

    # format the data passed into the data loader.
    def collate_fn(data):
        return collate(data, tokenizer, block_size=512, device=args.device)

    # PyTorch data loading utility that represents a Python iterable over a dataset
    iterator = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn,)

    return iterator


def load_and_cache_examples(args, tokenizer):
    dataset = CNNDMDataset(args.documents_dir)
    return dataset


def collate(data, tokenizer, block_size, device):
    """ Collate formats the data passed to the data loader.

    In particular we tokenize the data batch after batch to avoid keeping them
    all in memory. We output the data as a namedtuple to fit the original BertAbs's
    API.
    """
    data = [x for x in data if not len(x[1]) == 0]  # remove empty_files
    names = [name for name, _, _ in data]
    summaries = [" ".join(summary_list) for _, _, summary_list in data]

    encoded_text = [encode_for_summarization(story, summary, tokenizer) for _, story, summary in data]
    encoded_stories = torch.tensor(
        [truncate_or_pad(story, block_size, tokenizer.pad_token_id) for story, _ in encoded_text]
    )
    encoder_token_type_ids = compute_token_type_ids(encoded_stories, tokenizer.cls_token_id)
    encoder_mask = build_mask(encoded_stories, tokenizer.pad_token_id)

    batch = Batch(
        document_names=names,
        batch_size=len(encoded_stories),
        src=encoded_stories.to(device),
        segs=encoder_token_type_ids.to(device),
        mask_src=encoder_mask.to(device),
        tgt_str=summaries,
    )

    return batch


def decode_summary(summary_tokens, tokenizer):
    """ Decode the summary and return it in a format
    suitable for evaluation.
    """
    summary_tokens = summary_tokens.to("cpu").numpy()
    summary = tokenizer.decode(summary_tokens)
    sentences = summary.split(".")
    sentences = [s + "." for s in sentences]
    return sentences

# identify parameters to run within scripts
def main():
    """ The main function defines the interface with the users.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--documents_dir",
        default=None,
        type=str,
        required=True,
        help="The folder where the documents to summarize are located.",
    )
    parser.add_argument(
        "--summaries_output_dir",
        default=None,
        type=str,
        required=False,
        help="The folder in wich the summaries should be written. Defaults to the folder where the documents are",
    )
    parser.add_argument(
        "--compute_rouge",
        default=False,
        type=bool,
        required=False,
        help="Compute the ROUGE metrics during evaluation. Only available for the CNN/DailyMail dataset.",
    )
    # EVALUATION options
    parser.add_argument(
        "--no_cuda", default=False, type=bool, help="Whether to force the execution on CPU.",
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    # BEAM SEARCH arguments
    parser.add_argument(
        "--min_length", default=50, type=int, help="Minimum number of tokens for the summaries.",
    )
    parser.add_argument(
        "--max_length", default=200, type=int, help="Maixmum number of tokens for the summaries.",
    )
    parser.add_argument(
        "--beam_size", default=5, type=int, help="The number of beams to start with for each example.",
    )
    parser.add_argument(
        "--alpha", default=0.95, type=float, help="The value of alpha for the length penalty in the beam search.",
    )
    parser.add_argument(
        "--block_trigram",
        default=True,
        type=bool,
        help="Whether to block the existence of repeating trigrams in the text generated by beam search.",
    )
    args = parser.parse_args()

    # Select device (distibuted not available)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Check the existence of directories
    if not args.summaries_output_dir:
        args.summaries_output_dir = args.documents_dir

    if not documents_dir_is_valid(args.documents_dir):
        raise FileNotFoundError(
            "We could not find the directory you specified for the documents to summarize, or it was empty. Please specify a valid path."
        )
    os.makedirs(args.summaries_output_dir, exist_ok=True)

    evaluate(args)

# verify that path for the dataset exists
def documents_dir_is_valid(path):
    if not os.path.exists(path):
        return False

    file_list = os.listdir(path)
    if len(file_list) == 0:
        return False

    return True


if __name__ == "__main__":
    main()
