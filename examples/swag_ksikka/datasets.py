"""
Code to create data different 
"""
import os
import csv
import pickle
import torch
import torchvision
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer


def _swag(data_dir, split):
    sample = {}
    data = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    sample["context"] = data["startphrase"].values.tolist()
    sample["labels"] = data["label"].values.tolist()
    sample["answers"] = []
    for idx, it in data.iterrows():
        _answers = []
        for i in range(4):
            _answers.append(it[f"ending{i}"])
        sample["answers"].append(_answers)
    return sample


def swag(data_dir):
    train_data = _swag(data_dir, "train")
    val_data = _swag(data_dir, "val")
    # test_data = _swag(data_dir, "valid")
    return (train_data, val_data)


def encode_dataset2(*splits, encoder):
    encoded_splits = []
    for split_id, split in enumerate(splits):
        print(f"Encoding to tokens and ids for split {split_id}")
        curr_split = split
        curr_split["context"] = [
            encoder.convert_tokens_to_ids(encoder.tokenize(ii))
            for ii in split["context"]
        ]
        for idx, _ans in enumerate(split["answers"]):
            curr_split["answers"][idx] = [
                encoder.convert_tokens_to_ids(encoder.tokenize(ii)) for ii in _ans
            ]

        encoded_splits.append(curr_split)

    return encoded_splits


def transform_story(data_raw, encoder, max_len=0):
    """
    Transforms indexed tokens into a BERT input [CLS]...context...[SEP].....question....[SEP].....answer
    Output for a single (Q,A,C) is 4 x 512 since separate inputs are generated for each answer
    """
    pad_token_id = encoder.pad_token_id
    sep_token_id = encoder.sep_token_id
    cls_token_id = encoder.cls_token_id
    # TODO: Truncation for max-length
    max_len = np.minimum(encoder.max_len, 512)
    # assert max_len > 320, "truncation is not implemented, 320 is max-length for input seq"
    n_batch = len(data_raw["answers"])

    data_tx = pad_token_id * np.ones((n_batch, 4, max_len)).astype("int32")
    attention_masks = np.zeros((n_batch, 4, max_len)).astype("int32")
    labels = np.zeros((n_batch)).astype("int32")
    max_length_in_data = 0

    for b in tqdm(range(n_batch), desc="Encoding a split"):
        labels[b] = int(data_raw["labels"][b])
        context = data_raw["context"][b] + [sep_token_id]
        answers = [data_raw["answers"][b][ii] + [sep_token_id] for ii in range(4)]
        label = data_raw["labels"][b]
        lc = len(context)

        pos = 0
        data_tx[b, :, pos] = cls_token_id
        pos += 1
        data_tx[b, :, pos : pos + lc] = context
        pos += lc
        for ii in range(4):
            _answer = answers[ii]
            la = len(_answer)
            data_tx[b, ii, pos : pos + la] = _answer
            max_length_in_data = np.maximum(max_length_in_data, pos + la)
            attention_masks[b, ii, : pos + la] = 1
    return data_tx, labels, attention_masks


def get_swag_data(data_dir, tokenizer, num_validation_samples=500):
    """
    Directly returns Pytorch style Dataset for train and val
    """
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    curr_data_dir = "data"

    data_path = os.path.join(curr_data_dir, "data_swag.pkl")
    # Transformed data
    tx_data_path = os.path.join(curr_data_dir, "data_swag_transformed.pkl")

    if not os.path.exists(tx_data_path):
        if not os.path.exists(data_path):
            # Encode data
            print("Encoding dataset")
            train_data, val_data = encode_dataset2(*swag(data_dir), encoder=tokenizer)
            pickle.dump(
                (train_data, val_data), open(data_path, "wb"),
            )
        else:
            train_data, val_data = pickle.load(open(data_path, "rb"))

        print("Transforming dataset")
        train_data_tx, train_labels, train_attention_masks = transform_story(
            train_data, tokenizer, max_len=150
        )
        val_data_tx, val_labels, val_attention_masks = transform_story(
            val_data, tokenizer, max_len=150
        )
        pickle.dump(
            (
                train_data_tx,
                train_labels,
                train_attention_masks,
                val_data_tx,
                val_labels,
                val_attention_masks,
            ),
            open(tx_data_path, "wb"),
        )
    else:
        (
            train_data_tx,
            train_labels,
            train_attention_masks,
            val_data_tx,
            val_labels,
            val_attention_masks,
        ) = pickle.load(open(tx_data_path, "rb"))

    dataset_train = TensorDataset(
        torch.Tensor(train_data_tx).long(),
        torch.Tensor(train_labels).long(),
        torch.Tensor(train_attention_masks).long(),
    )
    dataset_val = TensorDataset(
        torch.Tensor(val_data_tx).long()[:num_validation_samples],
        torch.Tensor(val_labels).long()[:num_validation_samples],
        torch.Tensor(val_attention_masks).long()[:num_validation_samples],
    )

    print("Loaded the dataset")
    return dataset_train, dataset_val


if __name__ == "__main__":
    print("running")
    data_dir = "../../data/swag"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset_train, dataset_val = get_swag_data(
        data_dir, tokenizer, num_validation_samples=100
    )
    print("running")
