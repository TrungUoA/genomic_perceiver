#!/usr/bin/env python3
import os
from os.path import exists
import pandas
import numpy as np
from pandas_plink import read_plink1_bin
import torch
from torch import tensor
from torch.utils import data
import math
from process_snv_mat import get_tok_mat
import pickle

data_dir = os.environ['UKBB_DATA'] + "/"
# gwas_dir = os.environ['UKBB_DATA'] + "/gwas_associated_only/"
gwas_dir = os.environ['UKBB_DATA'] + "/"
plink_base = os.environ['PLINK_FILE']
urate_file = os.environ['URATE_FILE']
MY_DIR = "/data/tugn232/ukbb/"
_ENC_VER = 2

class Tokenised_SNVs:
    def __init__(self, geno, encoding: int):
        # val == 0 means we have two 'a0' vals
        # val == 2 means two 'a1' vals
        # val == 1 means one of each
        tok_mat, tok_to_string, string_to_tok, num_toks = get_tok_mat(geno, encoding)

        self.string_to_tok = string_to_tok
        self.tok_to_string = tok_to_string
        self.tok_mat = tok_mat
        self.num_toks = num_toks
        self.positions = torch.tensor(geno.pos.values, dtype=torch.long)

def read_from_plink(remove_nan=False, small_set=False, subsample_control=True, encoding: int = 2):
    print("using data from:", data_dir)
    bed_file = gwas_dir+plink_base+".bed"
    bim_file = gwas_dir+plink_base+".bim"
    fam_file = gwas_dir+plink_base+".fam"
    print("bed_file:", bed_file)
    geno_tmp = read_plink1_bin(bed_file, bim_file, fam_file)
    geno_tmp["sample"] = pandas.to_numeric(geno_tmp["sample"])
    urate_tmp = pandas.read_csv(data_dir + urate_file)
    withdrawn_ids = pandas.read_csv(data_dir + "w12611_20220222.csv", header=None, names=["ids"])

    usable_ids = list(set(urate_tmp.eid) - set(withdrawn_ids.ids))
    urate = urate_tmp[urate_tmp["eid"].isin(usable_ids)]
    del urate_tmp
    geno = geno_tmp[geno_tmp["sample"].isin(usable_ids)]
    del geno_tmp
    if small_set:
        num_samples = 1000
        num_snps = 200
        geno = geno[0:num_samples, 0:num_snps]
        urate = urate[0:num_samples]

    if (subsample_control):
        gout_cases = urate[urate.gout]["eid"]
        non_gout_cases = urate[urate.gout == False]["eid"]
        # non_gout_cases = np.where(urate.gout == False)[0]
        non_gout_sample = np.random.choice(non_gout_cases, size=len(gout_cases), replace=False)
        sample_ids = list(set(gout_cases).union(non_gout_sample))
        urate = urate[urate["eid"].isin(sample_ids)]
        geno = geno[geno["sample"].isin(sample_ids)]

    geno_mat = geno.values
    positions = np.asarray(geno.pos)

    num_zeros = np.sum(geno_mat == 0)
    num_ones = np.sum(geno_mat == 1)
    num_twos = np.sum(geno_mat == 2)
    num_non_zeros = np.sum(geno_mat != 0)
    num_nan = np.sum(np.isnan(geno_mat))
    total_num = num_zeros + num_non_zeros
    # geno_med = np.bincount(geno_mat)
    values, counts = np.unique(geno_mat, return_counts=True)
    most_common = values[np.argmax(counts)]

    print(
        "geno mat contains {:.2f}% zeros, {:.2f}% ones, {:.2f}% twos {:.2f}% nans".format(
            100.0 * num_zeros / total_num,
            100 * (num_ones / total_num),
            100 * (num_twos / total_num),
            100.0 * num_nan / total_num,
        )
    )
    print("{:.2f}% has gout".format(100 * np.sum(urate.gout) / len(urate)))

    if remove_nan:
        geno_mat[np.isnan(geno_mat)] = most_common

    # we ideally want the position and the complete change
    snv_toks = Tokenised_SNVs(geno, encoding)

    return snv_toks, urate

def check_pos_neg_frac(dataset: data.TensorDataset):
    pos, neg, unknown = 0, 0, 0
    for (_, _, y) in dataset:
        if y == 1:
            pos = pos + 1
        elif y == 0:
            neg = neg + 1
        else:
            unknown = unknown + 1
    return pos, neg, unknown

def get_train_test(geno, pheno, batch_size, test_split, using_torchtensor=False, include_position=False):
    urate = pheno["urate"].values
    gout = pheno["gout"].values
    test_cutoff = (int)(math.ceil(test_split * geno.tok_mat.shape[0]))
    test_cutoff = int( test_cutoff / batch_size + 0.5 ) * batch_size
    test_seqs = geno.tok_mat[:test_cutoff,]
    test_phes = gout[:test_cutoff].astype(np.int32)#[:, None]
    train_seqs = geno.tok_mat[test_cutoff:,]
    train_phes = gout[test_cutoff:,].astype(np.int32)#[:, None]
    print("seqs shape: ", train_seqs.shape)
    print("phes shape: ", train_phes.shape)

    if using_torchtensor:
        if include_position:
            positions = tensor(geno.positions)
            training_dataset = data.TensorDataset(
                positions.repeat(len(train_seqs), 1), tensor(train_seqs), tensor(train_phes, dtype=torch.int64)
            )
            test_dataset = data.TensorDataset(
                positions.repeat(len(test_seqs), 1), tensor(test_seqs), tensor(test_phes, dtype=torch.int64)
            )
        else:
            training_dataset = data.TensorDataset( tensor(train_seqs), tensor(train_phes, dtype=torch.int64) )
            test_dataset = data.TensorDataset( tensor(test_seqs), tensor(test_phes, dtype=torch.int64) )
    else:
        if include_position:
            positions = geno.positions.value
            training_dataset = ( np.tile( positions, [len(train_seqs),1]), train_seqs, train_phes )
            test_dataset = ( np.tile( positions, [len(train_seqs),1]), test_seqs, test_phes )
        else:
            training_dataset = ( train_seqs, train_phes )
            test_dataset = ( test_seqs, test_phes )
    return training_dataset, test_dataset


def get_data(enc_ver=_ENC_VER, test_split=0.3, batch_size=12, save_pickle=False):
    geno_file = MY_DIR + plink_base + '_encv-' + str(enc_ver) + '_geno_cache.pickle'
    pheno_file = MY_DIR + plink_base +'_encv-' + str(enc_ver) +  '_pheno_cache.pickle'
    if exists(geno_file) and exists(pheno_file):
        with open(geno_file, "rb") as f:
            geno = pickle.load(f)
        with open(pheno_file, "rb") as f:
            pheno = pickle.load(f)
    else:
        # geno, pheno = read_from_plink(small_set=True)
        print("reading data from plink")
        geno, pheno = read_from_plink(small_set=False, subsample_control=True, encoding=enc_ver)
        # geno_preprocessed_file =
        print("done")
        if save_pickle:
            print("writing to pickle")
            with open(geno_file, "wb") as f:
                pickle.dump(geno, f, pickle.HIGHEST_PROTOCOL)
            with open(pheno_file, "wb") as f:
                pickle.dump(pheno, f, pickle.HIGHEST_PROTOCOL)
            print("done")

    train, test = get_train_test(geno, pheno, batch_size, test_split)
    return train, test, geno, pheno, enc_ver

def load_vocab_size():
    geno_file = MY_DIR + plink_base + '_encv-' + str(_ENC_VER) + '_geno_cache.pickle'
    if exists(geno_file):
        with open(geno_file, "rb") as f:
            tokenizer = pickle.load(f)
            return tokenizer.num_toks
    return 32


if __name__ == "__main__":
    train, test, geno, pheno, enc_ver = get_data(enc_ver=_ENC_VER, test_split=0.3, batch_size=12, save_pickle=True)
    print("test")
