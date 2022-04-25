#!/usr/bin/env python3


import os

import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from utils.functions import save_model, load_model
from utils.lstm_utils import LM
from models.asr.transformer import Transformer, Encoder, Decoder
from utils import constant
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer, calculate_cer_en_zh

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1


def evaluate(model, test_loader, lm=None):
    """
    Evaluation
    args:
        model: Model object
        test_loader: DataLoader object
    """
    model.eval()

    total_word, total_char, total_cer, total_wer = 0, 0, 0, 0
    total_en_cer, total_zh_cer, total_en_char, total_zh_char = 0, 0, 0, 0

    with torch.no_grad():
        test_pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
        for i, (data) in enumerate(test_pbar):
            src, tgt, src_percentages, src_lengths, tgt_lengths = data

            if constant.USE_CUDA:
                src = src.cuda()
                tgt = tgt.cuda()

            batch_ids_hyps, batch_strs_hyps, batch_strs_gold = model.evaluate(
                src, src_lengths, tgt, beam_search=constant.args.beam_search, beam_width=constant.args.beam_width,
                beam_nbest=constant.args.beam_nbest, lm=lm, lm_rescoring=constant.args.lm_rescoring,
                lm_weight=constant.args.lm_weight, c_weight=constant.args.c_weight, verbose=constant.args.verbose)

            for x in range(len(batch_strs_gold)):
                hyp = batch_strs_hyps[x].replace(constant.EOS_CHAR, "").replace(constant.SOS_CHAR, "").replace(
                    constant.PAD_CHAR, "")
                print(hyp)
                gold = batch_strs_gold[x].replace(constant.EOS_CHAR, "").replace(constant.SOS_CHAR, "").replace(
                    constant.PAD_CHAR, "")

                wer = calculate_wer(hyp, gold)
                cer = calculate_cer(hyp.strip(), gold.strip())

                en_cer, zh_cer, num_en_char, num_zh_char = calculate_cer_en_zh(hyp, gold)
                total_en_cer += en_cer
                total_zh_cer += zh_cer
                total_en_char += num_en_char
                total_zh_char += num_zh_char

                total_wer += wer
                total_cer += cer
                total_word += len(gold.split(" "))
                total_char += len(gold)

            test_pbar.set_description("TEST CER:{:.2f}% WER:{:.2f}% CER_EN:{:.2f}% CER_ZH:{:.2f}%".format(
                total_cer * 100 / total_char, total_wer * 100 / total_word, total_en_cer * 100 / max(1, total_en_char),
                total_zh_cer * 100 / max(1, total_zh_char)))


load_path = "save/best_model.th"
model, opt, epoch, metrics, loaded_args, label2id, id2label = load_model(load_path)

audio_conf = dict(sample_rate=loaded_args.sample_rate,
                  window_size=loaded_args.window_size,
                  window_stride=loaded_args.window_stride,
                  window=loaded_args.window,
                  noise_dir=loaded_args.noise_dir,
                  noise_prob=loaded_args.noise_prob,
                  noise_levels=(loaded_args.noise_min, loaded_args.noise_max))

test_manifest_path = ["manifests/aishell_test_lang_manifest.csv"]

test_data = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath_list=test_manifest_path, label2id=label2id,
                               normalize=True, augment=False)

test_sampler = BucketingSampler(test_data, batch_size=1)
test_loader = AudioDataLoader(test_data, num_workers=2, batch_sampler=test_sampler)

# print('语音识别最终结果：\n',res)

lm = None


if constant.args.lm_rescoring:
    lm = LM(constant.args.lm_path)

    # print(model)

evaluate(model, test_loader, lm=lm)