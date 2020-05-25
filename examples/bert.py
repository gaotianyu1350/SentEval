from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import argparse

# import senteval
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'

sys.path.insert(0, PATH_SENTEVAL)
import senteval
import transformers
from transformers import BertModel, BertTokenizer

def prepare(params, samples):
    pass

def batcher(params, batch):
    global args
    if args.avg:
        sentences = [' '.join(s) for s in batch]
        batch = params['tokenizer'].batch_encode_plus(
                sentences,
                add_special_tokens=True,
                max_length=128,
                pad_to_max_length=True,
                return_tensors='pt',
                return_attention_masks=True)
        input_ids = batch['input_ids'].cuda()
        att_mask = batch['attention_mask'].cuda()
        with torch.no_grad():
            seq, _ = params['model'](input_ids, attention_mask=att_mask) # (B, L, H)
            emb = []
            length = (att_mask == 1).sum(-1)
            for i in range(seq.size(0)):
                emb.append(seq[i][1:length[i]-1].mean(0))
            emb = torch.stack(emb)

        return emb.cpu()
    else:
        sentences = [' '.join(s) for s in batch]
        batch = params['tokenizer'].batch_encode_plus(
                sentences,
                add_special_tokens=True,
                max_length=128,
                pad_to_max_length=True,
                return_tensors='pt',
                return_attention_masks=True)
        input_ids = batch['input_ids'].cuda()
        att_mask = batch['attention_mask'].cuda()
        with torch.no_grad():
            _, embeddings = params['model'](input_ids, attention_mask=att_mask)

        return embeddings.cpu()

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='bert-base-uncased', help='model name or path')
    parser.add_argument('--linguistic', action='store_true')
    parser.add_argument('--avg', action='store_true')
    global args
    args = parser.parse_args()

    model = BertModel.from_pretrained(args.model)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    params_senteval['model'] = model.cuda()
    params_senteval['tokenizer'] = tokenizer

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']
    
    linguistic_tasks = ['SentLen', 'WC', 'TreeDepth', 'TopConst', 'BShift', 'Tense', 'SubjNum', 'ObjNum', 'SOMO', 'Coordlnv']
    
    if not args.linguistic:
        results = se.eval(transfer_tasks)

        sts_task_list = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        other_task_list = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKEntailment']

        # STS pearson
        print("######### STS pearson ###########")
        summ = 0
        s_title = ''
        s_result = ''
        for task in sts_task_list:
            try:
                task_result = results[task]['all']['pearson']['mean'] * 100
            except:
                task_result = results[task]['pearson'] * 100
            s_title += '%6s' % (task[:5])
            s_result += ' %.2f' % (task_result)
            summ += task_result
        print(s_title)
        print(s_result)
        print('avg: %.2f' % (summ / len(sts_task_list)))

        # STS spearman
        print("######### STS spearman ###########")
        summ = 0
        s_title = ''
        s_result = ''
        for task in sts_task_list:
            try:
                task_result = results[task]['all']['spearman']['mean'] * 100
            except:
                task_result = results[task]['spearman'] * 100
            s_title += '%6s' % (task[:5])
            s_result += ' %.2f' % (task_result)
            summ += task_result
        print(s_title)
        print(s_result)
        print('avg: %.2f' % (summ / len(sts_task_list)))

        # Other
        print("######### Other ###########")
        summ = 0
        s_title = ''
        s_result = ''
        for task in other_task_list:
            task_result = results[task]['acc']
            s_title += '%6s' % (task[:5])
            s_result += ' %.2f' % (task_result)
            summ += task_result
        print(s_title)
        print(s_result)
        print('avg: %.2f' % (summ / len(other_task_list)))

    else:
        results = se.eval(transfer_tasks)
        print(result)
        #print("######### Linguistic ###########")
        #summ = 0
        #s_title = ''
        #s_result = ''
        #for task in linguistic_tasks:
        #    task_result = results[task]['acc']
        #    s_title += '%6s' % (task[:5])
        #    s_result += ' %.2f' % (task_result)
        #    summ += task_result
        #print(s_title)
        #print(s_result)
        #print('avg: %.2f' % (summ / len(other_task_list)))

