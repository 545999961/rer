import copy
import sys
from typing import Union, List, Tuple, Any

import numpy as np
import torch
from peft import PeftModel
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, is_torch_npu_available
from mistral_model import CostWiseMistralForCausalLM, CostWiseHead
from mistral_config import CostWiseMistralConfig
import warnings
from torch.utils.data import Dataset
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def last_logit_pool(logits: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i]] for i in range(batch_size)], dim=0)

def set_nested_attr(obj, attr, value):
    attributes = attr.split('.')
    for attribute in attributes[:-1]:
        obj = getattr(obj, attribute)
    setattr(obj, attributes[-1], value)


def get_nested_attr(obj, attr):
    attributes = attr.split('.')
    for attribute in attributes:
        obj = getattr(obj, attribute)
    return obj

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class DatasetForReranker(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer_path: str,
            max_len: int = 512,
            query_prefix: str = 'A: ',
            passage_prefix: str = 'B: ',
            cache_dir: str = None,
            prompt: str = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                       trust_remote_code=True,
                                                       cache_dir=cache_dir)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        self.dataset = dataset
        self.max_len = max_len
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.total_len = len(self.dataset)

        if prompt is None:
            prompt = "Predict whether passage B contains an answer to query A."
        self.prompt_inputs = self.tokenizer(prompt,
                                            return_tensors=None,
                                            add_special_tokens=False)['input_ids']
        sep = "\n"
        self.sep_inputs = self.tokenizer(sep,
                                         return_tensors=None,
                                         add_special_tokens=False)['input_ids']

        self.encode_max_length = self.max_len - len(self.sep_inputs) - len(self.prompt_inputs)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        query, passage = self.dataset[item]
        query = self.query_prefix + query
        passage = self.passage_prefix + passage
        query_inputs = self.tokenizer(query,
                                      return_tensors=None,
                                      add_special_tokens=False,
                                      max_length=32,
                                      # max_length=self.max_len * 3 // 4,
                                      # padding='max_length',
                                      truncation=True)
        passage_inputs = self.tokenizer(passage,
                                        return_tensors=None,
                                        add_special_tokens=False,
                                        max_length=self.max_len,
                                        truncation=True)
        item = self.tokenizer.prepare_for_model(
            [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
            self.sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=self.encode_max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
        item['input_ids'] = item['input_ids'] + self.sep_inputs + self.prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        item.pop('token_type_ids') if 'token_type_ids' in item.keys() else None
        if 'position_ids' in item.keys():
            item['position_ids'] = list(range(len(item['input_ids'])))

        return item, len([self.tokenizer.bos_token_id] + query_inputs['input_ids'] + self.sep_inputs), len(self.sep_inputs + self.prompt_inputs)

class collater():
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_multiple_of = 8
        self.label_pad_token_id = -100
        warnings.filterwarnings("ignore",
                                message="`max_length` is ignored when `padding`=`True` and there is no truncation strategy.")

    def __call__(self, data):
        # print(self.tokenizer.padding_side)
        # features = [feature[0] for feature in data]
        # query_lengths = [feature[1] for feature in data]
        # prompt_lengths = [feature[2] for feature in data]
        features = data[0]
        query_lengths = data[1]
        prompt_lengths = data[2]

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        collected = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_len,
            pad_to_multiple_of=8,
            return_tensors='pt',
        )

        return collected, query_lengths, prompt_lengths


class MatroyshkaReranker:
    def __init__(
            self,
            model_name_or_path: str = None,
            peft_name_or_path: List[str] = None,
            use_fp16: bool = False,
            use_bf16: bool = False,
            cache_dir: str = None,
            device: int = 0,
            compress_ratio: int = 0,
            compress_layers: List[int] = 6,
            cutoff_layers: List[int] = None,
            layer_wise: bool = False,
            start_layer: int = 4
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       cache_dir=cache_dir,
                                                       trust_remote_code=True)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        config = CostWiseMistralConfig.from_pretrained(model_name_or_path,
                                                     cache_dir=cache_dir,
                                                     trust_remote_code=True)

        self.model = CostWiseMistralForCausalLM.from_pretrained(model_name_or_path,
                                                               config=config,
                                                               cache_dir=cache_dir,
                                                               trust_remote_code=True,
                                                               attn_implementation='sdpa',
                                                               torch_dtype=torch.bfloat16 if use_bf16 else torch.float32)
        if layer_wise:
            lm_head = nn.ModuleList([CostWiseHead(
                self.model.config.hidden_size, 1) for _ in range(
                start_layer,
                self.model.config.num_hidden_layers + 1,
                1)])
            state_dict_back = self.model.lm_head.state_dict()
            state_dict_back['weight'] = state_dict_back['weight'][self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
                                                                  : self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0] + 1, :]
            for i in range(len(lm_head)):
                lm_head[i].linear_head.load_state_dict(state_dict_back)
            self.model.set_output_embeddings(lm_head)
            self.model.config.start_layer = start_layer
            self.model.config.layer_sep = 1
            self.model.config.layer_wise = True
        if peft_name_or_path is not None:
            for peft_name in peft_name_or_path:
                self.model = PeftModel.from_pretrained(self.model, peft_name, token='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', cache_dir=cache_dir)
                self.model = self.model.merge_and_unload()

        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir

        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            use_fp16 = False
        if use_fp16 and use_bf16 is False:
            self.model.half()

        self.model = self.model.to(self.device)

        self.model.eval()

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        self.compress_ratio = compress_ratio
        self.compress_layer = compress_layers
        self.cutoff_layers = cutoff_layers
        self.layer_wise = layer_wise

    @torch.no_grad()
    def compute_score(self, sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], batch_size: int = 16,
                      max_length: int = 512, prompt: str = None,
                      normalize: bool = False, use_dataloader: bool = False) -> Union[List[float], List[List[float]]]:
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        length_sorted_idx = np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs])
        sentences_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]

        if use_dataloader:
            dataset = DatasetForReranker(sentences_sorted,
                                        self.model_name_or_path,
                                        max_length,
                                        cache_dir=self.cache_dir,
                                        prompt=prompt)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=False,
                                    num_workers=min(batch_size, 16),
                                    collate_fn=collater(self.tokenizer, max_length))

            all_scores = []
            for data in tqdm(dataloader):
                inputs = data[0]
                query_lengths = data[1]
                prompt_lengths = data[2]
                inputs = inputs.to(self.device)

                outputs = self.model(**inputs,
                                    output_hidden_states=True,
                                    compress_layer=self.compress_layer,
                                    compress_ratio=self.compress_ratio,
                                    query_lengths=query_lengths,
                                    prompt_lengths=prompt_lengths,
                                    cutoff_layers=self.cutoff_layers)
                if self.layer_wise:
                    scores = []
                    for i in range(len(outputs.logits)):
                        logits = last_logit_pool(outputs.logits[i], outputs.attention_masks[i])
                        scores.append(logits.cpu().float().tolist())
                    if len(all_scores) == 0:
                        for i in range(len(scores)):
                            all_scores.append([])
                    for i in range(len(scores)):
                        all_scores[i].extend(scores[i])
                else:
                    logits = outputs.logits
                    # print(logits, query_lengths)
                    # sys.exit()
                    scores = last_logit_pool(logits, inputs['attention_mask'])
                    scores = scores[:, self.yes_loc].cpu().float().tolist()
                    all_scores.extend(scores)
        else:
            prompt = "Predict whether passage B contains an answer to query A."
            prompt_inputs = self.tokenizer(prompt,
                                                return_tensors=None,
                                                add_special_tokens=False)['input_ids']
            sep = "\n"
            sep_inputs = self.tokenizer(sep,
                                             return_tensors=None,
                                             add_special_tokens=False)['input_ids']
            encode_max_length = max_length + len(sep_inputs) + len(prompt_inputs)
            all_scores = []
            for batch_start in trange(0, len(sentences_sorted), batch_size):
                batch_sentences = sentences_sorted[batch_start:batch_start + batch_size]
                batch_sentences = [(f'A: {q}', f'B: {p}') for q,p in batch_sentences]
                queries = [s[0] for s in batch_sentences]
                passages = [s[1] for s in batch_sentences]
                queries_inputs = self.tokenizer(queries,
                                                return_tensors=None,
                                                add_special_tokens=False,
                                                max_length=32,
                                                truncation=True)
                passages_inputs = self.tokenizer(passages,
                                                 return_tensors=None,
                                                 add_special_tokens=False,
                                                 max_length=max_length,
                                                 truncation=True)
                query_lengths = []
                prompt_lengths = []
                batch_inputs = []
                for query_inputs, passage_inputs in zip(queries_inputs['input_ids'], passages_inputs['input_ids']):
                    item = self.tokenizer.prepare_for_model(
                        [self.tokenizer.bos_token_id] + query_inputs,
                        sep_inputs + passage_inputs,
                        truncation='only_second',
                        max_length=encode_max_length,
                        padding=False,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        add_special_tokens=False
                    )
                    item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
                    item['attention_mask'] = [1] * len(item['input_ids'])
                    item.pop('token_type_ids') if 'token_type_ids' in item.keys() else None
                    if 'position_ids' in item.keys():
                        item['position_ids'] = list(range(len(item['input_ids'])))
                    batch_inputs.append(item)
                    query_lengths.append(len([self.tokenizer.bos_token_id] + query_inputs + sep_inputs))
                    prompt_lengths.append(len(sep_inputs + prompt_inputs))

                collater_instance = collater(self.tokenizer, max_length)
                batch_inputs = collater_instance(
                    [
                        [{'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']} for item in
                     batch_inputs],
                        query_lengths,
                        prompt_lengths
                    ])[0]

                batch_inputs = {key: val.to(self.device) for key, val in batch_inputs.items()}

                outputs = self.model(**batch_inputs,
                                    output_hidden_states=True,
                                    compress_layer=self.compress_layer,
                                    compress_ratio=self.compress_ratio,
                                    query_lengths=query_lengths,
                                    prompt_lengths=prompt_lengths,
                                    cutoff_layers=self.cutoff_layers)
                if self.layer_wise:
                    scores = []
                    for i in range(len(outputs.logits)):
                        logits = last_logit_pool(outputs.logits[i], outputs.attention_masks[i])
                        scores.append(logits.cpu().float().tolist())
                    if len(all_scores) == 0:
                        for i in range(len(scores)):
                            all_scores.append([])
                    for i in range(len(scores)):
                        all_scores[i].extend(scores[i])
                else:
                    logits = outputs.logits
                    # print(logits, query_lengths)
                    # sys.exit()
                    scores = last_logit_pool(logits, batch_inputs['attention_mask'])
                    scores = scores[:, self.yes_loc].cpu().float().tolist()
                    all_scores.extend(scores)

        if self.layer_wise:
            for i in range(len(all_scores)):
                all_scores[i] = [all_scores[i][idx] for idx in np.argsort(length_sorted_idx)]
        else:
            all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        # if normalize:
        #     all_scores = [sigmoid(score) for score in all_scores]
        #
        # if len(all_scores) == 1:
        #     return all_scores[0]

        return all_scores


    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings