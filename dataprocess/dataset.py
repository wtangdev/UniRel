import numpy as np
import torch
from torch.utils.data import Dataset


class UniRelDataset(Dataset):
    def __init__(self,
                 samples,
                 data_processor,
                 tokenizer,
                 mode='train',
                 max_length=102,
                 ignore_label=-100,
                 model_type='bert',
                 no_entity_label='O',
                 ngram_dict=None,
                 enhanced=False,
                 predict=False,
                 eval_type="eval"):
        super(UniRelDataset, self).__init__()

        self.max_length = max_length
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.ignore_label = ignore_label
        self.mode = mode
        self.no_entity_label = no_entity_label
        self.ngram_dict = ngram_dict
        self.model_type = model_type
        self.enhanced = enhanced
        self.predict = predict
        self.eval_type = eval_type
        self.num_rels = data_processor.num_rels

        self.texts = samples['text']
        self.spo_lists = samples["spo_list"]
        self.spo_span_lists = samples["spo_span_list"]
        self.tail_labels = samples["tail_label"]

        self.max_label_len = data_processor.max_label_len
        self.pred2text = data_processor.pred2text


        self.pred_str = data_processor.pred_str
        self.pred_inputs = tokenizer.encode_plus(self.pred_str,
                                                 add_special_tokens=False)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(text,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True)
        token_len = self.max_length
        num_rels = self.num_rels

        tail_label = torch.tensor(self.tail_labels[idx], dtype=torch.long)
        sep_idx = inputs["input_ids"].index(self.tokenizer.sep_token_id)
        input_ids = inputs["input_ids"] + self.pred_inputs["input_ids"]
        input_ids[sep_idx] = self.tokenizer.pad_token_id

        attention_mask = inputs["attention_mask"] + [1] * num_rels
        attention_mask[sep_idx] = 0
        token_type_ids = inputs["token_type_ids"] + [1] * num_rels


        return {
            "input_ids":
            torch.tensor(np.array(input_ids, dtype=np.int64),
                         dtype=torch.long),
            "attention_mask":
            torch.tensor(np.array(attention_mask, dtype=np.int64),
                         dtype=torch.long),
            "token_type_ids":
            torch.tensor(np.array(token_type_ids, dtype=np.int64),
                         dtype=torch.long),
            "token_len_batch":
            torch.tensor(token_len, dtype=torch.long),
            "tail_label": tail_label,
        }

    def __len__(self):
        return len(self.texts)



class UniRelSpanDataset(Dataset):
    def __init__(self,
                 samples,
                 data_processor,
                 tokenizer,
                 mode='train',
                 max_length=102,
                 ignore_label=-100,
                 model_type='bert',
                 no_entity_label='O',
                 ngram_dict=None,
                 enhanced=False,
                 predict=False,
                 eval_type="eval"):
        super(UniRelSpanDataset, self).__init__()

        self.max_length = max_length
        self.data_processor = data_processor
        self.tokenizer = tokenizer
        self.ignore_label = ignore_label
        self.mode = mode
        self.no_entity_label = no_entity_label
        self.ngram_dict = ngram_dict
        self.model_type = model_type
        self.enhanced = enhanced
        self.predict = predict
        self.eval_type = eval_type
        self.num_rels = data_processor.num_rels

        self.texts = samples['text']
        self.spo_lists = samples["spo_list"]
        self.spo_span_lists = samples["spo_span_list"]
        self.head_labels = samples["head_label"]
        self.tail_labels = samples["tail_label"]
        self.span_labels = samples["span_label"]

        self.max_label_len = data_processor.max_label_len
        self.pred2text = data_processor.pred2text


        self.pred_str = data_processor.pred_str
        self.pred_inputs = tokenizer.encode_plus(self.pred_str,
                                                 add_special_tokens=False)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(text,
                                            max_length=self.max_length,
                                            padding='max_length',
                                            truncation=True)
        token_len = self.max_length
        num_rels = self.num_rels

        head_label = torch.tensor(self.head_labels[idx], dtype=torch.long)
        tail_label = torch.tensor(self.tail_labels[idx], dtype=torch.long)
        span_label = torch.tensor(self.span_labels[idx], dtype=torch.long)
        sep_idx = inputs["input_ids"].index(self.tokenizer.sep_token_id)
        input_ids = inputs["input_ids"] + self.pred_inputs["input_ids"]
        input_ids[sep_idx] = self.tokenizer.pad_token_id

        attention_mask = inputs["attention_mask"] + [1] * num_rels
        attention_mask[sep_idx] = 0
        token_type_ids = inputs["token_type_ids"] + [1] * num_rels


        return {
            "input_ids":
            torch.tensor(np.array(input_ids, dtype=np.int64),
                         dtype=torch.long),
            "attention_mask":
            torch.tensor(np.array(attention_mask, dtype=np.int64),
                         dtype=torch.long),
            "token_type_ids":
            torch.tensor(np.array(token_type_ids, dtype=np.int64),
                         dtype=torch.long),
            "token_len_batch":
            torch.tensor(token_len, dtype=torch.long),
            "head_label": head_label,
            "tail_label": tail_label,
            "span_label": span_label,
        }

    def __len__(self):
        return len(self.texts)
