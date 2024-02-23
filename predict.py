import os
import numpy as np
import torch

from transformers import BertTokenizerFast
import dataprocess.rel2text
from model.model_transformers import UniRelModel
from dataprocess.data_extractor import *
from dataprocess.data_metric import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class UniRel:
    def __init__(self, model_path, max_length=128, dataset_name="nyt") -> None:
        self.model = UniRelModel.from_pretrained(model_path)
        # The BERT-Base-Uncased vocabulary has a size of 30522 with only 994 unused slots (in com- parison, BERT-Base-Cased has only 101 unused slots).
        added_token = [f"[unused{i}]" for i in range(1, 17)]
        # load tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-cased", additional_special_tokens=added_token, do_basic_tokenize=False)
        self.max_length = max_length
        self.max_length = max_length
        self._get_pred_str(dataset_name)


        
    
    def _get_pred_str(self, dataset_name):
        self.pred2text = None
        if dataset_name == "nyt":
            self.pred2text=dataprocess.rel2text.nyt_rel2text
        elif dataset_name == "nyt_star":
            self.pred2text=dataprocess.rel2text.nyt_rel2text
        elif dataset_name == "webnlg":
            self.pred2text=dataprocess.rel2text.webnlg_rel2text
            cnt = 1
            exist_value=[]
            # Some hard to convert relation directly use [unused]
            for k in self.pred2text:
                v = self.pred2text[k]
                if isinstance(v, int):
                    self.pred2text[k] = f"[unused{cnt}]" 
                    cnt += 1
                    continue
                ids = self.tokenizer(v)
                if len(ids["input_ids"]) != 3:
                    print(k, "   ", v)
                if v in exist_value:
                    print("exist", k, "  ", v)
                else:
                    exist_value.append(v)
        elif dataset_name == "webnlg_star":
            self.pred2text = dataprocess.rel2text.webnlg_rel2text
            cnt = 1
            exist_value=[]
            for k in self.pred2text:
                v = self.pred2text[k]
                if isinstance(v, int):
                    self.pred2text[k] = f"[unused{cnt}]" 
                    cnt += 1
                    continue
                ids = self.tokenizer(v)
                if len(ids["input_ids"]) != 3:
                    print(k, "   ", v)
                if v in exist_value:
                    print("exist", k, "  ", v)
                else:
                    exist_value.append(v)
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
        else:
            print("dataset name error")
            exit(0)
        self.pred_str = ""
        self.max_label_len = 1
        self.pred2idx = {}
        idx = 0
        for k in self.pred2text:
            self.pred2idx[k] = idx
            self.pred_str += self.pred2text[k] + " "
            idx += 1
        self.num_rels = len(self.pred2text.keys())
        print(f'relations: \n{self.pred2text.keys()}')
        print(f'num_rels: {self.num_rels}')
        self.idx2pred = {value: key for key, value in self.pred2idx.items()}
        self.pred_str = self.pred_str[:-1]
        # self.pred_str: all relations in natural language, not token id: "rel0 rel1 rel2 ... reln"
        # print("pred_str: \n", self.pred_str)
        self.pred_inputs = self.tokenizer.encode_plus(self.pred_str,
                                                 add_special_tokens=False)
        # {pred_inputs: 'input_ids':[], 'token_type_ids':[], 'attention_mask':[]}
    
    def _data_process(self, text):
        # text could be a list of sentences or a single sentence
        if isinstance(text, str):
            text = [text]
        # [CLS] and [SEP] are added to the input_ids
        inputs = self.tokenizer.batch_encode_plus(text, max_length=self.max_length, padding="max_length", truncation=True)
        batched_input_ids = []
        batched_attention_mask = []
        batched_token_type_ids = []
        for b_input_ids, b_attention_mask, b_token_type_ids in zip(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]):
            # self.pred_inputs["input_ids"]: the token ids of all relations
            # input_ids: [CLS] + text + [SEP] + [rel0] + [rel1] + ... + [reln]
            input_ids = b_input_ids + self.pred_inputs["input_ids"]
            # where is the [SEP] token in the input
            sep_idx = b_input_ids.index(self.tokenizer.sep_token_id)
            # replace the [SEP] token with [PAD] token
            input_ids[sep_idx] = self.tokenizer.pad_token_id
            attention_mask = b_attention_mask + [1]*self.num_rels
            # Pre-[SEP] token (now [PAD]) is masked as 0
            attention_mask[sep_idx] = 0
            token_type_ids = b_token_type_ids + [1]*self.num_rels
            batched_input_ids.append(input_ids)
            batched_attention_mask.append(attention_mask)
            batched_token_type_ids.append(token_type_ids)
        return batched_input_ids, batched_attention_mask, batched_token_type_ids
    

    def _get_e2r(self, e2r_pred):
        """
        Extract entity-relation (subject-relation) and entity-entity interactions from given Attention Matrix.
        Only Extract the upper-right triangle, so should input transpose of the original
        Attention Matrix to extract relation-entity (relation-object) interactions.
        """
        # token_len is the max length for text, not inlude the [cls] and [sep] and relations
        token_len = self.max_length-2
        e2r = {}
        tok_tok = set()
        # e_va: (arr0, arr1), arr1 is the row, arr0 is the column
        e_va = np.where(e2r_pred == 1)
        for h, r in zip(e_va[0], e_va[1]):
            h = int(h)
            r = int(r)
            # if h or r is at the pos of [cls] or [pad], then continue
            if h == 0 or r == 0 or r == token_len+1 or h > token_len:
                continue
            # Entity-Entity
            if r < token_len+1:
                tok_tok.add((h,r))
            # Entity-Relation
            else:
                # e2r: {entity_pos: [rel_num1, rel_num2, ...]}
                r = int(r-token_len-2)
                if h not in e2r:
                    e2r[h] = []
                e2r[h].append(r)
        return e2r, tok_tok
    
    def _get_span_att(self, span_pred):
        token_len = self.max_length-2
        span_va = np.where(span_pred == 1)
        t2_span = dict()
        h2_span = dict()
        for s, e in zip(span_va[0], span_va[1]):
            # if s > token_len or e > token_len or s == 0 or e == 0:
            if s > token_len or e > token_len:
                continue
            if e < s:
                continue
            if e not in t2_span:
                t2_span[e] = []
            if s not in h2_span:
                h2_span[s] = []
            s = int(s)
            e = int(e)
            t2_span[e].append((s,e))
            h2_span[s].append((s,e))
        return  h2_span, t2_span

    def _extractor(self, outputs, input_ids_list):
        # outputs: {loss, head_preds, tail_preds, span_preds} with shape (batch_size, max_length, max_length)
        # input_ids_list: (batch_size, max_length) max length = |text| + |relations| + 2(cls, pad(pre-sep))
        preds_list = []
        for head_pred, tail_pred, span_pred, input_ids in zip(outputs["head_preds"], outputs["tail_preds"], outputs["span_preds"], input_ids_list):
            pred_spo_text = set()
            # self._get_e2r() will get:
            # {entity_pos: [rel_num1, rel_num2, ...]}
            # and set((h_entity_pos, r_entity_pos), ...)
            s_h2r, s2s = self._get_e2r(head_pred)
            # .T to get the object(tail)
            s_t2r, _ = self._get_e2r(head_pred.T)
            e_h2r, e2e = self._get_e2r(tail_pred)
            e_t2r, _ = self._get_e2r(tail_pred.T)
            # span_pred show the start and end position of each entity
            # {start_token_pos: [(start_token_pos, end_token_pos), ...]}
            # {end_token_pos: [(start_token_pos, end_token_pos), ...]}
            start2span, end2span = self._get_span_att(span_pred)
            # print(f'start2span, end2span: {start2span}\n{end2span}')
            for l, r in e2e:
                # l is already have a relation with r
                # here is to check if the l or r are involved with any relations
                if l not in e_h2r or r not in e_t2r:
                    continue
                if l not in end2span or r not in end2span:
                    continue
                # l_spans: [(start_token_pos, end_token_pos), ...], r_spans the same
                l_spans, r_spans = end2span[l], end2span[r]
                for l_span in l_spans:
                    for r_span in r_spans:
                        l_s, r_s = l_span[0], r_span[0]
                        # the end pos are related but if the start pos are not related, then continue
                        if (l_s, r_s) not in s2s:
                            continue
                        # if the start pos are not related to any relations, then continue
                        if l_s not in s_h2r or r_s not in s_t2r:
                            continue
                        common_rels = set(s_h2r[l_s]) & set(s_t2r[r_s]) & set(e_h2r[l]) & set(e_t2r[r])
                        # l_span_new = (l_span[0]+1, l_span[1])
                        # r_span_new = (r_span[0]+1, r_span[1])
                        l_span_new = (l_span[0], l_span[1])
                        r_span_new = (r_span[0], r_span[1])
                        for rel in common_rels:
                            pred_spo_text.add((
                                self.tokenizer.decode(input_ids[l_span_new[0]:l_span_new[1]+1]),
                                self.idx2pred[rel],
                                self.tokenizer.decode(input_ids[r_span_new[0]:r_span_new[1]+1])
                            ))
                        # print(f'pred_spo_text: {pred_spo_text}')
            preds_list.append(list(pred_spo_text))
        return preds_list

    def predict(self, text):
        input_ids, attention_mask, token_type_ids = self._data_process(text)
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            token_type_ids = torch.tensor(token_type_ids)
        else:
            # only one sentence, add a batch dimension
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)
            token_type_ids = torch.tensor(token_type_ids).unsqueeze(0)
        self.model.eval()
        # print(self.model)
        # print the head number of the attention
        # print(self.model.config)
        with torch.no_grad():
            # outputs format: {loss, head_preds, tail_preds, span_preds}
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            # print(f"outputs.span_preds.shape: {outputs.span_preds.shape}")
            results = self._extractor(outputs, input_ids)
        return results  


if __name__ == "__main__":
    model_path = "/home/tian/Projects/UniRel/model/nyt-checkpoint-final"
    unirel = UniRel(model_path, dataset_name="nyt")
    
    # print(unirel.predict("In perhaps the most ambitious Mekong cruise attempt, Impulse Tourism, an operator based in Chiang Mai, Thailand, is organizing an expedition starting in November in Jinghong, a small city in the Yunnan province in China."))



    # print(unirel.predict("Adisham Hall in Sri Lanka was constructed between 1927 and 1931 at St Benedicts Monastery , Adisham , Haputhale , Sri Lanka in the Tudor and Jacobean style of architecture"))

    t = "I don't think the Yunnan government or any other organization has dominion over the jungles of Xishuangbanna."
    # less sensitive to negative statements.

    print(unirel.predict(t))
    # print(unirel._data_process(t))
    # print(unirel.predict([
    #     "Anson was born in 1979 in Hong Kong.",
    #     "In perhaps the most ambitious Mekong cruise attempt, Impulse Tourism, an operator based in Chiang Mai, Thailand, is organizing an expedition starting in November in Jinghong, a small city in the Yunnan province in China.",
    #     "Adisham Hall in Sri Lanka was constructed between 1927 and 1931 at St Benedicts Monastery , Adisham , Haputhale , Sri Lanka in the Tudor and Jacobean style of architecture"
    # ]))
    # print(unirel.predict("These are tough changes , and some of them will be quite controversial among our colleagues here , '' Senator Joseph I. Lieberman , Democrat of Connecticut , said Thursday as he endorsed a plan developed by Senator John McCain , Republican of Arizona , after his hearings into Mr. Abramoff 's bilking of Indian tribes through a lobbying operation ."))

    print("end")
    # print the sum number of trainable parameters
    # print(sum(p.numel() for p in unirel.model.parameters() if p.requires_grad))
        