import json
import numpy as np
import torch
import transformers
from transformers import (BertTokenizerFast)
import logging
import os
import sys

logger = transformers.utils.logging.get_logger(__name__)

def calclulate_f1(statics_dict, prefix=""):
    """
    Calculate the prec, recall and f1-score for the given state dict.
    The state dict contains predict_num, golden_num, correct_num.
    Reutrn a dict in the form as "prefx-recall": 0.99.
    """
    prec, recall, f1 = 0, 0, 0
    if statics_dict["c"] != 0:
        prec = float(statics_dict["c"] / statics_dict["p"])
        recall = float(statics_dict["c"] / statics_dict["g"])
        f1 = float(prec * recall) / float(prec + recall) * 2
    return {prefix+"-prec": prec, prefix+"-recall": recall, prefix+"-f1": f1}

def combine_dict(dicta, dictb):
    new_dict = {}
    for e in dicta:
        if e not in new_dict:
            new_dict[e] = []
        new_dict[e] += dicta[e]
    for e in dictb:
        if e not in new_dict:
            new_dict[e] = []
        new_dict[e] += dictb[e]
        new_dict[e] = list(set(new_dict[e]))
    return new_dict


def unirel_extractor(tokenizer,
                   dataset,
                   predictions,
                   path,
                   ):
    """
    Extractor triples from the modeled Attention matrix
    """
    # Minus the [cls] and [sep] 
    token_len = dataset.max_length - 2

    def get_e2r(e2r_pred):
        """
        Extract entity-relation (subject-relation) and entity-entity interactions from given Attention Matrix.
        Only Extract the upper-right triangle, so should input transpose of the original
        Attention Matrix to extract relation-entity (relation-object) interactions.
        """
        e2r = {}
        tok_tok = set()
        e_va = np.where(e2r_pred == 1)
        for h, r in zip(e_va[0], e_va[1]):
            h = int(h)
            r = int(r)            
            if h == 0 or r == 0 or r == token_len+1 or h > token_len:
                continue
            # Entity-Entity
            if r < token_len+1:
                tok_tok.add((h,r))
            # Entity-Relation
            else:
                r = int(r-token_len-2)
                if h not in e2r:
                    e2r[h] = []
                e2r[h].append(r)
        return e2r, tok_tok


    state_dict = {"p": 0, "c": 0, "g": 0}
    e2e_state_dict = {"p": 0, "c": 0, "g": 0}
    e2e_tail_state_dict = {"p": 0, "c": 0, "g": 0}
    e2e_plain_state_dict = {"p": 0, "c": 0, "g": 0}
    h2r_state_dict = {"p": 0, "c": 0, "g": 0}
    t2r_state_dict = {"p": 0, "c": 0, "g": 0}
    idx2pred = dataset.data_processor.idx2pred
    extract_data = []
    path = os.path.join(path, dataset.mode + '_predict_sard.json')
    tail_labels = predictions.label_ids
    tail_preds = predictions.predictions
    # NOTE: This is only for test!
    # tail_preds, head_preds, span_preds = predictions.label_ids
    curr_data_idx = 0
    for tail_pred, tail_label in zip(tail_preds, tail_labels):
        input_ids = dataset[curr_data_idx]["input_ids"]
        text = dataset.texts[curr_data_idx]
        spo_list = dataset.spo_lists[curr_data_idx]
        spo_span_list = dataset.spo_span_lists[curr_data_idx]
        gold_spo_text = set()
        gold_ee_list = set()
        gold_plain_ee_list = set()
        gold_sr_list = {}
        gold_or_list = {}
        gold_er_list = {}
        gold_entity_list = set()
        gold_ee_tail_list = set()
        # Extract golden triples with same tokenizer 
        for spo in spo_span_list:
            rel_str = dataset.data_processor.idx2pred[spo[1]]
            left_str = tokenizer.decode(input_ids[spo[0][1]])
            right_str = tokenizer.decode(input_ids[spo[2][1]])
            gold_ee_list.add((spo[0][1], spo[2][1]))
            gold_plain_ee_list.add((spo[0][1], spo[2][1]))
            gold_plain_ee_list.add((spo[2][1], spo[0][1]))
            gold_ee_tail_list.add((spo[0][1], spo[2][1]))
            # gold_ee_list.add((spo[2][1], spo[0][1]))
            gold_entity_list.add(spo[0][1])
            gold_entity_list.add(spo[2][1])            
            if spo[0][1] not in gold_sr_list:
                gold_sr_list[spo[0][1]] = set()
            gold_sr_list[spo[0][1]].add(spo[1])
            if spo[2][1] not in gold_or_list:
                gold_or_list[spo[2][1]] = set()
            gold_or_list[spo[2][1]].add(spo[1])
            if spo[0][1] not in gold_er_list:
                gold_er_list[spo[0][1]] = set()
            if spo[2][1] not in gold_er_list:
                gold_er_list[spo[2][1]] = set()
            gold_er_list[spo[0][1]].add(spo[1])
            gold_er_list[spo[2][1]].add(spo[1])

            gold_spo_text.add((left_str, rel_str, right_str))
        
        curr_data_idx += 1
        pred_spo_text = set()
        pred_spo_span_list = set()
        pred_ee_list = set()
        pred_ee_tail_list = set()
        pred_plain_ee_list = set()
        # h2r: subject(head) - relation, e2e: entity - entity
        e_h2r, e2e = get_e2r(tail_pred)
        # t2r: object(tail) - relation
        e_t2r, t_e2e = get_e2r(tail_pred.T)
        # For each possible entity pair
        for left, right in e2e:
            # Consider both directions
            for l,r in [(left, right), (right, left)]:
                pred_plain_ee_list.add((l,r))
                # Find mutual relations
                if l in e_h2r and r in e_t2r:
                    common_rels = set(e_h2r[l]) & set(e_t2r[r])
                    for rel in common_rels:
                        pred_ee_list.add((l, r))
                        pred_spo_span_list.add((
                            l, rel, r
                        ))
                        pred_spo_text.add((
                            tokenizer.decode(input_ids[l]),
                            idx2pred[rel],
                            tokenizer.decode(input_ids[r])
                        ))
        state_dict["p"] += len(pred_spo_text)
        state_dict["g"] += len(gold_spo_text)
        state_dict["c"] += len(pred_spo_text & gold_spo_text)
        
        e2e_state_dict["p"] += len(pred_ee_list)
        e2e_state_dict["g"] += len(gold_ee_list)
        e2e_state_dict["c"] += len(set(pred_ee_list) & set(gold_ee_list))

        e2e_plain_state_dict["p"] += len(pred_plain_ee_list)
        e2e_plain_state_dict["g"] += len(gold_plain_ee_list)
        e2e_plain_state_dict["c"] += len(set(pred_plain_ee_list) & set(gold_plain_ee_list))

        e2e_tail_state_dict["p"] += len(pred_ee_tail_list)
        e2e_tail_state_dict["g"] += len(gold_ee_tail_list)
        e2e_tail_state_dict["c"] += len(set(pred_ee_tail_list) & set(gold_ee_tail_list))
        for e in e_h2r:
            if e in gold_sr_list:
                h2r_state_dict["p"] += len(e_h2r[e])
                h2r_state_dict["g"] += len(gold_sr_list[e])
                h2r_state_dict["c"] += len(set(e_h2r[e]) & set(gold_sr_list[e]))
        for e in e_t2r:
            if e in gold_or_list:
                t2r_state_dict["p"] += len(e_t2r[e])
                t2r_state_dict["g"] += len(gold_or_list[e])
                t2r_state_dict["c"] += len(set(e_t2r[e]) & set(gold_or_list[e]))
        

        extract_data.append({
            "text": text,
            "gold_spo_list": list(spo_list),
            "pred_spo_list": list(pred_spo_text),
            "gold_spo_tail_list": list(gold_spo_text),
            "pred_spo_span_list": list(pred_spo_span_list),
            "gold_spo_span_list": list(spo_span_list),
        })
    # print(calclulate_f1(state_dict))
    all_metirc_results = calclulate_f1(state_dict, 'all')
    print(f"\nall:  {state_dict} \n {calclulate_f1(state_dict, 'all')}")
    print(f"\ne2e:  {e2e_state_dict} \n {calclulate_f1(e2e_state_dict, 'e2e')}")
    print(f"\nh2r:  {h2r_state_dict} \n {calclulate_f1(h2r_state_dict, 'h2r')}")
    print(f"\nt2r:  {t2r_state_dict} \n {calclulate_f1(t2r_state_dict, 't2r')}")
    print(f"\ne2e_tail:  {e2e_tail_state_dict} \n {calclulate_f1(e2e_tail_state_dict, 'e2e_tail')}")
    print(f"\ne2e without rel:  {e2e_plain_state_dict} \n {calclulate_f1(e2e_plain_state_dict, 'e2e_plain')}")
    logger.info(f"\nall:  {calclulate_f1(state_dict, 'all')}")
    logger.info(f"\ne2e:  {calclulate_f1(e2e_state_dict, 'e2e')}")
    logger.info(f"\nh2r:  {calclulate_f1(h2r_state_dict, 'h2r')}")
    logger.info(f"\nt2r:  {calclulate_f1(t2r_state_dict, 't2r')}")
    logger.info(f"\ne2e_tail:  {calclulate_f1(e2e_tail_state_dict, 'e2e_tail')}")
    logger.info(f"\ne2e without rel:  {e2e_plain_state_dict} \n {calclulate_f1(e2e_plain_state_dict, 'e2e_plain')}")
    with open(path, "w") as wp:
        json.dump(extract_data, wp, indent=2, ensure_ascii=False)
    return all_metirc_results["all-prec"], all_metirc_results["all-recall"], all_metirc_results["all-f1"]

