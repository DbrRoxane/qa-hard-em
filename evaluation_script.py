import sys
import ujson as json
import re
import string
from collections import Counter
import pickle
from IPython import embed
import nltk
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as ms

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def rougel_score(prediction, ground_truth):
    rouge = Rouge()

    #normalized_prediction = normalize_answer(prediction)
    #normalized_ground_truth = normalize_answer(ground_truth)
    if len(prediction) > 0 and prediction != ".":
        score =  max([s["rouge-l"]["f"] for s in rouge.get_scores(prediction, ground_truth)])
    else:
        score = 0
    return score

def bleu_score(prediction, ground_truth, n=1):
    #normalized_prediction = normaliz_answer(prediction)
    #normalized_ground_truth = normalize_answer(ground_truth)

    pred_tokenized = nltk.word_tokenize(prediction)
    gt_tokenized = nltk.word_tokenize(ground_truth)
    chencherry = SmoothingFunction()

    if n ==1:
        score = sentence_bleu([gt_tokenized], pred_tokenized, weights=(1,0,0,0))
    elif len(pred_tokenized) != 0:
        score = sentence_bleu([gt_tokenized], pred_tokenized, smoothing_function=chencherry.method1 ) #bleu-4 automatically
    else:
        print("weird")
        score = 0
    return score

def meteor_score(prediction, ground_truth):
    #normalized_prediction = normalize_answer(prediction)
    #normalized_ground_truth = normalize_answer(ground_truth)
    return ms(ground_truth, prediction)

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    rougel = rougel_score(prediction, gold)
    bleu1 = bleu_score(prediction, gold, 1)
    bleu4 = bleu_score(prediction, gold, 4)
    meteor = meteor_score(prediction, gold)

    metrics['em'] += em
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    metrics['rougel'] += rougel
    metrics['bleu1'] += bleu1
    metrics['bleu4'] += bleu4
    metrics['meteor'] += meteor
    return em, prec, recall, rougel, bleu1, bleu4, meteor

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'rougel': 0, 'bleu1': 0, 'bleu4': 0, 'meteor': 0,
        'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    for dp in gold:
        cur_id = dp['_id']
        em, prec, recall, rougel, bleu1, bleu4, meteor = update_answer(
            metrics, prediction['answer'][cur_id], dp['answer'])

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    print(metrics)

def analyze(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)
    metrics = {'rougel': 0, 'bleu1': 0, 'bleu4': 0, 'meteor': 0,
        'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}

    for dp in gold:
        cur_id = dp['_id']

        em, prec, recall, rougel, bleu1, bleu4, meteor = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if (prec + recall == 0):
            f1 = 0
        else:
            f1 = 2 * prec * recall / (prec+recall)

        print (dp['answer'], prediction['answer'][cur_id])
        print (f1, em, rougel, bleu1, bleu4, meteor)
        a = input()


if __name__ == '__main__':
    #eval(sys.argv[1], sys.argv[2])
    analyze(sys.argv[1], sys.argv[2])









