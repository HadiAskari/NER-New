import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup
from model_nll import NLLModel
from utils_nll import set_seed, collate_fn, read_conll, get_labels
from torch.cuda.amp import autocast, GradScaler
import seqeval.metrics
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import datetime
from datetime import datetime
import wandb
import random
import time
import psutil

CACHE_DIR = "cache"
BEST_DEV_F1 = 0.0

def train(args, model, train_features, benchmarks):

    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    total_steps = int(len(train_dataloader) * args.epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler()

    num_steps = 0
    start_time = time.time()
    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        epoch_start_time = time.time()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            if num_steps < args.alpha_warmup_ratio * total_steps:
                args.alpha_t = 0.0
            else:
                args.alpha_t = args.alpha
            batch = {key: value.to(args.device) for key, value in batch.items()}
            with autocast():
                outputs = model(**batch)
            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                wandb.log({'loss': loss.item()}, step=num_steps)
            if step == len(train_dataloader) - 1:
                # loop over dev and test
                for tag, features, sents in benchmarks:
                    results = evaluate(args, model, features, sents, tag=tag)
                    wandb.log(results, step=num_steps)
        torch.save(model.state_dict(), os.path.join("model", "model_epoch_"+ str(epoch) +".pt"))
        #model.models[0].save_pretrained()
        elapsed_time = time.time() - epoch_start_time
        peak_memory = torch.cuda.memory_allocated() / 1024 / 1024
        cache_memory = torch.cuda.memory_cached() / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        wandb.log({'epoch_wall_time': elapsed_time, 'epoch_peak_memory_MB': peak_memory, 'epoch_cached_memory_MB': cache_memory, 'epoch_cpu_percent': cpu_percent}, step=num_steps)


def convert_spans(arr):
    """List of ner arrays to be converted to spans"""
    spans = []
    for sublist in arr:
        curr = []
        for word in sublist:
            if word.startswith("B-"):
                curr.append("B-NE")
            elif word.startswith("I-"):
                curr.append("I-NE")
            else:
                curr.append("O")
        spans.append(curr)
    return spans

def evaluate(args, model, features, sentences, tag="dev"):
    global BEST_DEV_F1
    label_list = get_labels(label_semantics=False, path=args.label_file)
    ID_TO_LABEL = {i: label for i, label in enumerate(label_list)}

    dataloader = DataLoader(features, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, keys = [], []

    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        key = batch['labels'].cpu().numpy().flatten().tolist()
        batch['labels'] = None
        with torch.no_grad():
            logits = model(**batch)[0]
            pred = np.argmax(logits.cpu().numpy(), axis=-1).tolist()

        # unpack label
        pred_arr, key_arr = [], []
        for i in range(len(key)):
            if key[i] == -1:  # skip -1 key
                if pred_arr:
                    preds.append(pred_arr)
                    keys.append(key_arr)
                    pred_arr = []
                    key_arr = []
            else:
                key_arr.append(ID_TO_LABEL[key[i]])
                pred_arr.append(ID_TO_LABEL[pred[i]])

    model.zero_grad()

    micro_precision = precision_score(keys, preds, average='micro')
    micro_recall = recall_score(keys, preds, average='micro')
    micro_f1 = f1_score(keys, preds, average='micro')

    output = {
        tag + "_precision": micro_precision,
        tag + "_recall": micro_recall,
        tag + "_f1": micro_f1,
    }

    # write seqeval results to file
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # date_hour = datetime.now().strftime("%Y-%m-%d-%H")
    date_hour = datetime.now().strftime("%Y-%m-%d")
    model_name = "bertoverflow" if "word_piece_ner" in args.model_name_or_path else args.model_name_or_path
    fname = f"{date_hour}_self_{tag}_model{model_name}_nmod{args.n_model}_alpha{args.alpha}_warmup{args.alpha_warmup_ratio}_data{args.data_dir}"
    outfile = os.path.join(args.out_dir, fname)

    if BEST_DEV_F1 < micro_f1 or tag == "test":
        with open(outfile, "w") as writer:
            true_spans = convert_spans(keys)
            pred_spans = convert_spans(preds)
            writer.write("Overall Span -> \n")
            writer.write(f"Precision score: {precision_score(true_spans, pred_spans)}\n")
            writer.write(f"Recall score: {recall_score(true_spans, pred_spans)}\n")
            writer.write(f"F1 score: {f1_score(true_spans, pred_spans)}\n")
            writer.write(classification_report(true_spans, pred_spans, digits=5))

            writer.write("\n\nOverall -> \n")
            writer.write(f"Precision score: {precision_score(keys, preds)}\n")
            writer.write(f"Recall score: {recall_score(keys, preds)}\n")
            writer.write(f"F1 score: {f1_score(keys, preds)}\n")
            writer.write(classification_report(keys, preds, digits=5))

        # update if F1 dev
        if BEST_DEV_F1 < micro_f1 and tag=="dev": BEST_DEV_F1 = micro_f1

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="software", type=str)
    parser.add_argument("--out_dir", default="nll", type=str)
    parser.add_argument("--label_file", default="labels.json", type=str)
    parser.add_argument("--model_name_or_path", default="bert-large-cased", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_model", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)

    # Tai: Add label semantics
    parser.add_argument("--label_semantics", type=bool, default=False)
    args = parser.parse_args()

    wandb.login(key="b9f22b44e0a2f20841eaf31a8473648709d3cade")
    wandb.init(
        project="NER4SE",
        entity="meguru",
        config=args,
        dir=os.path.join(CACHE_DIR, "wandb"),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.num_class = len(get_labels(label_semantics=False, path=args.label_file))
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding='max_length',
        truncation=True,
        cache_dir = CACHE_DIR
    )
    model = NLLModel(args)

    base = os.path.join(os.path.join(CACHE_DIR, "ner"), args.data_dir)
    train_file = os.path.join(base, "train.txt")
    dev_file = os.path.join(base, "dev.txt")
    test_file = os.path.join(base, "test.txt")

    train_features, train_sentences = read_conll(args, train_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features, test_sentences = read_conll(args, test_file, tokenizer, max_seq_length=args.max_seq_length)
    # S-NER does not have dev
    if not args.data_dir.startswith("s-ner"):
        dev_features, dev_sentences = read_conll(args, dev_file, tokenizer, max_seq_length=args.max_seq_length)
        benchmarks = (
            ("dev", dev_features, dev_sentences),
            ("test", test_features, test_sentences),
        )
    else:
        benchmarks = (("test", test_features, test_sentences),)

    train(args, model, train_features, benchmarks)


if __name__ == "__main__":
    main()
