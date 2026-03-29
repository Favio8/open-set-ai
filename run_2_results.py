import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ECGDataset_unseen_MHL_stage2
from loss_func import OpenMaxSeparation
from resnet import build_model
from utils import (
    collapse_unknown_predictions,
    compute_open_set_metrics,
    ensure_dir,
    get_dataset_defaults,
    get_device,
    parse_unseen_classes,
    pin_memory_for_device,
    save_dataframe,
    save_embedding_csv,
    set_seed,
    str2bool,
)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CPSC18')
    parser.add_argument('--model', type=str, default='resnet34')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--transform_type', type=str, default='hardneg')
    parser.add_argument('--data_path', type=str, default='./data_path/')
    parser.add_argument('--open_world_path', type=str, default='./OpenMax/')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--normalize_signal', type=str2bool, default=False)
    parser.add_argument('--whether_tsne', type=str2bool, default=False)
    parser.add_argument('--save_predictions', type=str2bool, default=True)
    return parser.parse_args()



def normalize_model_name(name: str) -> str:
    return str(name).replace('-', '_')



def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory_for_device(device),
        drop_last=False,
    )


@torch.no_grad()

def infer_test_outputs(loader, model, device):
    model.eval()
    feats, logits, labels = [], [], []
    for batch in tqdm(loader, leave=False):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        out = model(x, return_dict=True)
        feats.append(out['feat'].detach().cpu())
        logits.append(out['logits'].detach().cpu())
        labels.append(y.long().cpu())
    return {
        'feat': torch.cat(feats, dim=0) if feats else torch.empty(0),
        'logits': torch.cat(logits, dim=0) if logits else torch.empty(0),
        'labels': torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long),
    }



def load_checkpoint_and_model(args, device):
    model_tag = normalize_model_name(args.model)
    if args.model_path:
        ckpt_path = args.model_path
    else:
        ckpt_path = os.path.join('models', f"{args.dataset}_{model_tag}_{args.transform_type}_seed{args.seed}_mcmi.pth")

    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_args: Dict[str, object] = checkpoint['args']
    split_meta: Dict[str, object] = checkpoint['split_meta']
    num_known_classes = int(split_meta['num_known_classes'])
    num_open_classes = num_known_classes + max(1, int(ckpt_args.get('novel_clusters', 1)))

    model = build_model(
        ckpt_args.get('model', args.model),
        input_channels=int(ckpt_args.get('leads', 12)),
        num_classes=num_open_classes,
        known_classes=num_known_classes,
        feature_dim=int(ckpt_args.get('feature_dim', 256)),
        instance_dim=int(ckpt_args.get('instance_dim', 128)),
        class_dim=int(ckpt_args.get('class_dim', 128)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    separator = OpenMaxSeparation.from_state_dict(checkpoint['separator_state'])
    return ckpt_path, checkpoint, ckpt_args, split_meta, model, separator



def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.use_gpu, args.gpu_id)
    if device.type == 'cpu':
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    ensure_dir('./results_final')
    ensure_dir(args.open_world_path)

    ckpt_path, checkpoint, ckpt_args, split_meta, model, separator = load_checkpoint_and_model(args, device)
    defaults = get_dataset_defaults(args.dataset)
    leads = int(ckpt_args.get('leads', defaults.leads))
    hz = int(ckpt_args.get('Hz', defaults.hz))
    duration = int(ckpt_args.get('duration', defaults.duration))
    length = hz * duration
    data_dir = ckpt_args.get('data_dir', os.path.join(ckpt_args.get('data_path', args.data_path), defaults.data_dir_name))
    stage2_split_csv = checkpoint['args'].get('stage2_split_csv', os.path.join(args.open_world_path, f"{args.dataset}_{normalize_model_name(args.model)}_label_{args.transform_type}_{args.seed}.csv"))

    test_dataset = ECGDataset_unseen_MHL_stage2(
        phase='test',
        data_dir=data_dir,
        label_csv=stage2_split_csv,
        leads=leads,
        length=length,
        transform_type=args.transform_type,
        dual_view=False,
        label_key='open_label',
        normalize=args.normalize_signal,
    )
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device)
    test_outputs = infer_test_outputs(test_loader, model, device)

    unknown_open_label = int(split_meta['unknown_open_label'])
    pred_dict = separator.predict_open_labels(test_outputs['feat'], test_outputs['logits'], unknown_label=unknown_open_label)
    y_true = test_outputs['labels'].numpy().astype(int)
    y_pred = pred_dict['pred_open'].numpy().astype(int)
    y_pred = collapse_unknown_predictions(y_pred, unknown_open_label)
    novel_scores = pred_dict['novel_score'].numpy().astype(float)

    metrics, report_df = compute_open_set_metrics(
        y_true_open=y_true,
        y_pred_open=y_pred,
        unknown_label=unknown_open_label,
        novel_scores=novel_scores,
    )

    model_tag = normalize_model_name(args.model)
    base = f"{args.dataset}_{model_tag}_{args.transform_type}_seed{args.seed}"
    summary_path = os.path.join('./results_final', f"{base}.csv")
    pred_path = os.path.join('./results_final', f"{base}_predictions.csv")
    report_path = os.path.join('./results_final', f"{base}_report.csv")

    summary_df = pd.DataFrame([
        {
            'dataset': args.dataset,
            'model': ckpt_args.get('model', args.model),
            'checkpoint': ckpt_path,
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'old_macro_f1': metrics['old_macro_f1'],
            'new_f1': metrics['new_f1'],
            'auroc_known_vs_unknown': metrics['auroc_known_vs_unknown'],
            'num_known_classes': int(split_meta['num_known_classes']),
            'unknown_open_label': unknown_open_label,
        }
    ])
    save_dataframe(summary_df, summary_path)
    save_dataframe(report_df.T.reset_index().rename(columns={'index': 'class_or_avg'}), report_path)

    if args.save_predictions:
        pred_df = test_dataset.df.copy().reset_index(drop=True)
        pred_df['true_open_label'] = y_true
        pred_df['pred_open_label'] = y_pred
        pred_df['pred_known_label'] = pred_dict['pred_known'].numpy().astype(int)
        pred_df['novel_score'] = novel_scores
        pred_df['known_score'] = pred_dict['known_score'].numpy().astype(float)
        pred_df['nearest_known'] = pred_dict['nearest_known'].numpy().astype(int)
        pred_df['nearest_dist'] = pred_dict['nearest_dist'].numpy().astype(float)
        pred_df['is_pred_novel'] = pred_dict['novel_mask'].numpy().astype(int)
        save_dataframe(pred_df, pred_path)

    if args.whether_tsne:
        emb_path = os.path.join(args.open_world_path, f"{args.dataset}_{model_tag}_emb_tst_{args.transform_type}_{args.seed}.csv")
        label_path = os.path.join(args.open_world_path, f"{args.dataset}_{model_tag}_y_tst_{args.transform_type}_{args.seed}.csv")
        save_embedding_csv(emb_path, test_outputs['feat'].numpy())
        pd.DataFrame(y_true).to_csv(label_path, index=False, header=False)

    print('\nEvaluation finished.')
    print(f"Checkpoint: {ckpt_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Known-vs-Unknown AUROC: {metrics['auroc_known_vs_unknown']:.4f}")
    print(f"Saved summary to: {summary_path}")
    if args.save_predictions:
        print(f"Saved per-sample predictions to: {pred_path}")


if __name__ == '__main__':
    main()
