import argparse
import json
import os
import shutil
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ECGDataset_unseen, ECGDataset_unseen_MHL_stage2
from loss_func import CELoss, MCMILoss, OpenMaxSeparation, PseudoLabelGenerator
from resnet import build_model
from utils import (
    DatasetDefaults,
    compute_open_set_metrics,
    ensure_dir,
    gen_label_csv_unseen_setting,
    gen_label_csv_unseen_setting_2_MHL,
    get_dataset_defaults,
    get_device,
    parse_unseen_classes,
    pin_memory_for_device,
    save_dataframe,
    save_embedding_csv,
    save_json,
    set_seed,
    str2bool,
)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_szhou_data', type=str, default='True')
    parser.add_argument('--dataset', type=str, default='CPSC18')
    parser.add_argument('--Georgia_aug', type=str2bool, default=True)
    parser.add_argument('--model', type=str, default='resnet34')
    parser.add_argument('--data_path', type=str, default='./data_path/')
    parser.add_argument('--whether_tsne', type=str2bool, default=False)
    parser.add_argument('--loss_func', type=str, default='contrastive', help='contrastive / cross_entropy')
    parser.add_argument('--CL_alpha', type=float, default=0.6, help='deprecated, kept for compatibility')
    parser.add_argument('--CL_temp', type=float, default=0.07)
    parser.add_argument('--transform_type', type=str, default='hardneg')

    parser.add_argument('--leads_zdd', type=str, default='all')
    parser.add_argument('--leads', type=int, default=12)
    parser.add_argument('--classes', type=int, default=-1, help='deprecated, automatically inferred')
    parser.add_argument('--Hz', type=int, default=500)
    parser.add_argument('--duration', type=int, default=30)

    parser.add_argument('--epochs', type=int, default=120, help='stage-1 supervised epochs')
    parser.add_argument('--stage2_epochs', type=int, default=40, help='stage-2 pseudo-label discovery epochs')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('--stage2_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--test_results_path', type=str, default='./results/')
    parser.add_argument('--val_results_path', type=str, default='./results_val/')
    parser.add_argument('--open_world_path', type=str, default='./OpenMax/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trn_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--tes_ratio', type=float, default=0.2, help='kept for compatibility')

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--instance_dim', type=int, default=128)
    parser.add_argument('--class_dim', type=int, default=128)
    parser.add_argument('--novel_clusters', type=int, default=1)
    parser.add_argument('--unseen_classes', type=str, default='')
    parser.add_argument('--distance_scale', type=float, default=2.0)
    parser.add_argument('--min_known_score', type=float, default=0.35)
    parser.add_argument('--refresh_pseudo_every', type=int, default=1)
    parser.add_argument('--normalize_signal', type=str2bool, default=False)
    parser.add_argument('--save_split_files', type=str2bool, default=True)
    parser.add_argument('--save_logits_csv', type=str2bool, default=False)
    return parser.parse_args()



def normalize_model_name(name: str) -> str:
    return str(name).replace('-', '_')



def prepare_dataset_args(args) -> DatasetDefaults:
    defaults = get_dataset_defaults(args.dataset)
    args.leads = defaults.leads if args.leads <= 0 else args.leads
    args.Hz = defaults.hz if args.Hz <= 0 else args.Hz
    args.duration = defaults.duration if args.duration <= 0 else args.duration
    args.length = args.Hz * args.duration
    unseen_classes = parse_unseen_classes(args.unseen_classes, defaults.unseen_classes)
    args.unseen_classes_list = unseen_classes
    if args.novel_clusters <= 0:
        args.novel_clusters = max(1, len(unseen_classes))
    args.data_dir = os.path.join(args.data_path, defaults.data_dir_name)
    args.label_file = os.path.join(args.data_path, defaults.label_file_name)
    return defaults



def prepare_paths(args) -> None:
    ensure_dir('models')
    ensure_dir('logs')
    ensure_dir(args.test_results_path)
    ensure_dir(args.val_results_path)
    ensure_dir(args.open_world_path)
    ensure_dir('./results_final')

    model_tag = normalize_model_name(args.model)
    base = f"{args.dataset}_{model_tag}_{args.transform_type}_seed{args.seed}"
    args.stage1_split_csv = os.path.join(args.open_world_path, f"{args.dataset}_{model_tag}_label_{args.transform_type}_Stage1_{args.seed}.csv")
    args.stage2_split_csv = os.path.join(args.open_world_path, f"{args.dataset}_{model_tag}_label_{args.transform_type}_{args.seed}.csv")
    args.stage1_model_path = os.path.join('models', f"{base}_stage1.pth")
    args.final_model_path = args.model_path if args.model_path else os.path.join('models', f"{base}_mcmi.pth")
    args.training_log_path = os.path.join('logs', f"{base}.json")
    args.val_csv_path = os.path.join(args.val_results_path, f"{base}.csv")
    args.trn_y_best_path = os.path.join(args.open_world_path, f"{args.dataset}_{model_tag}_y_trn_{args.transform_type}_{args.seed}.csv")
    args.tst_y_path = os.path.join(args.open_world_path, f"{args.dataset}_{model_tag}_y_tst_{args.transform_type}_{args.seed}.csv")
    args.trn_probs_best_path = os.path.join(args.open_world_path, f"{args.dataset}_{model_tag}_trn_probs_{args.transform_type}_{args.seed}.csv")
    args.tst_probs_path = os.path.join(args.open_world_path, f"{args.dataset}_{model_tag}_tst_probs_{args.transform_type}_{args.seed}.csv")
    args.TSNEemb_trn_best_path = os.path.join(args.open_world_path, f"{args.dataset}_{model_tag}_emb_trn_{args.transform_type}_{args.seed}.csv")
    args.TSNEemb_tst_path = os.path.join(args.open_world_path, f"{args.dataset}_{model_tag}_emb_tst_{args.transform_type}_{args.seed}.csv")



def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory_for_device(device),
        drop_last=False,
    )



def build_split_files(args) -> Dict[str, object]:
    stage1_df = gen_label_csv_unseen_setting(
        data_dir=args.data_dir,
        label_file=args.label_file,
        output_csv=args.stage1_split_csv,
        unseen_class_name=args.unseen_classes_list,
        trn_ratio=args.trn_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        transform_type=args.transform_type,
    )
    stage2_seen_counts = gen_label_csv_unseen_setting_2_MHL(
        label_file=args.label_file,
        output_csv=args.stage2_split_csv,
        unseen_class_name=args.unseen_classes_list,
        trn_ratio=args.trn_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    stage1_df = pd.read_csv(args.stage1_split_csv)
    num_known_classes = int(stage1_df.loc[stage1_df['label'] >= 0, 'label'].max()) + 1
    split_meta = {
        'num_known_classes': num_known_classes,
        'unknown_open_label': num_known_classes,
        'seen_train_count': int((stage1_df['split'] == 'train').sum()),
        'seen_valid_count': int((stage1_df['split'] == 'valid').sum()),
        'stage2_seen_counts': stage2_seen_counts,
        'unseen_classes': list(args.unseen_classes_list),
    }
    return split_meta



def instantiate_model(args, num_known_classes: int):
    num_open_classes = int(num_known_classes + max(1, args.novel_clusters))
    model = build_model(
        args.model,
        input_channels=args.leads,
        num_classes=num_open_classes,
        known_classes=num_known_classes,
        feature_dim=args.feature_dim,
        instance_dim=args.instance_dim,
        class_dim=args.class_dim,
    )
    return model



def to_device(batch_tensor, device):
    return batch_tensor.to(device, non_blocking=True)



def detach_to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()



def train_stage1_epoch(loader, model, criterion, optimizer, device, num_known_classes: int):
    model.train()
    running = {'loss': 0.0, 'sup_ins': 0.0, 'ce': 0.0}
    count = 0

    for view1, view2, labels in tqdm(loader, leave=False):
        view1 = to_device(view1, device)
        view2 = to_device(view2, device)
        labels = to_device(labels, device)

        out1 = model(view1, return_dict=True)
        out2 = model(view2, return_dict=True)
        if isinstance(criterion, MCMILoss):
            loss_dict = criterion.forward_initial_pair(out1, out2, labels, known_classes=num_known_classes)
        else:
            ce1 = criterion(None, out1['logits'][:, :num_known_classes], labels)
            ce2 = criterion(None, out2['logits'][:, :num_known_classes], labels)
            ce = 0.5 * (ce1 + ce2)
            zero = ce.detach() * 0.0
            loss_dict = {'loss': ce, 'sup_ins': zero, 'ce': ce}
        loss = loss_dict['loss']

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        count += bs
        for k in running:
            running[k] += float(loss_dict[k].item()) * bs

    if count == 0:
        return {k: 0.0 for k in running}
    return {k: v / count for k, v in running.items()}



def train_stage2_epoch(loader, model, criterion, optimizer, device):
    model.train()
    running = {'loss': 0.0, 'psup_ins': 0.0, 'psup_cls': 0.0, 'mi': 0.0}
    count = 0

    for view1, view2, pseudo_labels in tqdm(loader, leave=False):
        view1 = to_device(view1, device)
        view2 = to_device(view2, device)
        pseudo_labels = to_device(pseudo_labels, device)

        out1 = model(view1, return_dict=True)
        out2 = model(view2, return_dict=True)
        loss_dict = criterion.forward_continuous_pair(out1, out2, pseudo_labels)
        loss = loss_dict['loss']

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = pseudo_labels.size(0)
        count += bs
        for k in running:
            running[k] += float(loss_dict[k].item()) * bs

    if count == 0:
        return {k: 0.0 for k in running}
    return {k: v / count for k, v in running.items()}


@torch.no_grad()

def collect_outputs(loader, model, device):
    model.eval()
    feats, ins_embs, cls_embs, logits_list, labels_list = [], [], [], [], []
    for batch in tqdm(loader, leave=False):
        if len(batch) == 4:
            x, _, labels, _ = batch
        elif len(batch) == 3:
            x, labels, _ = batch
        elif len(batch) == 2:
            x, labels = batch
        else:
            raise ValueError(f'Unexpected batch format with length {len(batch)}')
        x = to_device(x, device)
        out = model(x, return_dict=True)
        feats.append(detach_to_cpu(out['feat']))
        ins_embs.append(detach_to_cpu(out['instance_emb']))
        cls_embs.append(detach_to_cpu(out['class_emb']))
        logits_list.append(detach_to_cpu(out['logits']))
        labels_list.append(detach_to_cpu(labels.long()))

    return {
        'feat': torch.cat(feats, dim=0) if feats else torch.empty(0),
        'instance_emb': torch.cat(ins_embs, dim=0) if ins_embs else torch.empty(0),
        'class_emb': torch.cat(cls_embs, dim=0) if cls_embs else torch.empty(0),
        'logits': torch.cat(logits_list, dim=0) if logits_list else torch.empty(0),
        'labels': torch.cat(labels_list, dim=0) if labels_list else torch.empty(0, dtype=torch.long),
    }


@torch.no_grad()

def fit_separator_from_loader(loader, model, device, separator: OpenMaxSeparation):
    outputs = collect_outputs(loader, model, device)
    separator.fit(outputs['feat'], outputs['labels'])
    return separator, outputs


@torch.no_grad()

def evaluate_open_world_known_only(loader, model, device, separator: Optional[OpenMaxSeparation], unknown_label: int):
    outputs = collect_outputs(loader, model, device)
    y_true = outputs['labels'].numpy().astype(int)

    if outputs['logits'].numel() == 0:
        empty_metrics = {'accuracy': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0, 'weighted_f1': 0.0, 'old_macro_f1': 0.0, 'new_f1': 0.0, 'auroc_known_vs_unknown': float('nan')}
        return empty_metrics, pd.DataFrame(), outputs, np.array([], dtype=int), np.array([], dtype=float)

    if separator is None:
        y_pred = torch.argmax(outputs['logits'], dim=1).numpy().astype(int)
        novel_scores = np.zeros_like(y_pred, dtype=np.float32)
    else:
        pred = separator.predict_open_labels(outputs['feat'], outputs['logits'], unknown_label=unknown_label)
        y_pred = pred['pred_open'].cpu().numpy().astype(int)
        novel_scores = pred['novel_score'].cpu().numpy().astype(np.float32)

    metrics, report_df = compute_open_set_metrics(y_true, y_pred, unknown_label=unknown_label, novel_scores=novel_scores)
    return metrics, report_df, outputs, y_pred, novel_scores


@torch.no_grad()

def refresh_pseudo_labels(discover_loader, model, device, separator: OpenMaxSeparation, generator: PseudoLabelGenerator):
    outputs = collect_outputs(discover_loader, model, device)
    separation = separator.predict(outputs['feat'], outputs['logits'])
    pseudo_labels = generator.generate_pseudo_labels(
        features=outputs['feat'],
        mask_novel=separation['novel_mask'],
        predictions=outputs['logits'],
        num_known_classes=separator.num_known_classes,
        known_predictions=separation,
        cluster_features=outputs['instance_emb'],
    )
    pseudo_stats = {
        'pseudo_known': int((pseudo_labels < separator.num_known_classes).sum().item()),
        'pseudo_novel': int((pseudo_labels >= separator.num_known_classes).sum().item()),
        'pred_novel_mask': int(separation['novel_mask'].sum().item()),
        'pred_known_mask': int((~separation['novel_mask']).sum().item()),
    }
    return pseudo_labels.cpu().numpy().astype(np.int64), outputs, separation, pseudo_stats



def save_checkpoint(path: str, model, optimizer, args, separator: OpenMaxSeparation, split_meta: Dict[str, object], history: Dict[str, object], best_val_metric: float):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'split_meta': split_meta,
        'separator_state': separator.state_dict(),
        'best_val_metric': float(best_val_metric),
        'history': history,
    }
    torch.save(checkpoint, path)



def export_embeddings_and_labels(train_outputs, test_outputs, test_labels_path: str, train_labels_path: str, train_emb_path: str, test_emb_path: str, save_logits: bool = False, train_logits_path: Optional[str] = None, test_logits_path: Optional[str] = None):
    save_embedding_csv(train_emb_path, train_outputs['feat'].numpy())
    save_embedding_csv(test_emb_path, test_outputs['feat'].numpy())
    pd.DataFrame(train_outputs['labels'].numpy()).to_csv(train_labels_path, index=False, header=False)
    pd.DataFrame(test_outputs['labels'].numpy()).to_csv(test_labels_path, index=False, header=False)
    if save_logits and train_logits_path is not None and test_logits_path is not None:
        pd.DataFrame(train_outputs['logits'].numpy()).to_csv(train_logits_path, index=False, header=False)
        pd.DataFrame(test_outputs['logits'].numpy()).to_csv(test_logits_path, index=False, header=False)



def main():
    args = parse_args()
    set_seed(args.seed)
    defaults = prepare_dataset_args(args)
    prepare_paths(args)
    split_meta = build_split_files(args)
    num_known_classes = int(split_meta['num_known_classes'])
    unknown_open_label = int(split_meta['unknown_open_label'])

    if str(args.loss_func).lower() == 'cross_entropy':
        args.stage2_epochs = 0

    device = get_device(args.use_gpu, args.gpu_id)
    if device.type == 'cpu':
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    train_dataset = ECGDataset_unseen(
        phase='train',
        data_dir=args.data_dir,
        label_csv=args.stage1_split_csv,
        leads=args.leads,
        length=args.length,
        transform_type=args.transform_type,
        dual_view=True,
        label_key='label',
        normalize=args.normalize_signal,
    )
    val_dataset = ECGDataset_unseen(
        phase='valid',
        data_dir=args.data_dir,
        label_csv=args.stage1_split_csv,
        leads=args.leads,
        length=args.length,
        transform_type=args.transform_type,
        dual_view=False,
        label_key='label',
        normalize=args.normalize_signal,
    )
    known_train_eval_dataset = ECGDataset_unseen(
        phase='train',
        data_dir=args.data_dir,
        label_csv=args.stage1_split_csv,
        leads=args.leads,
        length=args.length,
        transform_type=args.transform_type,
        dual_view=False,
        label_key='label',
        normalize=args.normalize_signal,
    )
    discover_raw_dataset = ECGDataset_unseen_MHL_stage2(
        phase='train_valid',
        data_dir=args.data_dir,
        label_csv=args.stage2_split_csv,
        leads=args.leads,
        length=args.length,
        transform_type=args.transform_type,
        dual_view=False,
        label_key='open_label',
        normalize=args.normalize_signal,
    )
    discover_train_dataset = ECGDataset_unseen_MHL_stage2(
        phase='train_valid',
        data_dir=args.data_dir,
        label_csv=args.stage2_split_csv,
        leads=args.leads,
        length=args.length,
        transform_type=args.transform_type,
        dual_view=True,
        label_key='open_label',
        normalize=args.normalize_signal,
    )
    test_dataset = ECGDataset_unseen_MHL_stage2(
        phase='test',
        data_dir=args.data_dir,
        label_csv=args.stage2_split_csv,
        leads=args.leads,
        length=args.length,
        transform_type=args.transform_type,
        dual_view=False,
        label_key='open_label',
        normalize=args.normalize_signal,
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device)
    known_train_eval_loader = make_loader(known_train_eval_dataset, args.batch_size, False, args.num_workers, device)
    discover_raw_loader = make_loader(discover_raw_dataset, args.batch_size, False, args.num_workers, device)
    discover_train_loader = make_loader(discover_train_dataset, args.batch_size, True, args.num_workers, device)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device)

    model = instantiate_model(args, num_known_classes).to(device)

    use_mcmi = str(args.loss_func).lower() != 'cross_entropy'
    criterion = MCMILoss(temp=args.CL_temp, alpha=args.alpha, beta=args.beta) if use_mcmi else CELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    history = {'stage1': [], 'stage2': []}
    t_start = time.time()

    # ---------- Stage 1: supervised training on known classes ----------
    best_val_metric = -1.0
    best_stage1_report = pd.DataFrame()
    for epoch in range(1, args.epochs + 1):
        train_stats = train_stage1_epoch(train_loader, model, criterion, optimizer, device, num_known_classes)
        scheduler.step()
        val_metrics, report_df, _, _, _ = evaluate_open_world_known_only(val_loader, model, device, separator=None, unknown_label=unknown_open_label)

        epoch_log = {
            'epoch': epoch,
            **train_stats,
            **{f'val_{k}': float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        }
        history['stage1'].append(epoch_log)

        if val_metrics['macro_f1'] >= best_val_metric:
            best_val_metric = val_metrics['macro_f1']
            best_stage1_report = report_df.copy()
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args),
                    'split_meta': split_meta,
                    'best_val_metric': float(best_val_metric),
                },
                args.stage1_model_path,
            )

        print(
            f"[Stage1][Epoch {epoch:03d}/{args.epochs:03d}] "
            f"loss={train_stats['loss']:.4f} sup_ins={train_stats['sup_ins']:.4f} ce={train_stats['ce']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

    stage1_checkpoint = torch.load(args.stage1_model_path, map_location=device)
    model.load_state_dict(stage1_checkpoint['model_state_dict'])

    separator = OpenMaxSeparation(
        num_known_classes=num_known_classes,
        distance_scale=args.distance_scale,
        min_known_score=args.min_known_score,
    )
    separator, train_known_outputs = fit_separator_from_loader(known_train_eval_loader, model, device, separator)

    final_best_metric = best_val_metric
    final_report = best_stage1_report.copy()
    final_outputs_for_export = None

    # ---------- Stage 2: pseudo-label discovery ----------
    if args.stage2_epochs > 0:
        stage2_optimizer = torch.optim.AdamW(model.parameters(), lr=args.stage2_lr, weight_decay=args.weight_decay)
        stage2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(stage2_optimizer, T_max=max(1, args.stage2_epochs))
        generator = PseudoLabelGenerator(num_known_classes=num_known_classes, n_clusters=args.novel_clusters, random_state=args.seed)
        best_stage2_metric = -1.0

        for epoch in range(1, args.stage2_epochs + 1):
            if epoch == 1 or ((epoch - 1) % max(1, args.refresh_pseudo_every) == 0):
                separator, train_known_outputs = fit_separator_from_loader(known_train_eval_loader, model, device, separator)
                pseudo_labels, discover_outputs, separation, pseudo_stats = refresh_pseudo_labels(discover_raw_loader, model, device, separator, generator)
                discover_train_dataset.set_pseudo_labels(pseudo_labels)
            train_stats = train_stage2_epoch(discover_train_loader, model, criterion, stage2_optimizer, device)
            stage2_scheduler.step()
            separator, train_known_outputs = fit_separator_from_loader(known_train_eval_loader, model, device, separator)
            val_metrics, report_df, _, y_pred_val, novel_scores_val = evaluate_open_world_known_only(
                val_loader, model, device, separator=separator, unknown_label=unknown_open_label
            )

            epoch_log = {
                'epoch': epoch,
                **train_stats,
                **pseudo_stats,
                **{f'val_{k}': float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
            }
            history['stage2'].append(epoch_log)

            if val_metrics['macro_f1'] >= best_stage2_metric:
                best_stage2_metric = val_metrics['macro_f1']
                final_best_metric = best_stage2_metric
                final_report = report_df.copy()
                save_checkpoint(
                    args.final_model_path,
                    model,
                    stage2_optimizer,
                    args,
                    separator,
                    split_meta,
                    history,
                    best_val_metric=best_stage2_metric,
                )

            print(
                f"[Stage2][Epoch {epoch:03d}/{args.stage2_epochs:03d}] "
                f"loss={train_stats['loss']:.4f} ins={train_stats['psup_ins']:.4f} cls={train_stats['psup_cls']:.4f} mi={train_stats['mi']:.4f} "
                f"pseudo_known={pseudo_stats['pseudo_known']} pseudo_novel={pseudo_stats['pseudo_novel']} "
                f"val_macro_f1={val_metrics['macro_f1']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            )

        if os.path.exists(args.final_model_path):
            checkpoint = torch.load(args.final_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            separator = OpenMaxSeparation.from_state_dict(checkpoint['separator_state'])
        else:
            save_checkpoint(args.final_model_path, model, stage2_optimizer, args, separator, split_meta, history, best_val_metric=best_stage2_metric)
    else:
        save_checkpoint(args.final_model_path, model, optimizer, args, separator, split_meta, history, best_val_metric=best_val_metric)

    # ---------- Export embeddings / labels (optional, for compatibility and TSNE) ----------
    separator, train_known_outputs = fit_separator_from_loader(known_train_eval_loader, model, device, separator)
    test_outputs = collect_outputs(test_loader, model, device)
    final_outputs_for_export = test_outputs

    if args.whether_tsne:
        export_embeddings_and_labels(
            train_outputs=train_known_outputs,
            test_outputs=test_outputs,
            test_labels_path=args.tst_y_path,
            train_labels_path=args.trn_y_best_path,
            train_emb_path=args.TSNEemb_trn_best_path,
            test_emb_path=args.TSNEemb_tst_path,
            save_logits=args.save_logits_csv,
            train_logits_path=args.trn_probs_best_path if args.save_logits_csv else None,
            test_logits_path=args.tst_probs_path if args.save_logits_csv else None,
        )

    # ---------- Save validation summary ----------
    elapsed = time.time() - t_start
    summary = {
        'dataset': args.dataset,
        'model': args.model,
        'transform_type': args.transform_type,
        'seed': args.seed,
        'num_known_classes': num_known_classes,
        'unknown_open_label': unknown_open_label,
        'unseen_classes': args.unseen_classes_list,
        'stage1_model_path': args.stage1_model_path,
        'final_model_path': args.final_model_path,
        'stage1_split_csv': args.stage1_split_csv,
        'stage2_split_csv': args.stage2_split_csv,
        'elapsed_seconds': elapsed,
        'best_val_macro_f1': float(final_best_metric),
        'history': history,
    }
    save_json(summary, args.training_log_path)
    if not final_report.empty:
        save_dataframe(final_report.T.reset_index().rename(columns={'index': 'class_or_avg'}), args.val_csv_path)

    print('\nTraining finished.')
    print(f"Best validation macro-F1: {final_best_metric:.4f}")
    print(f"Final checkpoint: {args.final_model_path}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
