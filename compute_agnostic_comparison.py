#!/usr/bin/env python3
"""
Controlled comparison: semantic mIoU vs class-agnostic mIoU
on the SAME 100 frames with the SAME per-frame averaging protocol.

Loads pre-computed SepSAM segment_ids from sepsam_agnostic_eval/,
computes both metrics, and reports the gap.
"""

import os, sys, glob, numpy as np, json, time

SEQ = "08"
DATA_ROOT = f"/home/adlab35/datasets/semantickitti/dataset/sequences/{SEQ}"
LABEL_DIR = os.path.join(DATA_ROOT, "labels")
SEG_DIR = "/home/adlab35/sepsam_agnostic_eval/segment_ids"
OUT_PATH = "/home/adlab35/agnostic_comparison_result.json"

THING_CLASSES = {
    10: 'car', 11: 'bicycle', 13: 'bus', 15: 'motorcycle',
    18: 'truck', 20: 'other-vehicle', 30: 'person',
    31: 'bicyclist', 32: 'motorcyclist'
}

def load_label(path):
    labels = np.fromfile(path, dtype=np.int32)
    return labels & 0xFFFF, (labels >> 16) & 0xFFFF

def compute_frame_semantic_miou(pred_labels, gt_sem, gt_inst):
    """Standard SemanticKITTI mIoU: per-class, per-GT-instance best-IoU matching.
    Returns: (mean_iou, n_instances, per_class_dict)"""
    per_class_ious = {}
    all_inst_ious = []

    for cls_id, cls_name in THING_CLASSES.items():
        cls_mask = gt_sem == cls_id
        if cls_mask.sum() == 0:
            continue
        gt_insts = np.unique(gt_inst[cls_mask])
        gt_insts = gt_insts[gt_insts > 0]

        cls_ious = []
        for iid in gt_insts:
            gt_mask = cls_mask & (gt_inst == iid)
            if gt_mask.sum() < 10:
                continue
            # Find best IoU among all predictions (class-agnostic matching)
            best_iou = 0.0
            pred_insts = np.unique(pred_labels[gt_mask])
            pred_insts = pred_insts[pred_insts >= 0]
            for pid in pred_insts:
                pred_mask = pred_labels == pid
                inter = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                iou = inter / max(1, union)
                best_iou = max(best_iou, iou)
            cls_ious.append(best_iou)
            all_inst_ious.append((cls_name, gt_mask.sum(), best_iou))

        if cls_ious:
            per_class_ious[cls_name] = {
                'count': len(cls_ious),
                'mean_iou': float(np.mean(cls_ious)),
                'median_iou': float(np.median(cls_ious)),
            }

    # Semantic mIoU: mean of per-class means
    if per_class_ious:
        semantic_miou = np.mean([v['mean_iou'] for v in per_class_ious.values()])
    else:
        semantic_miou = 0.0

    # Class-agnostic mIoU: mean of ALL instance IoUs (no per-class grouping)
    agnostic_miou = np.mean([x[2] for x in all_inst_ious]) if all_inst_ious else 0.0

    n_total = len(all_inst_ious)
    return semantic_miou, agnostic_miou, n_total, per_class_ious, all_inst_ious


print("=" * 70)
print("SepSAM 语义 vs 类无关 mIoU 对照实验")
print("=" * 70)

seg_files = sorted(glob.glob(os.path.join(SEG_DIR, "*_segid.npy")))
print(f"找到 {len(seg_files)} 个已保存的分割结果")

frame_results = []
all_instance_ious = []  # (cls_name, n_pts, iou, frame)

t_start = time.time()
valid_frames = 0

for i, seg_path in enumerate(seg_files):
    frame = os.path.splitext(os.path.basename(seg_path))[0].replace('_segid', '')
    label_path = os.path.join(LABEL_DIR, frame + ".label")

    if not os.path.exists(label_path):
        print(f"  [{i+1}/{len(seg_files)}] {frame}: 标签文件缺失，跳过")
        continue

    pred_ids = np.load(seg_path)
    gt_sem, gt_inst = load_label(label_path)

    sem_miou, agn_miou, n_inst, per_cls, inst_list = compute_frame_semantic_miou(
        pred_ids, gt_sem, gt_inst
    )

    frame_results.append({
        'frame': frame,
        'semantic_miou': sem_miou,
        'agnostic_miou': agn_miou,
        'n_instances': n_inst,
        'per_class': per_cls,
    })

    for cls_name, n_pts, iou in inst_list:
        all_instance_ious.append({
            'frame': frame,
            'class': cls_name,
            'n_points': n_pts,
            'iou': iou,
        })

    valid_frames += 1

    if (i + 1) % 20 == 0:
        elapsed = time.time() - t_start
        print(f"  [{i+1}/{len(seg_files)}] {frame}: "
              f"语义mIoU={sem_miou*100:.1f}%  类无关mIoU={agn_miou*100:.1f}%  "
              f"实例数={n_inst}  ({elapsed:.0f}s)", flush=True)

elapsed = time.time() - t_start
print(f"\n处理完成: {valid_frames} 有效帧, 耗时 {elapsed:.1f}s")

# ============ Aggregate Results ============
sem_mious = [f['semantic_miou'] for f in frame_results]
agn_mious = [f['agnostic_miou'] for f in frame_results]
total_inst = sum(f['n_instances'] for f in frame_results)

# Per-class aggregation across all frames
from collections import defaultdict
global_per_class = defaultdict(lambda: {'count': 0, 'ious': []})
for inst in all_instance_ious:
    global_per_class[inst['class']]['count'] += 1
    global_per_class[inst['class']]['ious'].append(inst['iou'])

print()
print("=" * 70)
print("对照实验结果")
print("=" * 70)
print(f"评估帧数:       {valid_frames}")
print(f"GT thing 实例:  {total_inst}")
print()
print(f"===== 语义 mIoU (标准: per-class → per-frame → mean) =====")
print(f"  平均:   {np.mean(sem_mious)*100:.2f}%")
print(f"  中位数: {np.median(sem_mious)*100:.2f}%")
print(f"  标准差: {np.std(sem_mious)*100:.2f}%")
print(f"  P25:    {np.percentile(sem_mious, 25)*100:.2f}%")
print(f"  P75:    {np.percentile(sem_mious, 75)*100:.2f}%")
print()
print(f"===== 类无关 mIoU (全局 per-instance mean) =====")
print(f"  平均:   {np.mean(agn_mious)*100:.2f}%")
print(f"  中位数: {np.median(agn_mious)*100:.2f}%")
print(f"  标准差: {np.std(agn_mious)*100:.2f}%")
print()

# Also compute: per-frame agnostic (mean of per-frame agnostic means)
per_frame_agn = np.mean(agn_mious)
print(f"===== 对比 =====")
print(f"  语义 mIoU (per-class, per-frame mean):      {np.mean(sem_mious)*100:.2f}%")
print(f"  类无关 mIoU (per-instance, per-frame mean): {per_frame_agn*100:.2f}%")
print(f"  差值:                                        {(np.mean(sem_mious) - per_frame_agn)*100:+.1f} 百分点")
print()

# Raw global mean (not per-frame)
all_ious_raw = [inst['iou'] for inst in all_instance_ious]
print(f"  全局 per-instance mean IoU (不过帧):        {np.mean(all_ious_raw)*100:.2f}%")
print(f"  全局 per-instance median IoU:               {np.median(all_ious_raw)*100:.2f}%")
print()

print(f"===== Per-Class Breakdown =====")
print(f"  {'Class':<16} {'Count':>6} {'MeanIoU':>9} {'MedIoU':>9} {'IoU≥50%':>8}")
print(f"  {'-'*16} {'-'*6} {'-'*9} {'-'*9} {'-'*8}")
for cls_name in sorted(global_per_class.keys()):
    info = global_per_class[cls_name]
    ious = np.array(info['ious'])
    print(f"  {cls_name:<16} {info['count']:>6} {ious.mean()*100:>8.1f}% {np.median(ious)*100:>8.1f}% {(ious>=0.5).sum()/len(ious)*100:>7.1f}%")

print()
print(f"===== By GT Instance Size =====")
size_bins = [(5, 30, '<30 pts'), (30, 100, '30-100'), (100, 500, '100-500'), (500, 99999, '>500')]
for lo, hi, label in size_bins:
    ious = [inst['iou'] for inst in all_instance_ious if lo <= inst['n_points'] < hi]
    if ious:
        print(f"  {label:<14} {len(ious):>5} instances  meanIoU={np.mean(ious)*100:.1f}%  medIoU={np.median(ious)*100:.1f}%")

# ============ Save Report ============
report = {
    'description': 'Controlled semantic vs class-agnostic mIoU comparison on same 100 frames',
    'num_frames': valid_frames,
    'num_gt_instances': total_inst,
    'semantic_miou_per_frame_mean': float(np.mean(sem_mious)),
    'semantic_miou_per_frame_median': float(np.median(sem_mious)),
    'semantic_miou_per_frame_std': float(np.std(sem_mious)),
    'agnostic_miou_per_frame_mean': float(np.mean(agn_mious)),
    'agnostic_miou_per_frame_median': float(np.median(agn_mious)),
    'agnostic_miou_per_frame_std': float(np.std(agn_mious)),
    'agnostic_raw_global_mean': float(np.mean(all_ious_raw)),
    'agnostic_raw_global_median': float(np.median(all_ious_raw)),
    'gap_semantic_minus_agnostic_pp': float((np.mean(sem_mious) - np.mean(agn_mious)) * 100),
    'per_class': {k: {'count': v['count'], 'mean_iou': float(np.mean(v['ious'])),
                      'median_iou': float(np.median(v['ious']))}
                  for k, v in global_per_class.items()},
    'per_frame': frame_results,
}

with open(OUT_PATH, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\n报告已保存: {OUT_PATH}")
