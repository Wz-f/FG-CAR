"""
FG-CARE Main Entry Point (DGL Version) with Ablation Study Support
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import config
from model import FGCARE, get_ablation_description
from dataset import collate_fn
from engine import train_epoch, evaluate
import utils


def main():
    print("=" * 60)
    print("FG-CARE: Fine-Grained Cross-Modal Attention Reasoning")
    print("DGL Version with Ablation Study Support")
    print("=" * 60)
    
    utils.set_seed(42)
    
    # ============================================================
    # 消融实验配置
    # 可选: 'full', 'no_semantic_aug', 'no_graph_cl', 
    #       'no_global_cl', 'no_fine_grained', 'no_caf'
    # ============================================================
    ablation_mode = config.ablation_mode
    print(f"\nAblation Mode: {get_ablation_description(ablation_mode)}")
    
    dataset_name = "we"
    print(f"Dataset: {dataset_name}")
    
    # 根据数据集自动调整图像特征维度
    image_feat_dim = config.get_image_feat_dim(dataset_name)

    if dataset_name == "me15":
        train_dataset, test_dataset = utils.set_up_mediaeval2015()
    elif dataset_name == "we":
        train_dataset, test_dataset = utils.set_up_weibo()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 创建模型，传入消融模式
    model = FGCARE(
        text_dim=config.text_feat_dim,
        img_dim=image_feat_dim,
        hidden_dim=config.hidden_dim,
        proj_dim=config.proj_dim,
        num_classes=config.num_classes,
        top_k=config.top_k,
        syntax_window=config.syntax_window,
        sim_threshold=config.sim_threshold,
        gat_heads=config.gat_heads,
        gat_layers=config.gat_layers,
        gat_dropout=config.gat_dropout,
        aug_keep_ratio=config.aug_base_keep_ratio,
        random_drop_ratio=config.random_drop_ratio,
        temperature=config.temperature,
        lambda_graph=config.lambda_graph,
        lambda_global=config.lambda_global,
        ablation_mode=ablation_mode  # 传入消融模式
    )
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Loss weights: λ_graph={model.lambda_graph}, λ_global={model.lambda_global}")
    
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    num_training_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = 0
    
    # 记录训练日志
    history = []
    
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{config.epochs} {'='*20}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        print(f"\nTraining - Loss: {train_metrics['total_loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']*100:.2f}%")
        
        val_metrics = evaluate(model, test_loader, device, epoch)
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_acc = val_metrics['accuracy']
            best_epoch = epoch
            save_path = f'{config.root_dir}best_fgcare_{dataset_name}_{ablation_mode}.pt'
            torch.save(model.state_dict(), save_path)
        
        # 记录本轮指标（结构化字典）
        report = val_metrics['report']
        fake = report.get('1', {})
        real = report.get('0', {})
        history.append({
            'epoch':    epoch,
            'acc':      val_metrics['accuracy'],
            'fake_pre': fake.get('precision', 0) * 100,
            'fake_rec': fake.get('recall',    0) * 100,
            'fake_f1':  fake.get('f1-score',  0) * 100,
            'real_pre': real.get('precision', 0) * 100,
            'real_rec': real.get('recall',    0) * 100,
            'real_f1':  real.get('f1-score',  0) * 100,
        })
        
        # 如果是第18轮，输出表格日志文件
        if epoch == 18:
            log_path = "training_log_epoch18.txt"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("FG-CARE Training Log (Epoch 1-18)\n")
                f.write(f"Ablation Mode: {ablation_mode}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write("=" * 76 + "\n")
                f.write(f"{'Epoch':<7} {'Acc.':>8}   "
                        f"{'Fake News':^26}   {'Real News':^26}\n")
                f.write(f"{'':7} {'':>8}   "
                        f"{'Pre.':>7} {'Rec.':>7} {'F1':>7}   "
                        f"{'Pre.':>7} {'Rec.':>7} {'F1':>7}\n")
                f.write("-" * 76 + "\n")
                for h in history:
                    f.write(
                        f"Ep {h['epoch']:<4} {h['acc']:>7.2f}%   "
                        f"{h['fake_pre']:>6.2f}% {h['fake_rec']:>6.2f}% {h['fake_f1']:>6.2f}%   "
                        f"{h['real_pre']:>6.2f}% {h['real_rec']:>6.2f}% {h['real_f1']:>6.2f}%\n"
                    )
                f.write("=" * 76 + "\n")
                f.write(f"Best Results (Epoch {best_epoch}): "
                        f"Acc={best_acc:.2f}%, F1={best_f1:.2f}%\n")
            print(f"\n[System] Epoch 18 log saved to {log_path}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Ablation Mode: {ablation_mode}")
    print(f"Best F1: {best_f1:.2f}%")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Best Epoch: {best_epoch}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()