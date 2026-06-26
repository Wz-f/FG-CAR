"""
FG-CARE Training and Evaluation Engine (DGL Version)
"""
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def print_train_report(report, loss):
    """
    打印训练报告（仅显示每个类别的详细指标）
    
    Args:
        report: sklearn的classification_report字典
        loss: 损失值
    """
    print(f"\n{'='*60}")
    print(f"Training Report")
    print(f"{'='*60}")
    print(f"Training Loss: {loss:.4f}")
    
    # Class 0 - Real News
    class_0 = report.get('0', {})
    print(f"\nClass 0 (Real News):")
    print(f"  Precision: {class_0.get('precision', 0)*100:.2f}%")
    print(f"  Recall:    {class_0.get('recall', 0)*100:.2f}%")
    print(f"  F1-score:  {class_0.get('f1-score', 0)*100:.2f}%")
    print(f"  Support:   {int(class_0.get('support', 0))}")
    
    # Class 1 - Fake News
    class_1 = report.get('1', {})
    print(f"\nClass 1 (Fake News):")
    print(f"  Precision: {class_1.get('precision', 0)*100:.2f}%")
    print(f"  Recall:    {class_1.get('recall', 0)*100:.2f}%")
    print(f"  F1-score:  {class_1.get('f1-score', 0)*100:.2f}%")
    print(f"  Support:   {int(class_1.get('support', 0))}")
    print(f"{'='*60}")


def print_validation_report(report, loss):
    """
    打印验证报告（包含每个类别和整体指标）
    
    Args:
        report: sklearn的classification_report字典
        loss: 损失值
    """
    print(f"\n{'='*60}")
    print(f"Validation Report")
    print(f"{'='*60}")
    print(f"Validation Loss: {loss:.4f}")
    
    # Class 0 - Real News
    class_0 = report.get('0', {})
    print(f"\nClass 0 (Real News):")
    print(f"  Precision: {class_0.get('precision', 0)*100:.2f}%")
    print(f"  Recall:    {class_0.get('recall', 0)*100:.2f}%")
    print(f"  F1-score:  {class_0.get('f1-score', 0)*100:.2f}%")
    print(f"  Support:   {int(class_0.get('support', 0))}")
    
    # Class 1 - Fake News
    class_1 = report.get('1', {})
    print(f"\nClass 1 (Fake News):")
    print(f"  Precision: {class_1.get('precision', 0)*100:.2f}%")
    print(f"  Recall:    {class_1.get('recall', 0)*100:.2f}%")
    print(f"  F1-score:  {class_1.get('f1-score', 0)*100:.2f}%")
    print(f"  Support:   {int(class_1.get('support', 0))}")
    
    # Overall metrics
    accuracy = report.get('accuracy', 0)
    macro_avg = report.get('macro avg', {})
    
    print(f"\n{'-'*60}")
    print(f"Overall Metrics:")
    print(f"  Accuracy:             {accuracy*100:.2f}%")
    print(f"  Macro Avg Precision:  {macro_avg.get('precision', 0)*100:.2f}%")
    print(f"  Macro Avg Recall:     {macro_avg.get('recall', 0)*100:.2f}%")
    print(f"  Macro Avg F1-score:   {macro_avg.get('f1-score', 0)*100:.2f}%")
    print(f"{'='*60}")


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """单个训练 epoch"""
    model.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_graph_loss = 0.0
    total_global_loss = 0.0
    total_alpha = 0.0
    
    predictions = []
    targets = []
    
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for step, batch in enumerate(pbar):
        text_global = batch['text_global'].to(device)
        img_global = batch['img_global'].to(device)
        text_local = batch['text_local']
        img_local = batch['img_local']
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits, loss, loss_dict = model(
            text_global=text_global,
            text_local_list=text_local,
            img_global=img_global,
            img_local_list=img_local,
            labels=labels,
            training=True
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss_dict.get('total_loss', 0)
        total_cls_loss += loss_dict.get('L_cls', 0)
        total_graph_loss += loss_dict.get('L_graph', 0)
        total_global_loss += loss_dict.get('L_global', 0)
        total_alpha += loss_dict.get('alpha_mean', 0)
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        targets.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f"{loss_dict.get('total_loss', 0):.4f}",
            'cls': f"{loss_dict.get('L_cls', 0):.4f}",
            'α': f"{loss_dict.get('alpha_mean', 0):.3f}"
        })
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {
        'total_loss': avg_loss,
        'cls_loss': total_cls_loss / num_batches,
        'graph_loss': total_graph_loss / num_batches,
        'global_loss': total_global_loss / num_batches,
        'alpha': total_alpha / num_batches
    }
    
    report = classification_report(targets, predictions, output_dict=True, 
                                   labels=[0, 1], zero_division=0, 
                                   target_names=['0', '1'])
    avg_metrics['accuracy'] = accuracy_score(targets, predictions)
    avg_metrics['report'] = report
    
    # 打印训练报告（仅类别指标，无整体指标）
    print_train_report(report, avg_loss)
    
    return avg_metrics


def evaluate(model, dataloader, device, epoch=1):
    """评估函数"""
    model.eval()
    
    total_loss = 0.0
    predictions = []
    targets = []
    all_alphas = []
    
    pbar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}")
    
    with torch.no_grad():
        for batch in pbar:
            text_global = batch['text_global'].to(device)
            img_global = batch['img_global'].to(device)
            text_local = batch['text_local']
            img_local = batch['img_local']
            labels = batch['label'].to(device)
            
            logits, _, loss_dict = model(
                text_global=text_global,
                text_local_list=text_local,
                img_global=img_global,
                img_local_list=img_local,
                labels=labels,
                training=False
            )
            
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            targets.extend(labels.cpu().numpy())
            all_alphas.append(loss_dict.get('alpha_mean', 0))
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_alpha = np.mean(all_alphas)
    
    report = classification_report(targets, predictions, output_dict=True,
                                   labels=[0, 1], zero_division=0,
                                   target_names=['0', '1'])
    
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 打印验证报告（包含整体指标）
    print_validation_report(report, avg_loss)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'alpha': avg_alpha,
        'report': report
    }