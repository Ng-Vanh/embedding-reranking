
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import os
import shutil
from glob import glob





class DOMRerankingDataset(Dataset):
    """
    Input format:
    {
        'question': "...",
        'nodes': [node1_text, node2_text, ..., nodeN_text],
        'gold_idx': idx gold node
    }
    
    Output format :
    - (question, node) 
    - Label: 1 nếu là gold node, 0 nếu không
    """
    
    def __init__(self, data_or_path, tokenizer, max_length: int = 512):
        # Support both file path and data list
        if isinstance(data_or_path, str):
            with open(data_or_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = data_or_path
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Chuyển từ listwise -> pointwise
        self.samples = []
        self.create_pointwise_samples()
        
    def create_pointwise_samples(self):
        """
        query:
        - 1 positive sample (gold node)
        - N-1 negative samples (non-gold nodes)
        """
        for item in self.data:
            question = item['question']
            nodes = item['nodes']
            gold_idx = item['gold_idx']
            
            for node_idx, node_text in enumerate(nodes):
                # Label: 1 nếu là gold node, 0 nếu không
                label = 1 if node_idx == gold_idx else 0
                
                self.samples.append({
                    'question': question,
                    'node': node_text,
                    'label': label,
                    'query_id': len(self.samples) // len(nodes),  
                    'node_idx': node_idx
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize: [CLS] question [SEP] node [SEP]
        encoding = self.tokenizer(
            sample['question'],
            sample['node'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(sample['label'], dtype=torch.float),
            'query_id': sample['query_id'],
            'node_idx': sample['node_idx']
        }



# 2. MODEL: DeBERTa Cross-Encoder


class DOMReranker(nn.Module):
    """
    DeBERTa Cross-Encoder cho reranking
    
    Architecture:
    - DeBERTa encoder
    - [CLS] -> Linear -> Sigmoid -> relevance score = [0, 1]
    """
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        super().__init__()
        
        # Load pretrained DeBERTa
        # num_labels=1: regression (output 1 score)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1
        )
        
    def forward(self, input_ids, attention_mask):
        """
        Output: relevance score = [0, 1]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # logits shape: (batch_size, 1)
        logits = outputs.logits.squeeze(-1)  # (batch_size,)
        
        # Sigmoid để có xác suất
        scores = torch.sigmoid(logits)
        
        return scores



# 3. LOSS FUNCTION: Binary Cross-Entropy


class RerankingLoss(nn.Module):
    """
    Binary Cross-Entropy Loss 
    """
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, scores, labels):
        """
        Args:
            scores: (batch_size,) predicted scores = [0, 1]
            labels: (batch_size,) ground truth = {0, 1}
        Returns:
            loss: scalar
        """
        return self.bce(scores, labels)



# 4. EVALUATION: Listwise Metrics


def evaluate_listwise(model, dataloader, device):
    """
    Metrics:
    - ACC@1: gold node ở top-1 hay không
    - MRR@10: Mean Reciprocal Rank
    - Recall@K: gold node trong top-K hay không
    """
    model.eval()
    
    # Group predictions by query_id
    query_predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()
            query_ids = batch['query_id'].cpu().numpy()
            node_indices = batch['node_idx'].cpu().numpy()
            
            # Predict scores
            scores = model(input_ids, attention_mask).cpu().numpy()
            
            # Group by query
            for qid, score, label, node_idx in zip(query_ids, scores, labels, node_indices):
                if qid not in query_predictions:
                    query_predictions[qid] = []
                query_predictions[qid].append({
                    'score': score,
                    'label': label,
                    'node_idx': node_idx
                })
    
    # Compute metrics mỗi query
    acc_at_1 = []
    mrr_at_10 = []
    recall_at_3 = []
    recall_at_5 = []
    recall_at_10 = []
    
    for qid, predictions in query_predictions.items():
        # Sort score 
        sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        # find gold node 
        gold_rank = None
        for rank, pred in enumerate(sorted_preds):
            if pred['label'] == 1:
                gold_rank = rank
                break
        
        if gold_rank is not None:
            # ACC@1
            acc_at_1.append(1.0 if gold_rank == 0 else 0.0)
            
            # MRR@10
            if gold_rank < 10:
                mrr_at_10.append(1.0 / (gold_rank + 1))
            else:
                mrr_at_10.append(0.0)
            
            # Recall@K
            recall_at_3.append(1.0 if gold_rank < 3 else 0.0)
            recall_at_5.append(1.0 if gold_rank < 5 else 0.0)
            recall_at_10.append(1.0 if gold_rank < 10 else 0.0)
    
    metrics = {
        'ACC@1': np.mean(acc_at_1),
        'MRR@10': np.mean(mrr_at_10),
        'Recall@3': np.mean(recall_at_3),
        'Recall@5': np.mean(recall_at_5),
        'Recall@10': np.mean(recall_at_10),
    }
    
    return metrics



# 5. TRAINING LOOP


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, writer=None, epoch=0):
    model.train()
    total_loss = 0
    step = epoch * len(dataloader)
    
    # Track score distribution
    pos_scores = []
    neg_scores = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        scores = model(input_ids, attention_mask)
        loss = loss_fn(scores, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Track scores for monitoring imbalance
        with torch.no_grad():
            pos_scores.extend(scores[labels==1].cpu().numpy())
            neg_scores.extend(scores[labels==0].cpu().numpy())
        
        # Log to TensorBoard every 100 steps
        if writer is not None and batch_idx % 100 == 0:
            global_step = step + batch_idx
            writer.add_scalar('Train/batch_loss', loss.item(), global_step)
            writer.add_scalar('Train/learning_rate', scheduler.get_last_lr()[0], global_step)
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute score statistics
    stats = {}
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        stats['pos_mean'] = np.mean(pos_scores)
        stats['pos_std'] = np.std(pos_scores)
        stats['neg_mean'] = np.mean(neg_scores)
        stats['neg_std'] = np.std(neg_scores)
        stats['separation'] = stats['pos_mean'] - stats['neg_mean']
    
    return avg_loss, stats



# 6. CHECKPOINT MANAGEMENT


def cleanup_checkpoints(output_dir, keep_last_n=2):
    checkpoint_dirs = glob(os.path.join(output_dir, 'checkpoint-epoch-*'))
    if len(checkpoint_dirs) <= keep_last_n:
        return
    
    # Sort by epoch number
    checkpoint_dirs = sorted(
        checkpoint_dirs,
        key=lambda x: int(x.split('-epoch-')[-1])
    )
    
    # Delete old checkpoints
    for old_checkpoint in checkpoint_dirs[:-keep_last_n]:
        if os.path.exists(old_checkpoint):
            shutil.rmtree(old_checkpoint)
            print(f"  Deleted old checkpoint: {old_checkpoint}")



# 7. MAIN TRAINING SCRIPT


def main():

    config = {
        'model_name': 'microsoft/deberta-v3-base',
        'train_data': '/mnt/disk2/anhnv/rr/stage2/data/stage2_train_top50_43526.json',
        'max_length': 512,
        'batch_size': 16,
        'gradient_accumulation_steps': 2,  
        'learning_rate': 2e-5,
        'warmup_ratio': 0.1,
        'num_epochs': 3,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'output_dir': './checkpoints/deberta_reranker-pointwise',
        'log_dir': './runs/deberta_reranker-pointwise',  
        'keep_last_n_checkpoints': 2,  
        'train_eval_split': 0.9,  
        'random_seed': 42,
    }
    
    # Device setup - Multi-GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Number of GPUs: {n_gpu}")
    
    # TensorBoard writer
    os.makedirs(config['log_dir'], exist_ok=True)
    writer = SummaryWriter(config['log_dir'])
    print(f"TensorBoard logs: {config['log_dir']}")
    print(f"Run: tensorboard --logdir={config['log_dir']}")
    
    # Tokenizer
    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Add special tokens
    special_tokens = ["[ACTION]", "[HISTORY]", "[TAG]", "[CLASS]", "[TEXT]", "[ID]", "[HREF]", "[TITLE]"]
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"Added {num_added_tokens} special tokens: {special_tokens}")
    
    # Load and split data (90% train, 10% eval)
    print("Loading and splitting data...")
    with open(config['train_data'], 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    # all_data = all_data[:2000]
    train_data, eval_data = train_test_split(
        all_data,
        train_size=config['train_eval_split'],
        random_state=config['random_seed'],
        shuffle=True
    )
    
    print(f"Total queries: {len(all_data)}")
    print(f"Train queries: {len(train_data)} ({len(train_data)/len(all_data)*100:.1f}%)")
    print(f"Eval queries: {len(eval_data)} ({len(eval_data)/len(all_data)*100:.1f}%)")
    
    # Create datasets
    print("\nCreating train dataset...")
    train_dataset = DOMRerankingDataset(
        train_data,
        tokenizer,
        config['max_length']
    )
    
    print("Creating eval dataset...")
    eval_dataset = DOMRerankingDataset(
        eval_data,
        tokenizer,
        config['max_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=4
    )
    
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total eval samples: {len(eval_dataset)}")
    
    # Model
    print(f"\Model Info: {config['model_name']}")
    model = DOMReranker(config['model_name'])
    
    # Resize token embeddings if special tokens were added
    if num_added_tokens > 0:
        model.model.resize_token_embeddings(len(tokenizer))
        print(f"Len token size {len(tokenizer)}")
    
    # Multi-GPU support
    if n_gpu > 1:
        model.model = nn.DataParallel(model.model)
    
    model = model.to(device)
    
    # Loss
    loss_fn = RerankingLoss()
    
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=config['weight_decay']
    )
    
    # warmup + linear decay
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print(f"Training steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    
    # Training loop
    best_mrr = 0.0
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*50}")
        
        # Train
        avg_loss, train_stats = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            writer=writer, epoch=epoch
        )
        print(f"Avg training loss: {avg_loss:.4f}")
        
        # Log training stats
        if train_stats:
            print(f"  Pos score: {train_stats['pos_mean']:.3f}")
            print(f"  Neg score: {train_stats['neg_mean']:.3f}")
            print(f"  Separation: {train_stats['separation']:.3f}")
            
            if train_stats['separation'] < 0.1:
                print(" separation low")
            
            # Log to TensorBoard
            writer.add_scalar('Train/epoch_loss', avg_loss, epoch)
            writer.add_scalar('Train/pos_score_mean', train_stats['pos_mean'], epoch)
            writer.add_scalar('Train/neg_score_mean', train_stats['neg_mean'], epoch)
            writer.add_scalar('Train/score_separation', train_stats['separation'], epoch)
        
        # Evaluate
        print("\nEval: ")
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        metrics = evaluate_listwise(model, eval_loader, device)
        
        print(f"\nEval results:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
            writer.add_scalar(f'Eval/{metric_name}', value, epoch)
        
        # Save best model
        if metrics['MRR@10'] > best_mrr:
            best_mrr = metrics['MRR@10']
            save_path = os.path.join(config['output_dir'], 'best_model')
            os.makedirs(save_path, exist_ok=True)
            
            # Save model (handle DataParallel)
            model_to_save = model.model.module if hasattr(model.model, 'module') else model.model
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"\n Saved best model MRR@10: {best_mrr:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            config['output_dir'], 
            f'checkpoint-epoch-{epoch+1}'
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model (handle DataParallel)
        model_to_save = model.model.module if hasattr(model.model, 'module') else model.model
        model_to_save.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f" Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        cleanup_checkpoints(config['output_dir'], keep_last_n=config['keep_last_n_checkpoints'])
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\n{'='*50}")
    print(f"Training completed! Best MRR@10: {best_mrr:.4f}")
    print(f"TensorBoard logs saved to: {config['log_dir']}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()