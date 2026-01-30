# Focal-InfoNCE: Technical Deep Dive

## 📖 Giới thiệu

Focal-InfoNCE là một cải tiến của InfoNCE loss dựa trên ý tưởng từ Focal Loss, được đề xuất để cải thiện chất lượng sentence embeddings trong unsupervised contrastive learning.

**Paper**: [Improving Unsupervised Sentence Embeddings with Focal-InfoNCE](https://openreview.net/pdf?id=j48JCRagwR)

---

## 🎯 Vấn đề với InfoNCE

### InfoNCE Loss (MultipleNegativesRankingLoss)

Công thức cơ bản:
```
L = -log( exp(s_pos / tau) / (exp(s_pos / tau) + Σ exp(s_neg / tau)) )
```

Trong đó:
- `s_pos`: similarity của positive pair
- `s_neg`: similarity của negative pairs
- `tau`: temperature parameter

### Vấn đề chính

1. **Không phân biệt độ khó**: Tất cả negative samples được đối xử như nhau
   - Hard negatives (similarity cao) bị lấn át bởi easy negatives
   - Gradient chủ yếu đến từ easy negatives → khai thác chưa tốt

2. **Dropout noise**: Positive pairs có similarity thấp bị phạt mạnh
   - Similarity thấp có thể do dropout ngẫu nhiên, không phải do model kém
   - Model bị penalty quá mức cho noise

3. **Embedding quality**: 
   - Alignment chưa tối ưu (positive pairs chưa đủ gần)
   - Uniformity kém (embeddings có anisotropy problem)

---

## 💡 Giải pháp: Focal-InfoNCE

### Ý tưởng cốt lõi

**Re-weighting loss theo độ khó của samples**

- Hard samples → trọng số cao → gradient lớn
- Easy samples → trọng số thấp → gradient nhỏ

Tương tự Focal Loss cho object detection, nhưng áp dụng cho contrastive learning.

---

## 🔬 Kỹ thuật chi tiết

### 1. Hard Negative Mining

**Công thức re-weighting:**
```
w_neg(s) = exp(s * (s + m) / tau)
```

**Phân tích:**
- `s`: cosine similarity của negative pair (0 đến 1)
- `m`: margin parameter (thường 0.25)
- `tau`: temperature (thường 0.05)

**Ví dụ cụ thể** (với m=0.25, tau=0.05):

| Similarity (s) | Weight w_neg | Gradient multiplier |
|---------------|--------------|---------------------|
| 0.1 (easy)    | exp(0.7)     | 2.0x                |
| 0.5 (medium)  | exp(7.5)     | 1808x               |
| 0.9 (hard)    | exp(23.0)    | 9.7 billion x       |

→ **Hard negatives được focus cực mạnh!**

**Hiệu ứng:**
- Model tập trung học phân biệt các negative pairs khó
- Cải thiện uniformity: embeddings phân bố đều hơn
- Giảm anisotropy problem

### 2. Dropout Noise Aware (Positive Re-weighting)

**Công thức:**
```
w_pos(s) = exp(-gamma * (1 - s))
```

**Phân tích:**
- `s`: cosine similarity của positive pair
- `gamma`: focal parameter (thường 1.0)

**Ví dụ cụ thể** (với gamma=1.0):

| Similarity (s) | 1 - s | Weight w_pos | Effect |
|---------------|-------|--------------|--------|
| 0.95 (good)   | 0.05  | exp(-0.05) = 0.95 | Penalty gần như đầy đủ |
| 0.70 (medium) | 0.30  | exp(-0.30) = 0.74 | Giảm penalty 26% |
| 0.50 (low)    | 0.50  | exp(-0.50) = 0.61 | Giảm penalty 39% |

→ **Positive có similarity thấp (có thể do dropout noise) bị phạt nhẹ hơn**

**Hiệu ứng:**
- Tránh over-penalize do dropout randomness
- Cải thiện alignment: positive pairs gần nhau hơn
- Training ổn định hơn

### 3. Toàn bộ Focal-InfoNCE Loss

**Pseudo-code:**

```python
# 1. Compute similarity matrix
sim_matrix = cosine_similarity(anchors, positives)  # (B, B)

# 2. Extract positive and negative similarities
pos_sim = diagonal(sim_matrix)  # (B,)
neg_sim = off_diagonal(sim_matrix)  # (B, B-1)

# 3. Positive re-weighting
pos_weight = exp(-gamma_pos * (1 - pos_sim))
pos_logits = (pos_sim / tau) * pos_weight

# 4. Negative re-weighting
neg_weight = exp(neg_sim * (neg_sim + margin) / tau)
neg_logits = (neg_sim / tau) * neg_weight

# 5. Combine and compute cross-entropy
logits = [pos_logits, neg_logits]  # (B, B)
labels = [0, 1, 2, ..., B-1]  # Positive is at diagonal
loss = CrossEntropyLoss(logits, labels)
```

---

## 📊 Hiệu quả theo Paper

### Kết quả trên STS tasks

| Model | InfoNCE (baseline) | Focal-InfoNCE | Improvement |
|-------|-------------------|---------------|-------------|
| BERT-base | 76.25% | 77.89% | **+1.64%** |
| BERT-large | 78.41% | 79.23% | **+0.82%** |
| RoBERTa-base | 76.57% | 78.08% | **+1.51%** |
| RoBERTa-large | 78.49% | 79.24% | **+0.75%** |

### Cải thiện Alignment & Uniformity

| Metric | InfoNCE | Focal-InfoNCE | Better? |
|--------|---------|---------------|---------|
| Alignment | -0.15 | **-0.18** | ✓ (Lower is better) |
| Uniformity | -2.10 | **-2.25** | ✓ (Lower is better) |

→ Focal-InfoNCE cải thiện cả 2 metrics quan trọng

---

## ⚙️ Hyperparameters

### Temperature (tau)

**Default**: 0.05

**Hiệu ứng:**
- **Thấp hơn (0.03)**: 
  - Gradient steeper
  - Tập trung mạnh vào hard samples
  - Risk: Training instability
  
- **Cao hơn (0.07)**:
  - Gradient smoother
  - Ổn định hơn
  - Risk: Không khai thác đủ hard samples

**Recommendation**: Start with 0.05, tune trong range [0.03, 0.07]

### Margin (m)

**Default**: 0.25

**Hiệu ứng:**
- **Lớn hơn (0.5)**:
  - Hard negative mining mạnh hơn
  - Focus nhiều vào các negative khó
  - Risk: Overfocus, ignore easy negatives hoàn toàn
  
- **Nhỏ hơn (0.1)**:
  - Hard negative mining nhẹ hơn
  - Gần với InfoNCE vanilla
  - Risk: Không khai thác đủ hard negatives

**Recommendation**: 0.25 là optimal cho hầu hết tasks

### Gamma (positive)

**Default**: 1.0

**Hiệu ứng:**
- **Lớn hơn (2.0)**:
  - Re-weighting mạnh hơn
  - Giảm penalty nhiều cho low-similarity positives
  - Risk: Quá tolerant với noisy positives
  
- **Nhỏ hơn (0.5)**:
  - Re-weighting nhẹ hơn
  - Gần với InfoNCE
  
**Recommendation**: 1.0 hoạt động tốt, có thể thử 0.5-2.0

### Gamma (negative)

**Default**: 1.0

**Hiệu ứng**: Tương tự gamma_pos, nhưng cho negative pairs

**Note**: Thường set bằng gamma_pos để đơn giản

---

## 🎓 So sánh với các kỹ thuật khác

### vs. Hard Negative Mining

| Aspect | External Mining | Focal-InfoNCE |
|--------|----------------|---------------|
| Implementation | Cần external system | Built-in loss |
| Data requirement | Thêm hard negatives | Chỉ cần in-batch |
| Computational cost | Cao (mining) | Thấp (re-weighting) |
| Flexibility | Cần design mining | Automatic |

→ **Focal-InfoNCE đơn giản hơn nhiều**

### vs. Supervised SimCSE

| Aspect | Supervised SimCSE | Focal-InfoNCE |
|--------|------------------|---------------|
| Data requirement | Labeled data (NLI) | Unlabeled data |
| Performance | Cao hơn | Thấp hơn một chút |
| Applicability | Limited domains | General |

→ **Focal-InfoNCE linh hoạt hơn (unsupervised)**

### vs. Augmentation-based

| Aspect | Data Aug | Focal-InfoNCE |
|--------|---------|---------------|
| Positive pairs | Aug (back-translation, etc.) | Dropout |
| Complexity | Cao | Thấp |
| Domain-specific | Yes (need design) | No |

→ **Focal-InfoNCE đơn giản, domain-agnostic**

---

## 🔧 Implementation Tips

### 1. Batch Size

**Critical**: Batch size càng lớn càng tốt

- In-batch negatives = (batch_size - 1)
- Batch nhỏ → ít hard negatives → hiệu quả giảm

**Recommendation**: 
- Minimum: 64
- Optimal: 128-256
- Maximum: Càng lớn càng tốt (memory allows)

### 2. Gradient Clipping

**Optional but recommended**

```python
training_args = SentenceTransformerTrainingArguments(
    max_grad_norm=1.0,  # Clip gradients
    ...
)
```

Hard negative re-weighting có thể tạo gradients lớn → clipping giúp stability

### 3. Learning Rate

**Keep same as InfoNCE**: 2e-5 (for BERT-like models)

Focal-InfoNCE không cần learning rate khác

### 4. Warmup

**Important**: Vẫn cần warmup

- Warmup ratio: 10% steps
- Giúp model không bị shock bởi hard negative weights ban đầu

### 5. Evaluation

**Monitor multiple metrics**:
- Recall@K (retrieval performance)
- Alignment & Uniformity (embedding quality)
- Similarity distributions (positive vs negative)

### 6. Ablation Studies

**Test variants** để hiểu impact:

1. **No positive re-weighting**: `gamma_pos = 0`
2. **No negative re-weighting**: `gamma_neg = 0, margin = 0`
3. **Only hard negative mining**: Keep `margin`, `gamma_neg`, set `gamma_pos = 0`
4. **Only dropout-aware**: Keep `gamma_pos`, set `gamma_neg = 0, margin = 0`

---

## 🐛 Debugging Guide

### Loss không giảm

**Possible causes:**
1. Learning rate quá thấp → tăng lên 2e-5
2. Margin quá lớn → giảm xuống 0.1-0.25
3. Batch size quá nhỏ → tăng lên ≥64

### Loss giảm quá nhanh (overfitting)

**Possible causes:**
1. Learning rate quá cao → giảm xuống 1e-5
2. Margin quá nhỏ → tăng lên 0.25-0.5
3. No regularization → thêm dropout, weight decay

### Training instability

**Possible causes:**
1. Temperature quá thấp → tăng lên 0.05-0.07
2. No gradient clipping → thêm `max_grad_norm=1.0`
3. Gamma quá lớn → giảm về 1.0

### Performance không cải thiện vs InfoNCE

**Possible causes:**
1. Batch size quá nhỏ → tăng lên
2. Hyperparameters chưa tune → chạy grid search
3. Task không phù hợp → Focal-InfoNCE tốt nhất với hard negative scenarios

---

## 📚 References

1. **Original Paper**: [Focal-InfoNCE](https://openreview.net/pdf?id=j48JCRagwR)
2. **SimCSE**: [Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821)
3. **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
4. **Alignment & Uniformity**: [Understanding Contrastive Representation Learning](https://arxiv.org/abs/2005.10242)

---

## 🤔 FAQ

### Q1: Focal-InfoNCE vs InfoNCE - khi nào nên dùng?

**Dùng Focal-InfoNCE khi:**
- Có nhiều hard negatives trong data
- Cần embedding space uniformity cao
- Có GPU đủ lớn cho batch size ≥64

**Dùng InfoNCE khi:**
- Data đơn giản, ít hard negatives
- Batch size nhỏ (<32)
- Cần training nhanh, ít tune hyperparams

### Q2: Có cần data augmentation không?

**Không cần** - Focal-InfoNCE chỉ dùng dropout như SimCSE

Nhưng **có thể thêm** augmentation để cải thiện thêm (optional)

### Q3: Supervised hay unsupervised?

**Paper gốc**: Unsupervised (dropout-based)

**Có thể mở rộng**: Sang supervised với labeled pairs

### Q4: Computational cost tăng bao nhiêu?

**Negligible** - chỉ thêm vài operations:
- exp() cho re-weighting
- Tổng cost tăng <5%

### Q5: Có work với multilingual không?

**Yes** - Architecture-agnostic, có thể dùng với:
- multilingual BERT
- XLM-R
- mBERT

---

**End of Technical Deep Dive**
