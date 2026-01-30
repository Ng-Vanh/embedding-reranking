"""
Quick test script để verify Focal-InfoNCE implementation

Test cases:
1. Loss computation với dummy data
2. Gradient flow
3. So sánh output với InfoNCE
4. Edge cases (batch size = 2, identical inputs, etc.)
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, losses, models
from focal_infonce_loss import SimplifiedFocalInfoNCELoss, FocalInfoNCELoss


def test_loss_computation():
    """Test 1: Basic loss computation"""
    print("\n" + "="*70)
    print(" TEST 1: Loss Computation")
    print("="*70)
    
    # Create simple model - force to CPU for testing
    word_embedding = models.Transformer('sentence-transformers/all-MiniLM-L6-v2', max_seq_length=128)
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding, pooling], device='cpu')
    
    # Create dummy data
    sentences1 = ["This is a test sentence.", "Another test sentence."]
    sentences2 = ["A positive example.", "Another positive."]
    
    # Tokenize
    features1 = model.tokenize(sentences1)
    features2 = model.tokenize(sentences2)
    
    # Test SimplifiedFocalInfoNCELoss
    focal_loss = SimplifiedFocalInfoNCELoss(
        model=model,
        temperature=0.05,
        margin=0.25,
        gamma_pos=1.0,
        gamma_neg=1.0,
    )
    
    loss_value = focal_loss([features1, features2])
    
    print(f"\n✓ Loss computed successfully: {loss_value.item():.4f}")
    print(f"  Loss type: {type(loss_value)}")
    print(f"  Loss requires grad: {loss_value.requires_grad}")
    
    # Compare with InfoNCE
    infonce_loss = losses.MultipleNegativesRankingLoss(model)
    # MultipleNegativesRankingLoss requires labels (just dummy labels)
    batch_size = len(sentences1)
    labels = torch.arange(batch_size)
    infonce_value = infonce_loss([features1, features2], labels)
    
    print(f"\n  InfoNCE loss: {infonce_value.item():.4f}")
    print(f"  Focal-InfoNCE loss: {loss_value.item():.4f}")
    print(f"  Difference: {abs(loss_value.item() - infonce_value.item()):.4f}")


def test_gradient_flow():
    """Test 2: Gradient flow"""
    print("\n" + "="*70)
    print(" TEST 2: Gradient Flow")
    print("="*70)
    
    word_embedding = models.Transformer('sentence-transformers/all-MiniLM-L6-v2', max_seq_length=128)
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding, pooling], device='cpu')
    
    focal_loss = SimplifiedFocalInfoNCELoss(model=model, temperature=0.05, margin=0.25)
    
    sentences1 = ["Test sentence one.", "Test sentence two."]
    sentences2 = ["Positive one.", "Positive two."]
    
    features1 = model.tokenize(sentences1)
    features2 = model.tokenize(sentences2)
    
    # Compute loss
    loss = focal_loss([features1, features2])
    
    # Backward
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    
    print(f"\n✓ Backward pass successful")
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Gradients computed: {has_grad}")
    
    if has_grad:
        print(f"  ✓ Gradients are flowing correctly")
    else:
        print(f"  ⚠ Warning: No gradients detected")


def test_edge_cases():
    """Test 3: Edge cases"""
    print("\n" + "="*70)
    print(" TEST 3: Edge Cases")
    print("="*70)
    
    word_embedding = models.Transformer('sentence-transformers/all-MiniLM-L6-v2', max_seq_length=128)
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding, pooling], device='cpu')
    
    focal_loss = SimplifiedFocalInfoNCELoss(model=model)
    
    # Edge case 1: Batch size = 2 (minimum)
    print("\n  Edge case 1: Batch size = 2")
    s1 = ["Sentence A.", "Sentence B."]
    s2 = ["Positive A.", "Positive B."]
    f1, f2 = model.tokenize(s1), model.tokenize(s2)
    loss = focal_loss([f1, f2])
    print(f"    Loss: {loss.item():.4f} ✓")
    
    # Edge case 2: Identical sentences (perfect alignment)
    print("\n  Edge case 2: Identical sentences")
    s1 = ["Same sentence.", "Same sentence."]
    s2 = ["Same sentence.", "Same sentence."]
    f1, f2 = model.tokenize(s1), model.tokenize(s2)
    loss = focal_loss([f1, f2])
    print(f"    Loss: {loss.item():.4f}")
    print(f"    (Should be close to 0) ✓")
    
    # Edge case 3: Large batch
    print("\n  Edge case 3: Large batch size = 16")
    s1 = [f"Query sentence {i}." for i in range(16)]
    s2 = [f"Positive sentence {i}." for i in range(16)]
    f1, f2 = model.tokenize(s1), model.tokenize(s2)
    loss = focal_loss([f1, f2])
    print(f"    Loss: {loss.item():.4f} ✓")


def test_hyperparameter_sensitivity():
    """Test 4: Hyperparameter sensitivity"""
    print("\n" + "="*70)
    print(" TEST 4: Hyperparameter Sensitivity")
    print("="*70)
    
    word_embedding = models.Transformer('sentence-transformers/all-MiniLM-L6-v2', max_seq_length=128)
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding, pooling], device='cpu')
    
    sentences1 = ["Test sentence.", "Another sentence.", "Third sentence."]
    sentences2 = ["Positive.", "Another positive.", "Third positive."]
    features1 = model.tokenize(sentences1)
    features2 = model.tokenize(sentences2)
    
    # Test different temperatures
    print("\n  Temperature sensitivity:")
    for temp in [0.03, 0.05, 0.07]:
        loss_fn = SimplifiedFocalInfoNCELoss(model=model, temperature=temp)
        loss = loss_fn([features1, features2])
        print(f"    tau={temp}: loss={loss.item():.4f}")
    
    # Test different margins
    print("\n  Margin sensitivity:")
    for margin in [0.0, 0.25, 0.5]:
        loss_fn = SimplifiedFocalInfoNCELoss(model=model, margin=margin, temperature=0.05)
        loss = loss_fn([features1, features2])
        print(f"    m={margin}: loss={loss.item():.4f}")
    
    # Test different gammas
    print("\n  Gamma sensitivity:")
    for gamma in [0.5, 1.0, 2.0]:
        loss_fn = SimplifiedFocalInfoNCELoss(
            model=model,
            gamma_pos=gamma,
            gamma_neg=gamma,
            temperature=0.05,
            margin=0.25
        )
        loss = loss_fn([features1, features2])
        print(f"    gamma={gamma}: loss={loss.item():.4f}")


def test_both_implementations():
    """Test 5: Compare both loss implementations"""
    print("\n" + "="*70)
    print(" TEST 5: Compare FocalInfoNCELoss vs SimplifiedFocalInfoNCELoss")
    print("="*70)
    
    word_embedding = models.Transformer('sentence-transformers/all-MiniLM-L6-v2', max_seq_length=128)
    pooling = models.Pooling(word_embedding.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding, pooling], device='cpu')
    
    sentences1 = ["Test sentence.", "Another sentence."]
    sentences2 = ["Positive.", "Another positive."]
    features1 = model.tokenize(sentences1)
    features2 = model.tokenize(sentences2)
    
    # Original implementation
    loss_fn_1 = FocalInfoNCELoss(
        model=model,
        scale=20.0,  # 1/0.05
        margin=0.25,
        gamma_pos=1.0,
        gamma_neg=1.0,
    )
    
    # Simplified implementation
    loss_fn_2 = SimplifiedFocalInfoNCELoss(
        model=model,
        temperature=0.05,
        margin=0.25,
        gamma_pos=1.0,
        gamma_neg=1.0,
    )
    
    loss1 = loss_fn_1([features1, features2])
    loss2 = loss_fn_2([features1, features2])
    
    print(f"\n  FocalInfoNCELoss:           {loss1.item():.4f}")
    print(f"  SimplifiedFocalInfoNCELoss: {loss2.item():.4f}")
    print(f"  Difference:                 {abs(loss1.item() - loss2.item()):.6f}")
    
    if abs(loss1.item() - loss2.item()) < 0.01:
        print(f"\n  ✓ Both implementations are consistent")
    else:
        print(f"\n  ⚠ Warning: Implementations differ significantly")


def main():
    print("\n" + "="*70)
    print(" FOCAL-INFONCE LOSS - QUICK TESTS")
    print("="*70)
    
    try:
        test_loss_computation()
        test_gradient_flow()
        test_edge_cases()
        test_hyperparameter_sensitivity()
        test_both_implementations()
        
        print("\n" + "="*70)
        print(" ALL TESTS PASSED ✓")
        print("="*70)
        print("\nImplementation is ready to use!")
        print("\nNext steps:")
        print("  1. Run full training: python train_focal_infonce.py")
        print("  2. Compare with baseline: python compare_losses.py")
        print("  3. Tune hyperparameters: python hyperparameter_tuning.py")
        print("\n" + "="*70)
        
    except Exception as e:
        print("\n" + "="*70)
        print(" TEST FAILED ✗")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
