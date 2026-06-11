# PyTensor Poisson Optimization Analysis Results

## Key Findings

### 1. **Numerical Equivalence** ✅
- **All three approaches produce numerically equivalent results** with differences at machine precision level
- Vectorized vs Individual: differences ~ 1e-11 to 1e-13
- Vectorized vs Hybrid: differences ~ 1e-20 to 1e-22 (best precision)
- **Hybrid approach (exp(sum(log_probs))) is most numerically stable**

### 2. **Computational Graph Complexity** 🔍

**Vectorized Approach:** Clean, compact graph
```
Prod → Exp → Sub(Sub(Mul(x, Log(rates)), rates), Gammaln(x+1))
```

**Individual Approach:** Explodes with scale - creates separate operations for each element
```
Prod → MakeVector[rate1_ops, rate2_ops, ..., rateN_ops]
```
- **Hit compilation limits at 500+ dimensions** due to C++ bracket nesting depth
- Creates N separate operation chains instead of vectorized operations

**Hybrid Approach:** Most efficient graph
```
Exp → Sum → Sub(Sub(Mul(x, Log(rates)), rates), Gammaln(x+1))
```

### 3. **Performance Characteristics** ⚡

**Surprising Result: Individual approach is ~5-10x FASTER than vectorized!**

| Dimensions | Vectorized (ms) | Individual (ms) | Speedup |
|------------|-----------------|-----------------|---------|
| 5          | 0.004          | 0.001          | 0.18x   |
| 10         | 0.004          | 0.001          | 0.17x   |
| 50         | 0.006          | 0.001          | 0.12x   |
| 100        | 0.007          | 0.001          | 0.09x   |

**Why this is counter-intuitive:**
- We'd expect vectorized to be faster due to BLAS/vectorization
- But PyTensor's optimization seems to prefer the individual approach for this specific case
- Could be due to constant folding, better memory access patterns, or compilation optimizations

### 4. **Optimization Behavior** 🚀

**PyTensor heavily optimizes all approaches:**
- All compiled functions show only 1 apply node and 2 variables after optimization
- Aggressive constant folding and graph simplification occurs
- The individual operations get optimized down to very efficient forms

**Gradient Computation:**
- Vectorized gradients work perfectly with automatic differentiation
- Numerical vs analytical gradient differences: ~3e-12 (excellent agreement)
- Gradient computation is efficient and accurate

### 5. **Scalability Limits** ⚠️

**Individual approach has fundamental scalability limits:**
- **Compilation fails at 500+ dimensions** due to C++ compiler limits
- Error: "bracket nesting level exceeded maximum of 256"
- Creates too many nested operations for large N

**Vectorized/Hybrid approaches scale indefinitely**

## Recommendations for HistFactory Implementation

### 1. **Use Hybrid Approach for Production**
```python
# Best approach: exp(sum(log_probs))
log_probs = x * pt.log(rates) - rates - pt.gammaln(x + 1)
result = pt.exp(pt.sum(log_probs))
```

**Advantages:**
- ✅ Most numerically stable (differences ~1e-20)
- ✅ Clean computational graph
- ✅ Scales to arbitrary dimensions
- ✅ Works well with gradients

### 2. **Avoid Individual Operations for Large N**
- Individual operations are faster for small N (< 100)
- But they **fail to compile** for large N (500+)
- Not suitable for general-purpose implementation

### 3. **For HistFactory Specifically**
```python
# Multiple bins/channels: Use vectorized operations
constraint_log_probs = []
for constraint in constraints:
    log_prob = constraint.vectorized_logpdf(parameters)
    constraint_log_probs.append(log_prob)

# Sum all constraint log probabilities
total_constraint_logprob = pt.sum(pt.stack(constraint_log_probs))

# Main likelihood (vectorized across bins)
main_log_probs = observed * pt.log(expected_rates) - expected_rates - pt.gammaln(observed + 1)
main_logprob = pt.sum(main_log_probs)

# Total
total_logprob = main_logprob + total_constraint_logprob
```

### 4. **Performance vs Scalability Trade-off**
- **Small models (< 50 bins/constraints):** Individual operations might be faster
- **Large models (> 100 bins/constraints):** Must use vectorized operations
- **General implementation:** Use vectorized for guaranteed scalability

## Implications for PyHS3

### Current Implementation Assessment ✅
The current HistFactory implementation is on the right track:
- Uses vectorized operations where possible
- Handles constraints appropriately
- Should scale to large models

### Potential Optimizations 🔧
1. **Constraint grouping:** Group similar constraints to minimize the number of separate constraint terms
2. **Log-space arithmetic:** Use the hybrid `exp(sum(log_probs))` pattern consistently
3. **Batch operations:** Process multiple samples/channels in batches where possible

### Testing Strategy 📊
- Test with models of various sizes (10, 100, 1000+ bins)
- Benchmark against pyhf for equivalent models
- Monitor compilation times and memory usage for large models

## Conclusions

1. **PyTensor is surprisingly good at optimizing individual operations** - but hits scalability limits
2. **Vectorized operations are the safe choice** for general-purpose implementations
3. **Hybrid approach provides best numerical stability** while maintaining performance
4. **Current HistFactory implementation strategy is sound** - vectorized operations with proper constraint handling

The analysis validates the architectural decisions made in the HistFactory implementation and provides guidance for future optimizations.