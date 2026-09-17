# mini-swin

The domain of the Swin Transformer: a hierarchical vision transformer that
computes self-attention within local windows and shifts them between layers
to enable cross-window connections.

## Language

### Model Architecture

**Patch**:
A non-overlapping square region of the input image (e.g. 4 × 4 pixels),
projected into a token vector by the Patch Embedding layer.
_Avoid_: tile, block, cell

**Token**:
A D-dimensional vector representing one Patch after embedding.
_Avoid_: feature, embedding (when referring to the vector itself)

**Window**:
A fixed-size group of spatially adjacent Tokens (e.g. 7 × 7) within which
self-attention is computed.
_Avoid_: region, partition

**Shifted Window**:
A Window grid offset by half the Window size between consecutive Swin Blocks,
enabling cross-Window information flow.
_Avoid_: displaced window, rolling window

**Stage**:
One of the hierarchical levels of the Swin Transformer (typically four).
Each Stage halves spatial resolution and doubles channel depth via Patch Merging.
_Avoid_: level, layer (too overloaded with neural-network layer)

**Patch Merging**:
The down-sampling operation between Stages: concatenates 2 × 2 neighbouring
Tokens and projects them to 2C channels.
_Avoid_: downsampling layer, pooling

**Swin Block**:
A transformer block consisting of Window-based Multi-Head Self-Attention
(regular or shifted), Layer Norm, and an MLP with GELU activation.
_Avoid_: transformer layer, encoder block

**Variant**:
A named size configuration of the Swin Transformer — Tiny (T), Small (S),
Base (B), or Large (L) — differing in embed dim, depths, and head counts.
_Avoid_: model size, configuration (when referring to the named preset)

### Training

**Checkpoint**:
A serialised snapshot of model weights (`.pth` file) saved during or after
training.
_Avoid_: saved model, weights file, snapshot

**Scheduler**:
A learning-rate schedule applied during training (e.g. cosine annealing with
linear warm-up).
_Avoid_: LR policy, learning rate strategy

### Serving

**Prediction**:
The model's output for a single image: a ranked list of (class label,
confidence) tuples, typically Top-5.
_Avoid_: inference result, output, response (too generic)

**Model Manager**:
A singleton service that loads and caches the active Checkpoint and exposes
an `infer(image) → Prediction` interface. The sole entry point into the
model module from the rest of the backend.
_Avoid_: model loader, model service, inference engine
