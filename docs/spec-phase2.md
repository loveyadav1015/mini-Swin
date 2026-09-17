# Spec — Phase 2: Core Model

> This spec describes what Phase 2 will implement so the next agent session has
> a clear contract.

---

## Goal

Implement the complete Swin Transformer architecture in
`backend/model/swin_transformer.py` and wire it up through
`backend/model/config.py` and `backend/model/model_manager.py`.

## Deliverables

### 1. `SwinConfig` dataclass (`backend/model/config.py`)

Define a frozen dataclass with fields for:

| Field | Type | Example (Tiny) |
|---|---|---|
| `image_size` | `int` | `224` |
| `patch_size` | `int` | `4` |
| `in_channels` | `int` | `3` |
| `num_classes` | `int` | `1000` |
| `embed_dim` | `int` | `96` |
| `depths` | `tuple[int, …]` | `(2, 2, 6, 2)` |
| `num_heads` | `tuple[int, …]` | `(3, 6, 12, 24)` |
| `window_size` | `int` | `7` |
| `mlp_ratio` | `float` | `4.0` |
| `drop_rate` | `float` | `0.0` |
| `attn_drop_rate` | `float` | `0.0` |
| `drop_path_rate` | `float` | `0.1` |

Provide class methods `SwinConfig.tiny()`, `.small()`, `.base()`, `.large()`
returning canonical configs.

### 2. Architecture modules (`backend/model/swin_transformer.py`)

Implement as `nn.Module` subclasses:

| Class | Responsibility |
|---|---|
| `PatchEmbed` | Partition image into patches, project to `embed_dim` via Conv2d. |
| `PatchMerging` | Concatenate 2 × 2 token groups → linear projection to 2C. |
| `WindowAttention` | Multi-head self-attention within a single window, including relative positional bias. |
| `SwinBlock` | LN → W-MSA (or SW-MSA) → residual → LN → MLP → residual. Accept a `shift_size` parameter to toggle shifted windows. |
| `SwinStage` | Stack of `depth` SwinBlocks alternating between `shift_size=0` and `shift_size=window_size//2`. Followed by optional PatchMerging. |
| `SwinTransformer` | Full model: PatchEmbed → 4 × SwinStage → global avg pool → Linear head → logits. |

### 3. Model Manager (`backend/model/model_manager.py`)

- Singleton that holds one `SwinTransformer` instance.
- `load(config, checkpoint_path)` — build model, load weights.
- `infer(image_tensor) → Prediction` — run forward pass, return top-k.
- Thread-safe (use `threading.Lock`).

## Verification

- Unit test: instantiate each Variant and assert output shape `(B, num_classes)`.
- Unit test: verify `PatchEmbed` output shape `(B, H/4 * W/4, C)`.
- Unit test: verify `WindowAttention` handles both regular and shifted windows.
- All tests go in `backend/tests/test_model.py` (new file).

## Non-Goals for Phase 2

- No training logic.
- No API wiring.
- No pre-trained weight loading (only random init + checkpoint structure).
