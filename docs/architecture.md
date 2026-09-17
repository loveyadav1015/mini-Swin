# Architecture Overview — mini-swin

## Summary

mini-swin is a from-scratch implementation of the
[Swin Transformer](https://arxiv.org/abs/2103.14030) for image classification,
served by a **FastAPI** backend and consumed by a **React + Vite** frontend.

## Key Architectural Decisions

| # | Decision | Record |
|---|----------|--------|
| 0001 | FastAPI + Vite tech stack | [ADR-0001](adr/0001-tech-stack.md) |

## Module Map

```
backend/
  model/        ← Pure PyTorch. Interface: model_manager.py
  training/     ← Orchestrates training. Interface: trainer.py
  data/         ← Dataset + transforms. Interface: dataloader.py
  evaluation/   ← Metrics & eval scripts. Interface: evaluate.py
  api/          ← FastAPI app. Interface: routers/*

frontend/
  src/
    api/        ← HTTP client to backend
    pages/      ← Route-level components
    components/ ← Reusable UI pieces
    hooks/      ← Shared stateful logic
    types/      ← TypeScript type definitions
```

> See `CONTEXT.md` at the repo root for the domain glossary.
