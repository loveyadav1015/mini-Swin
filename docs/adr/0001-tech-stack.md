# FastAPI + Vite + from-scratch Swin Transformer

We need a backend that serves a PyTorch Swin Transformer for inference and
manages training runs, plus a frontend for uploading images and viewing
Predictions. We chose **FastAPI** for async-native Python serving with
first-class Pydantic schemas, **Vite + React (TypeScript)** for sub-second HMR
and compile-time type safety, and a **from-scratch PyTorch** Swin
implementation (over ViT) for linear-complexity windowed attention and
hierarchical feature maps.
