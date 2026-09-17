# Handoff — Phase 1: Scaffold

## What was done

Phase 1 (Scaffold) is complete. The full directory and file skeleton for
`mini-swin` has been created with:

- **67 files** across `backend/`, `frontend/`, `configs/`, `docs/`, and root
- All Python modules are `# TODO` stubs
- All TypeScript files are `// TODO` stubs
- Config files (`.toml`, `.yaml`, `package.json`, `.env.example`) contain
  skeleton keys with placeholder values — no logic

### Key artifacts created

| File | Purpose |
|---|---|
| `CONTEXT.md` | Domain glossary (domain-modeling skill format) |
| `docs/roadmap.md` | Wayfinder phase map — 6 phases, Phase 1 `[done]` |
| `docs/architecture.md` | Module map with seam annotations |
| `docs/adr/0001-tech-stack.md` | ADR: FastAPI + Vite + from-scratch Swin |
| `docs/spec-phase2.md` | Spec for next phase (Core Model) |
| `.gitignore` | Python + Node + checkpoint ignores |
| `docker-compose.yml` | Two services: backend (8000) + frontend (5173) |

### Seam design decisions (codebase-design skill)

- `model/` is a deep module. Its **sole external interface** is
  `model_manager.py` — nothing outside `model/` should import
  `swin_transformer.py` or `config.py` directly.
- `api/` exposes its interface through `routers/`. The `dependencies.py` file
  wires the Model Manager into FastAPI's DI.
- `training/` is a self-contained module. Its interface is `trainer.py`.
- `data/` exposes its interface through `dataloader.py`.

## What is next

**Phase 2 — Core Model** (see `docs/spec-phase2.md` for the full spec):

1. Implement `SwinConfig` dataclass with Variant class methods
2. Build `PatchEmbed`, `WindowAttention`, `SwinBlock`, `PatchMerging`,
   `SwinStage`, and `SwinTransformer` as `nn.Module` subclasses
3. Implement `ModelManager` singleton
4. Write unit tests for shape correctness

## Suggested skills

- `codebase-design` — maintain seam discipline as modules fill in
- `domain-modeling` — keep `CONTEXT.md` current as new terms emerge
- `to-spec` — write Phase 3 spec after Phase 2 lands

## References

- [CONTEXT.md](../CONTEXT.md)
- [roadmap.md](roadmap.md)
- [spec-phase2.md](spec-phase2.md)
- [architecture.md](architecture.md)
- [ADR-0001](adr/0001-tech-stack.md)
