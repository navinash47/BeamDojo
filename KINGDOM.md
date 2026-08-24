# BeamDojo ↔ Kingdom

This repo is a **Kingdom research project** (`beamdojo`), not a cash venture.

## What agents must do

1. After a real experiment (smoke, train, eval video): update `STATUS.md`, append `tracking/expenses.jsonl`, drop a small mp4 in `proofs/`, then from Kingdom run `npm run sync`.
2. **Never commit** `.pt` / `.pth` / wandb / `.env.lambda` / `*.pem`. If you train, tell Avinash the checkpoint path on NFS (`/lambda/nfs/beamdojo/logs/...`) so he can copy it as insurance.
3. Push a git commit on this repo after a major chunk (stack, smoke, architecture) — not after every log line.
4. GPU only: CUDA on an RT-core card (A10). No Mac/CPU/fal as the simulator.

## Proof videos

| File | Experiment |
|------|------------|
| `proofs/stage1-smoke-gpu.mp4` | Stage 1 H1 · 64 envs · RTX 1280×720 from `model_4.pt` |

## Expenses

JSONL rows with `actual_usd`. Kingdom expenses ledger picks them up on sync.
