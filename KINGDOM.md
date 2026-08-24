# BeamDojo ↔ Kingdom

This repo is a **Kingdom research project** (`beamdojo`), not a cash venture.

## What agents must do

1. After a real experiment (smoke, train, eval video): update `STATUS.md`, append `tracking/expenses.jsonl`, drop a small mp4 in `proofs/`, then from Kingdom run `npm run sync`.
2. **Never commit** `.pt` / `.pth` / wandb / `.env.lambda` / `*.pem`. If you train, tell Avinash the checkpoint path on NFS (`/lambda/nfs/beamdojo/logs/...`) so he can copy it as insurance.
3. Push a git commit on this repo after a major chunk (stack, smoke, architecture) — not after every log line.
4. GPU only: CUDA on an RT-core card (A10). No Mac/CPU/fal as the simulator.

## Live training (webpage)

Lambda does not serve Isaac Sim on the public internet. Watch a run from the Mac via:

1. **Weights & Biases** (preferred): `WANDB_API_KEY` in `.env.lambda`, train with `--logger wandb --log_project_name beamdojo`. Open `https://wandb.ai/<entity>/beamdojo`.
2. **TensorBoard tunnel:** `ssh -L 6006:localhost:6006 lambda-beamdojo` then `tensorboard --logdir /lambda/nfs/beamdojo/logs --bind_all`.
3. **Kingdom Research Lab** (`/?tab=research`): syncs `tracking/training-status.json` (gitignored; see the example file) plus proof mp4s. `npm run dev` also polls `/live/training-status.json` every 5s from the BeamDojo checkout (and a fresh W&B heartbeat if `WANDB_API_KEY` is present). The live card shows mean reward / losses from that snapshot; full curves stay on W&B.

`scripts/cloud/train_stage1.sh` writes that status JSON (and refreshes it every 10 PPO iters) and prints the W&B URL.

## Proof videos

| File | Experiment |
|------|------------|
| `proofs/stage1-smoke-gpu.mp4` | Stage 1 H1 · 64 envs · RTX 1280×720 from `model_4.pt` |

## Expenses

JSONL rows with `actual_usd`. Kingdom expenses ledger picks them up on sync.
