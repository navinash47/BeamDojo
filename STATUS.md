# STATUS

- **Version:** Stage 1 smoke
- **Agent:** Agent Dojo
- **Progress:** 18%
- **Priority:** P1

## Next 3 tasks
1. Dual-terrain Stage 1 (flat physics + imagined beam height scan) then 1024-env CUDA train on the A10.
2. Log Lambda hours in `tracking/expenses.jsonl` whenever the GPU is running; terminate the instance when idle.
3. Stage 2 hard beam + Unitree G1 port — only after Stage 1 walks the imagined beam.

## Notes
- Official BeamDojo training code was never released. This repo is an Isaac Lab recreation (H1 Stage 1 first).
- **Do not git-commit checkpoints or weights.** Tell Avinash so he can copy them off-box as insurance.
- Kingdom Research Lab syncs this file, expenses, architecture, and proof videos.
