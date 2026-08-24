# STATUS

- **Version:** Dual-terrain Stage 1/2 + G1 double critic + Table IX DR (not yet a 10k-iter train)
- **Agent:** Agent Dojo
- **Progress:** 58%
- **Priority:** P1

## Next 3 tasks
1. A10 is **terminated when idle** (Lambda has no pause that keeps the VM). Run `bash scripts/cloud/mark_idle.sh` then confirm `model_*.pt` on `/lambda/nfs/beamdojo/logs` before terminate; keep the `beamdojo` filesystem.
2. Next GPU session: same region, attach NFS at launch, then `bash scripts/cloud/after_relaunch.sh` (Stage 1, 1024 envs, 10k iters, `--logger wandb`). Stage 2: `LOAD_RUN=<stage1-run-folder> bash scripts/cloud/train_stage2.sh` — that folder lives under `logs/rsl_rl/beamdojo_<robot>_stage1`, not stage2.
3. After Stage 1 walks the imagined beam: Stage 2 (`train_stage2.sh`) then G1 (`--robot g1`). Copy NFS checkpoints off-box — never git-commit `.pt`.

## Live training (browser)
- **Weights & Biases** is the webpage for live metrics. Train with `WANDB_API_KEY` in gitignored `.env.lambda`. Project: `beamdojo`. URL: `https://wandb.ai/<entity>/beamdojo` (set `WANDB_ENTITY` to make the link exact). Once a run starts, `training-status.json` prefers `wandb.run.url`.
- Lambda does **not** expose Isaac Sim as a public site. RTX proof remains `play_beamdojo.py --video`.
- TensorBoard: `ssh -L 6006:localhost:6006 lambda-beamdojo` then `tensorboard --logdir /lambda/nfs/beamdojo/logs`.
- Kingdom Research Lab reads `tracking/training-status.json` (gitignored; writer in `beamdojo_runtime.write_training_status`, refreshed every 10 PPO iters with mean reward / losses / FPS). `npm run dev` polls `/live/training-status.json` every 5s and can overlay a fresh W&B summary. Example: `tracking/training-status.example.json`.

## Notes
- Official BeamDojo training code was never released. This is an Isaac Lab recreation.
- Dual-terrain: **flat PhysX plane** + task heightfield for 15×15 scan and 15-sample foothold (eq. 2, XY task-map). Stage 1 visual beam has collision off; Stage 2 beam/stones collide; timeout-only in Stage 1; fall/off-terrain in Stage 2.
- Stage 2 disables `/World/ground` collision and drops onto a world catcher at z=−0.90 (below the 0.40 m height done) so the robot cannot walk beside the beam on the importer plane.
- Double critic: `ActorCriticDouble` + `PPODoubleCritic` (w1=1.0, w2=0.25), MLP `[512, 216, 128]`, injected into rsl-rl 3.0.1. Foothold GAE uses the same timeout bootstrap as rsl-rl (`R += γV·timeout`) so Stage 1 (timeout-only) does not treat every episode end as a true terminal for critic 2.
- Table IX / appendix VI-C: payload ±2 kg, CoM ±5 cm, friction 0.4–1.0, obs noise, elevation yaw/tilt/dilate/repeat. Play configs still disable DR. Payload/CoM DR targets `spec.torso_body` (`torso_link` on H1 and G1).
- Table IX / appendix VI-C: payload ±2 kg, CoM ±5 cm, friction 0.4–1.0, obs noise, elevation yaw/tilt/dilate/repeat. Play configs still disable DR.
- G1 uses `G1_MINIMAL_CFG`, feet `.*_ankle_roll_link`, 12 lower-body actions. Do not claim paper numbers until a real G1 train finishes.
- **Do not git-commit checkpoints or weights.**
