---
name: llava-next-remote-workflow
description: Project-specific workflow for the user's llava_next repository. Use when Codex needs to improve code in D:\codex\llava_next, push changes to GitHub, pull them on myserver at /mnt/data2/wuzhengxing/llava_next, run or debug inside Docker container qwen2vl_wzx mounted at /workspace, manage tmux training sessions, inspect logs, and map remote /workspace tracebacks back to local files.
---

# LLaVA Next Remote Workflow

## Fixed Project Facts

Use these values for this project:

```text
Local project path: D:\codex\llava_next
GitHub remote: git@github.com:WuZX-cloud/llava_next.git
Branch: main
SSH host alias: myserver
Remote host project path: /mnt/data2/wuzhengxing/llava_next
Docker container: qwen2vl_wzx
Container project path: /workspace
Default tmux session: train
Default log file: train.log
```

The workflow is local-first for code changes and remote-first for GPU/container execution:

```text
D:\codex\llava_next
  -> Codex inspects and edits code locally
  -> git commit && git push
GitHub WuZX-cloud/llava_next main
  -> myserver git pull --ff-only
/mnt/data2/wuzhengxing/llava_next mounted as /workspace
  -> docker exec qwen2vl_wzx ...
```

## Local Code Improvement

Start every code task in the local clone:

```powershell
cd D:\codex\llava_next
git status --short
git rev-parse --abbrev-ref HEAD
git remote -v
```

Use `rg`, `rg --files`, and focused file reads to understand the implementation. Patch local files only after understanding the requested behavior and surrounding project style.

Run focused local checks when possible, then inspect the diff:

```powershell
git diff --stat
git diff
```

If the change must be tested remotely, commit and push:

```powershell
git add <changed-files>
git commit -m "<message>"
git push origin main
```

## Remote Git Checks

Before pulling remotely, confirm the remote worktree is clean:

```powershell
ssh myserver "cd /mnt/data2/wuzhengxing/llava_next && git status --short && git rev-parse --abbrev-ref HEAD"
```

Pull local changes from GitHub to the remote host:

```powershell
ssh myserver "cd /mnt/data2/wuzhengxing/llava_next && git pull --ff-only"
```

If remote `git status --short` shows uncommitted changes, stop and report them instead of overwriting. The remote directory also contains ignored local assets such as models, datasets, outputs, and checkpoints; those should remain untracked.

## Quick Environment Checks

Verify SSH:

```powershell
ssh myserver "hostname"
```

Verify container mount:

```powershell
ssh myserver "docker exec qwen2vl_wzx bash -lc 'cd /workspace && pwd && ls'"
```

Verify GPU visibility:

```powershell
ssh myserver "docker exec qwen2vl_wzx bash -lc 'nvidia-smi'"
```

Verify project Git state on the host:

```powershell
ssh myserver "cd /mnt/data2/wuzhengxing/llava_next && git status --short && git log --oneline -1"
```

## Run Commands In The Container

Run short diagnostics directly:

```powershell
ssh myserver "cd /mnt/data2/wuzhengxing/llava_next && git pull --ff-only && docker exec qwen2vl_wzx bash -lc 'cd /workspace && <command> 2>&1'"
```

Examples:

```powershell
ssh myserver "cd /mnt/data2/wuzhengxing/llava_next && git pull --ff-only && docker exec qwen2vl_wzx bash -lc 'cd /workspace && python -V && git status 2>&1'"
```

```powershell
ssh myserver "cd /mnt/data2/wuzhengxing/llava_next && git pull --ff-only && docker exec qwen2vl_wzx bash -lc 'cd /workspace && bash test_train.sh 2>&1'"
```

## Start Training With tmux

Start a long run after pulling latest code:

```powershell
ssh myserver "cd /mnt/data2/wuzhengxing/llava_next && git pull --ff-only && docker exec qwen2vl_wzx bash -lc 'cd /workspace && tmux kill-session -t train 2>/dev/null || true; tmux new -d -s train \"<training-command> 2>&1 | tee train.log\"'"
```

Example:

```powershell
ssh myserver "cd /mnt/data2/wuzhengxing/llava_next && git pull --ff-only && docker exec qwen2vl_wzx bash -lc 'cd /workspace && tmux kill-session -t train 2>/dev/null || true; tmux new -d -s train \"bash train.sh 2>&1 | tee train.log\"'"
```

Use timestamped logs for experiments:

```powershell
ssh myserver "cd /mnt/data2/wuzhengxing/llava_next && git pull --ff-only && docker exec qwen2vl_wzx bash -lc 'cd /workspace && mkdir -p logs && tmux kill-session -t train 2>/dev/null || true; tmux new -d -s train \"<training-command> 2>&1 | tee logs/train_`$(date +%Y%m%d_%H%M%S).log\"'"
```

## Inspect Or Stop Training

Read recent logs:

```powershell
ssh myserver "docker exec qwen2vl_wzx bash -lc 'cd /workspace && tail -n 200 train.log'"
```

Follow logs when useful:

```powershell
ssh myserver "docker exec qwen2vl_wzx bash -lc 'cd /workspace && tail -f train.log'"
```

Capture tmux pane:

```powershell
ssh myserver "docker exec qwen2vl_wzx bash -lc 'tmux capture-pane -t train -p | tail -n 200'"
```

List tmux sessions:

```powershell
ssh myserver "docker exec qwen2vl_wzx bash -lc 'tmux ls'"
```

Stop training:

```powershell
ssh myserver "docker exec qwen2vl_wzx bash -lc 'tmux kill-session -t train'"
```

## Debugging Loop

When a remote run fails:

1. Read the latest log or tmux pane.
2. Identify the first actionable error.
3. Map `/workspace/...` paths to local files in `D:\codex\llava_next`.
4. Patch local code, scripts, or configs.
5. Run local checks if possible.
6. Commit and push when remote validation is needed.
7. Pull on myserver and rerun inside `qwen2vl_wzx`.
8. Continue until fixed or blocked by missing data, permissions, GPU availability, or environment state.

Path mapping examples:

```text
/workspace/qwen_3d/train/train_3d.py -> D:\codex\llava_next\qwen_3d\train\train_3d.py
/workspace/qwen_vl_finetune/train/train_multimodal.py -> D:\codex\llava_next\qwen_vl_finetune\train\train_multimodal.py
/workspace/train_bash/train.sh -> D:\codex\llava_next\train_bash\train.sh
```

## Repository Ignore Policy

This project intentionally tracks source code, scripts, Docker/requirements files, and small configs. It intentionally ignores models, checkpoints, datasets, processed data, images, logs, temporary files, and training outputs.

Do not add ignored data or checkpoint artifacts to Git unless the user explicitly requests a small fixture or config sample.

## Reporting

After completing a code-and-remote-validation cycle, report:

- Local files changed.
- Commit hash if committed.
- Remote command run.
- Relevant stdout, traceback, or log tail.
- Whether remote Git state is clean.
