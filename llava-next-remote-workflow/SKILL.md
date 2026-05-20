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

## Strict Safety Boundaries

Operate only in these approved areas:

```text
Local editable code area: D:\codex\llava_next
Remote host project area: /mnt/data2/wuzhengxing/llava_next
Container project area: /workspace
Project-owned tmux session: train
Project-owned logs: /workspace/train.log and /workspace/logs/
```

On the remote host, allowed actions are limited to:

- Read-only checks such as `pwd`, `ls`, `git status`, `git log`, `git diff`, `du` for project paths, and environment checks.
- `git pull --ff-only` inside `/mnt/data2/wuzhengxing/llava_next`.
- `docker exec qwen2vl_wzx bash -lc 'cd /workspace && ...'` for project-scoped diagnostics, tests, training, log reads, and tmux management.

Do not modify anything outside `/mnt/data2/wuzhengxing/llava_next` on the host or outside `/workspace` in the container. Do not edit host files directly except Git metadata created by normal `git pull` inside the project. Put source-code changes through the local repo, commit, push, and remote pull workflow.

Forbidden unless the user explicitly requests it in the current turn:

- Rebooting or shutting down the server.
- Starting, stopping, restarting, removing, or recreating system services, Docker daemon, containers other than the project-owned tmux session, or unrelated jobs.
- Using `sudo`, changing users, changing SSH config, changing firewall/network/storage/system package settings, or modifying global Git config.
- Running `rm`, `mv`, `chmod`, `chown`, `git reset --hard`, `git clean`, `docker stop`, `docker restart`, `docker rm`, `kill`, or `pkill` unless the target is proven to be inside the approved project scope and the user has approved destructive intent.
- Deleting models, datasets, checkpoints, outputs, processed data, or logs as a cleanup shortcut.

Before any host-side write, pull, or long run, perform a safety preflight:

```powershell
ssh myserver "cd /mnt/data2/wuzhengxing/llava_next && pwd && git status --short && git rev-parse --abbrev-ref HEAD && git log --oneline -1"
```

Proceed only if `pwd` is `/mnt/data2/wuzhengxing/llava_next`, the branch is `main`, and remote changes are understood. If the remote worktree contains unexpected modifications, stop and report them.

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

Only use commands that operate on the project and do not change host-level configuration or services.

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

Only stop the `train` tmux session when it is the project run for the current task or when the user explicitly asked to stop it. Do not stop other tmux sessions, processes, containers, or services.

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

Never use cleanup commands to force Git or training to proceed. If ignored artifacts, checkpoints, or logs interfere with a task, report the path and ask before deleting, moving, or changing permissions.

## Reporting

After completing a code-and-remote-validation cycle, report:

- Local files changed.
- Commit hash if committed.
- Remote command run.
- Relevant stdout, traceback, or log tail.
- Whether remote Git state is clean.
- Any safety-sensitive decision that was skipped, blocked, or required user approval.
