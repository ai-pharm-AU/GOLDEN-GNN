## Plan: Initialize and Push Deliverables Repo

Use this workspace as a new Git repo, push directly to `main`, authenticate with SSH, and include only GitHub-safe artifacts. The package already declares this intent in [README.md](README.md) and excludes known oversized KG artifacts in [.gitignore](.gitignore), with overflow tracked in [HF_UPLOAD_QUEUE.csv](HF_UPLOAD_QUEUE.csv). The plan below focuses on preventing accidental large-file commits, then doing a single clean first push to `git@github.com:ai-pharm-AU/GOLDEN-GNN.git`.

**Steps**
1. Preflight repository safety checks using [README.md](README.md), [.gitignore](.gitignore), and [HF_UPLOAD_QUEUE.csv](HF_UPLOAD_QUEUE.csv) to confirm “code/docs/config only” scope is enforced.
2. Initialize Git in workspace root (`/home/zzz0054/deliverables`), set branch to `main`, and add remote `origin` as SSH URL for `ai-pharm-AU/GOLDEN-GNN`.
3. Stage files selectively with ignore rules active, then run a staged-file size audit to ensure nothing near/over GitHub hard limits is included.
4. Create the initial commit (after your required checks pass per [CLAUDE.md](CLAUDE.md)) and push `main` to `origin`.
5. Validate remote state (`git remote -v`, `git ls-remote`, GitHub web check) and confirm only intended artifacts are present.

**Verification**
- Run all commands inside `tmux` sessions.
- Use `/home/zzz0054/GoldenF/.venv` for any project checks.
- Perform pre-commit checks first (per [CLAUDE.md](CLAUDE.md)); then run:
  - `git status --short`
  - staged size check (largest tracked files)
  - `git push -u origin main`
  - remote confirmation (`git ls-remote --heads origin main`)

**Decisions**
- Setup: initialize Git directly in this folder.
- Target: push directly to `main`.
- Scope: exclude large generated artifacts; keep HF handoff separate.
- Auth: SSH-based push to `ai-pharm-AU/GOLDEN-GNN`.

If you want, I can now refine this into an exact command-by-command runbook (with `tmux` command sequence) for handoff execution.
