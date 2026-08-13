# Agent Guides

Preferences for how AI agents should conduct verification, testing, and exploratory work in this repository — distinct from `dev_guides.md` (code conventions) and `test_guides.md` (test-writing conventions), which govern what gets committed. This file governs the agent's own working habits during a session.

## Clean Up Verification Side Effects

- When running a command purely to **verify** something works — not because the user asked for the resulting artifact itself — and that command creates a persistent side effect (a file, a directory, registered system state, an installed package, a running process), remove it once verification is complete, in the same turn, without waiting to be asked.
- This applies even when the artifact is gitignored. Gitignored files still show up in the file explorer, `ls`, and the IDE — being invisible to `git status` doesn't make an untracked, unexplained file any less confusing to find later.
- This applies doubly to state registered **outside the repo entirely** (e.g. a Jupyter kernelspec written to `~/Library/Jupyter/kernels/`, a globally-installed tool, a registered service) — these aren't cleaned up by any `git`-based check at all, so the agent is the only thing that will ever remove them.
- Before ending a turn that ran exploratory/verification commands, explicitly check what state was created and decide: does this persist because the user asked for it, or was it only for verification and now needs removing?

```
# example: verifying a generic Makefile target works across different inputs
make install_kernel KERNEL_NAME=nlp KERNEL_DEPS="rich"   # verification run only

# correct — clean up both the repo-local artifact and the out-of-repo registered state,
# in the same turn, before reporting the verification result
rm -rf .venv-nlp
jupyter kernelspec remove -f velari-nlp

# wrong — leaving .venv-nlp/ and the registered "velari-nlp" kernelspec behind because
# they're gitignored / outside the repo and "won't show up in git status anyway"
```

- Exception: artifacts the user actually asked to create (e.g. running `make install_kernel` for real, not as a verification probe) obviously stay — this rule is about throwaway verification runs, not the deliverable itself.
