# Repository Lineage

`main` is the only canonical accepted implementation. Development uses at most
one active `feature/*` branch at a time.

Completed work is retained by immutable tags:

- `thesis-*` tags identify qualified scientific milestones.
- `archive/*` tags retain rejected experiments, superseded probes, and recovery
  points that remain useful for audit.

After a feature is merged or archived, delete its local and remote branch. Do
not keep milestone branches open as historical storage.

Agent worktrees and clones must live outside the repository checkout. Before
removing one, require a clean status and preserve its exact head through
`main`, a published feature branch, or an annotated archive tag.

To find the current code, use `main`. To inspect a frozen result, use its
`thesis-*` tag. An active feature branch is experimental until it is qualified
and merged.
