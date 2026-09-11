# Scientific Repository Lineage

The repository contains historical development commits, but scientific claims are bound
to explicit immutable identities rather than to whichever branch is currently checked
out.

## Frozen experiment identities

| Role | Commit/tag |
| --- | --- |
| Final composed UPMEM executor | `459935f586fdd16c82013838e6d27a12604c3093` / `thesis-upmem-kernel-schedule-system-v1` |
| P6 qualified software/controller | `2beea27411c16e90ed76988613ddb00bcc09f942` / `thesis-upmem-cost-guided-software-v1` |
| P6 accepted result/audit package | `8df2ebac61bacd08309ea490309be5a8dcb943b2` / `thesis-upmem-cost-guided-results-v1` |

The P6 results commit is a direct descendant of the qualified P6 software commit and adds
the audit/result package and operational runbook; it does not redefine the physical
executor.

## Branch policy after finalization

The completed feature branch is merged to `main` by fast-forward only. Publication and
documentation corrections happen after the immutable scientific tags and do not change
the experiment identities above.

The standalone thesis repository is a publication/reproducibility projection of the
`thesis/` subtree. Its own release tag identifies the publication snapshot; it does not
retroactively become the source SHA recorded in historical physical evidence.

## Historical documents

Superseded designs and negative experiments remain accessible because they document the
research process. Their presence does not make them active protocols. The documentation
index labels them explicitly.

## Evidence lineage

Physical-source commit, analysis/reporting commit, archive hashes, and publication commit
are intentionally distinct where appropriate. A later reporting correction must never be
described as though the physical measurements were rerun at the later source.
