"""Label stage: which eligible packages were abandoned in each outcome window.

The label spec (provisional, decided 2026-10-31): a package that was actively
maintained at the snapshot is *abandoned* if its repository was archived during the
next ``outcome_months``, or if it made no release and at most ``max_commits_in_outcome``
human commits in that window.
"""
