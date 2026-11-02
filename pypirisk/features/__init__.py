"""Features stage: point-in-time features for each package at each snapshot.

Every feature is computed only from events dated strictly before the snapshot. The
leakage tests in ``tests/leakage`` check this by adding future events and asserting
the features do not change.
"""
