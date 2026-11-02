"""Graph stage: the dependency graph as it was at each snapshot.

Each snapshot's graph uses, for every package, the dependencies declared by the release
that was newest at that date, never today's. Network metrics (dependents, PageRank,
k-core) are computed per snapshot so they carry no information from the future.
"""
