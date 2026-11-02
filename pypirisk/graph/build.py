"""Build point-in-time dependency graphs and their network metrics."""

import networkx as nx
import pandas as pd


def current_versions(releases: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Return each package's newest release uploaded before ``snapshot``.

    Args:
        releases: ``name``, ``version``, ``uploaded_at``.
        snapshot: The snapshot time.

    Returns:
        ``name`` and ``version`` for every package that existed at the snapshot.
    """
    past = releases[releases["uploaded_at"] < snapshot]
    latest = past.sort_values("uploaded_at").groupby("name").tail(1)
    return latest[["name", "version"]].reset_index(drop=True)


def snapshot_edges(
    releases: pd.DataFrame, requires: pd.DataFrame, snapshot: pd.Timestamp
) -> pd.DataFrame:
    """List ``source -> target`` dependency edges as they were at ``snapshot``.

    Args:
        releases: The release table.
        requires: ``name``, ``version``, ``dependency``.
        snapshot: The snapshot time.

    Returns:
        De-duplicated edges whose both ends existed at the snapshot.
    """
    versions = current_versions(releases, snapshot)
    edges = versions.merge(requires, on=["name", "version"])
    edges = edges[
        edges["dependency"].isin(versions["name"]) & (edges["dependency"] != edges["name"])
    ]
    edges = edges.rename(columns={"name": "source", "dependency": "target"})
    return edges[["source", "target"]].drop_duplicates().reset_index(drop=True)


def network_metrics(names: list[str], edges: pd.DataFrame) -> pd.DataFrame:
    """Compute network metrics for every package in one snapshot's graph.

    PageRank runs on ``package -> dependency`` edges, so packages that many important
    packages depend on rank highest. It is divided by the snapshot's smallest PageRank,
    so the many leaf packages always score exactly 1 and the values stay comparable
    across snapshots whose graphs differ in size and shape. k-core is computed on the
    undirected graph.

    Args:
        names: Every package that existed at the snapshot.
        edges: That snapshot's edges.

    Returns:
        ``name``, ``dependents`` (in-degree), ``pagerank`` and ``kcore``.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(names)
    graph.add_edges_from(edges.itertuples(index=False, name=None))
    pagerank = nx.pagerank(graph) if graph.number_of_nodes() else {}
    floor = min(pagerank.values(), default=1.0)
    kcore = nx.core_number(graph.to_undirected()) if graph.number_of_nodes() else {}
    return pd.DataFrame(
        {
            "name": names,
            "dependents": [graph.in_degree(name) for name in names],
            "pagerank": [pagerank.get(name, floor) / floor for name in names],
            "kcore": [kcore.get(name, 0) for name in names],
        }
    )


def build_graphs(
    releases: pd.DataFrame, requires: pd.DataFrame, snapshots: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the graph and its metrics at every snapshot.

    Args:
        releases: The release table.
        requires: The dependency table.
        snapshots: ISO snapshot dates.

    Returns:
        The edges and the node metrics, each with a ``snapshot`` column.
    """
    all_edges, all_nodes = [], []
    for snapshot in snapshots:
        moment = pd.Timestamp(snapshot, tz="UTC")
        edges = snapshot_edges(releases, requires, moment)
        names = sorted(current_versions(releases, moment)["name"])
        all_edges.append(edges.assign(snapshot=snapshot))
        all_nodes.append(network_metrics(names, edges).assign(snapshot=snapshot))
    return pd.concat(all_edges, ignore_index=True), pd.concat(all_nodes, ignore_index=True)
