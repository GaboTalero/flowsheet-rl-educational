import pytest

from flowsheet_rl.graph_state import FlowsheetGraph


def test_topology_and_cycle_guard():
    g = FlowsheetGraph.initial(1, 1)
    g.validate()
    assert g.open_nodes == [1, 3]
    g.graph.add_edge(1, 0, port=0)
    with pytest.raises(ValueError, match="cycle"):
        g.validate()


def test_serialization_is_stable_and_copy_independent():
    g = FlowsheetGraph.initial(1, 1)
    assert g.record() == g.copy().record()
    c = g.copy()
    c.add_unit(1, "purge")
    assert g.open_nodes == [1, 3] and c.open_nodes == [3]
