import pytest

from ranker.layout import groups_for


def test_eight_nodes_are_fully_assigned_with_matched_b_c_dp():
    groups = [groups_for(8, process_id)[0] for process_id in range(8)]
    assert [group.arm for group in groups] == ["A", "A", "B", "B", "B", "C", "C", "C"]
    assert [group.nodes for group in groups] == [2, 2, 3, 3, 3, 3, 3, 3]
    assert [group.node_rank for group in groups] == [0, 1, 0, 1, 2, 0, 1, 2]
    assert [group.leader_index for group in groups] == [0, 0, 2, 2, 2, 5, 5, 5]
    assert sum(group.node_rank == 0 for group in groups) == 3


def test_initial_small_node_layouts_and_unsupported_requests():
    assert [group.arm for group in groups_for(1, 0)] == ["A", "B", "C"]
    assert [group.arm for group in groups_for(2, 0)] == ["B", "A"]
    assert [group.arm for group in groups_for(2, 1)] == ["C"]
    assert [groups_for(3, process_id)[0].arm for process_id in range(3)] == ["A", "B", "C"]
    with pytest.raises(ValueError):
        groups_for(8, 8)
    with pytest.raises(ValueError):
        groups_for(6, 0)