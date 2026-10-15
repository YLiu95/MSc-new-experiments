from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    arm: str
    nodes: int
    node_rank: int
    leader_index: int


def groups_for(nodes: int, process_id: int) -> tuple[Group, ...]:
    if not 0 <= process_id < nodes:
        raise ValueError("Task ID is outside the allocated node count")
    if nodes == 1:
        return tuple(Group(arm, 1, 0, 0) for arm in ("A", "B", "C"))
    if nodes == 2:
        arms = ("B", "A") if process_id == 0 else ("C",)
        return tuple(Group(arm, 1, 0, process_id) for arm in arms)
    if nodes == 3:
        return (Group("ABC"[process_id], 1, 0, process_id),)
    if nodes == 8:
        for arm, start, size in (("A", 0, 2), ("B", 2, 3), ("C", 5, 3)):
            if start <= process_id < start + size:
                return (Group(arm, size, process_id - start, start),)
    raise ValueError("Only the registered 1/2/3/8-node arm layouts are supported")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--process-id", type=int, required=True)
    arguments = parser.parse_args()
    for group in groups_for(arguments.nodes, arguments.process_id):
        print(f"{group.arm}:{group.nodes}:{group.node_rank}:{group.leader_index}")


if __name__ == "__main__":
    main()