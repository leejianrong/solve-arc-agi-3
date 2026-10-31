from solve_arc_agi_3.state_graph import StateGraph


def _graph() -> StateGraph:
    g = StateGraph()
    g.add(1, "A", 2)
    g.add(2, "A", 3)
    g.add(1, "B", 1)
    g.add(3, "B", 2)
    return g


def test_visits_and_unexplored() -> None:
    g = _graph()
    assert g.visits(2) == 2 and g.visits(1) == 1 and g.visits(99) == 0
    assert g.unexplored(1, ["A", "B", "C"]) == ["C"]
    assert g.unexplored(99, ["A"]) == ["A"]


def test_is_loop() -> None:
    assert StateGraph.is_loop([1, 2, 3, 2])
    assert not StateGraph.is_loop([1, 2, 3])
    assert not StateGraph.is_loop([])


def test_path_bfs() -> None:
    g = _graph()
    g.add(1, "C", 3)
    assert g.path(1, 3) == ["C"]
    assert g.path(2, 2) == []
    assert g.path(3, 1) is None
    assert _graph().path(1, 3) == ["A", "A"]


def test_summary() -> None:
    g = _graph()
    s = g.summary(1, ["A", "B", "C"], 200)
    assert "seen 1 time." in s and "Untried from here: C." in s
    assert "A->seen before" in s and "B->no change" in s
    h = StateGraph()
    h.add(1, "A", 2)
    assert "A->new" in h.summary(1, ["A"])
    assert len(g.summary(1, ["A", "B", "C"], 30)) <= 30
    assert "Untried from here: none" in g.summary(99, [])
