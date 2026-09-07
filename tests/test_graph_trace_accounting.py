import sqlite3
import pytest

from scripts.audit_generator_timeline import require_node_graph_trace


def test_aggregate_graph_trace_cannot_silently_be_counted_as_GPU_idle():
    db = sqlite3.connect(':memory:')
    require_node_graph_trace(db)
    db.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (start INTEGER, end INTEGER)')
    require_node_graph_trace(db)
    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (1, 10)')
    with pytest.raises(ValueError, match='cuda-graph-trace=node'):
        require_node_graph_trace(db)
    db.close()
