from scripts.probe_native_source_grouping import source_groups,ranked_exact_tokens


def test_geometric_partition_changes_membership_without_count_size_or_coverage_confounds():
    a=source_groups('flat_matched');b=source_groups('spatial8')
    assert [len(g) for g in a]==[len(g) for g in b]
    assert len(a)==len(b)==120
    assert a!=b
    assert sorted(x for g in a for x in g)==sorted(x for g in b for x in g)==list(range(7040))


def test_teacher_diagnostic_has_exact_budget_and_stable_ties():
    for kind in ('flat64','flat_matched','spatial8'):
        groups=source_groups(kind);ids=ranked_exact_tokens(groups,[0.]*len(groups),1760)
        assert len(ids)==len(set(ids))==1760
        assert ids==sorted(ids)
