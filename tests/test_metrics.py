from eval.retrieval_eval import recall_at_k, reciprocal_rank


def test_recall_finds_single_gold_in_range():
    # gold chunk 3 sits at position 2 of the ranking, within k=3
    ranked = [5, 3, 9]
    gold = [3]
    assert recall_at_k(ranked, gold, 3) == 1.0


def test_recall_misses_gold_outside_k():
    # same ranking, but k=1 only looks at position 1
    ranked = [5, 3, 9]
    gold = [3]
    assert recall_at_k(ranked, gold, 1) == 0.0


def test_recall_gives_partial_credit_for_multi_gold():
    # two gold chunks, only one inside k=2
    ranked = [5, 3, 9]
    gold = [3, 9]
    assert recall_at_k(ranked, gold, 2) == 0.5


def test_recall_full_credit_when_all_gold_in_range():
    ranked = [5, 3, 9]
    gold = [3, 9]
    assert recall_at_k(ranked, gold, 3) == 1.0


def test_recall_handles_empty_gold():
    # a golden.jsonl entry with no labels should not crash or score
    assert recall_at_k([5, 3, 9], [], 3) == 0.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([3, 5, 9], [3]) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank([5, 3, 9], [3]) == 0.5


def test_reciprocal_rank_uses_earliest_gold():
    # both 3 and 9 are gold; MRR should use whichever appears first
    assert reciprocal_rank([5, 3, 9], [3, 9]) == 0.5


def test_reciprocal_rank_zero_when_absent():
    assert reciprocal_rank([5, 7, 9], [3]) == 0.0