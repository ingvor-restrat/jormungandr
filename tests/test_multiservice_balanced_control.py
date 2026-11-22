import numpy as np

from examples.multiservice_balanced_control import balanced_worlds
from examples.multiservice_adaptation import context, evaluate, observe


def test_rotated_worlds_preserve_public_inputs_and_balance_hidden_winners():
    for n in (4, 8):
        worlds = balanced_worlds(31, n, groups=2, steps=16)
        for start in range(0, len(worlds), n):
            group = worlds[start : start + n]
            assert {w.best[0] for w in group} == set(range(n))
            first = group[0]
            expected = observe(first, 0, context(first), "no_history")
            for w in group:
                obs = observe(w, 0, context(w), "no_history")
                assert w.ids == first.ids
                np.testing.assert_array_equal(w.orders, first.orders)
                np.testing.assert_array_equal(w.exploration, first.exploration)
                np.testing.assert_array_equal(w.probes, first.probes)
                for key in ("history", "observation", "candidate_descriptions"):
                    assert obs.metadata["llm"][key] == expected.metadata["llm"][key]
                np.testing.assert_array_equal(
                    obs.candidate_features, expected.candidate_features
                )
        result = evaluate(None, worlds, "random")
        assert result["mean_reward"] == 1 / n
