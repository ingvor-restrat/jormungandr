import numpy as np
import torch

from examples.noisy_state_auxiliary import (
    HORIZONS,
    WINDOW,
    Model,
    Task,
    collect,
    features,
    filter_posterior,
    targets,
    trajectory,
    update,
)


def test_history_is_causal_and_targets_are_strictly_future():
    sensor = np.arange(40, dtype=np.float32)
    edited = sensor.copy()
    edited[20:] = -100
    np.testing.assert_array_equal(features(sensor)[:20], features(edited)[:20])
    np.testing.assert_array_equal(features(sensor)[19, :WINDOW], sensor[4:20])
    future, change = targets(sensor)
    assert len(future) == len(sensor) - max(HORIZONS)
    for j, horizon in enumerate(HORIZONS):
        np.testing.assert_array_equal(
            future[:, j], sensor[horizon : horizon + len(future)]
        )
    assert not change.any()
    assert features(sensor)[0, WINDOW:].sum() == 1


def test_filter_matches_brute_force_hidden_paths():
    from itertools import product

    z = np.array([0.3, -0.2, 0.8])
    noise, hazard = 0.9, 0.15
    numerator, denominator = 0.0, 0.0
    for path in product((0, 1), repeat=len(z)):
        prob = 0.5
        for t, state in enumerate(path):
            if t:
                prob *= hazard if state != path[t - 1] else 1 - hazard
            prob *= np.exp(-0.5 * ((z[t] - (2 * state - 1)) / noise) ** 2)
        numerator += prob * path[-1]
        denominator += prob
    np.testing.assert_allclose(
        filter_posterior(z, noise, hazard)[-1], numerator / denominator
    )
    np.testing.assert_allclose(
        filter_posterior(z, noise, hazard, 1), 1 / (1 + np.exp(-2 * z / noise**2))
    )


def test_episode_boundaries_and_repeatability():
    a, b = collect(123, 4), collect(123, 4)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
    assert len(a[0]) == 4 * (256 - max(HORIZONS))
    for i in range(4):
        assert a[0][i * (256 - max(HORIZONS)), WINDOW:].sum() == 1
    sensor, _, _, _ = trajectory(123, Task(0.7, 0.01))
    np.testing.assert_array_equal(a[3][0], sensor[list(HORIZONS)])


def test_future_and_privileged_labels_do_not_affect_reward_only_update():
    batch = [v[:64] for v in collect(41, 4)]
    models = []
    for corrupt in (False, True):
        torch.manual_seed(3)
        model = Model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        values = [v.clone() for v in batch]
        if corrupt:
            values[3].fill_(float("nan"))
            values[4].fill_(float("nan"))
            values[5].fill_(-999)
        update(model, optimizer, values, "reward", 0.1)
        models.append(model)
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b)


def test_predictive_loss_trains_shared_encoder_and_variance_head():
    torch.manual_seed(4)
    model = Model()
    old = {k: p.detach().clone() for k, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metrics = update(model, optimizer, [v[:128] for v in collect(71, 4)], "joint", 0.1)
    assert np.isfinite(list(metrics.values())).all()
    assert not torch.equal(model.forecast.weight, old["forecast.weight"])
    assert not torch.equal(model.encoder[0].weight, old["encoder.0.weight"])
    torch.testing.assert_close(model.state.weight, old["state.weight"])
