import pytest

from jormungandr.benchmarks import (
    RelationalSupervisionConfig,
    relational_supervision_corpus,
    run_relational_supervision_benchmark,
)


def test_relational_corpus_is_split_complete_and_candidate_conditioned() -> None:
    config = RelationalSupervisionConfig(
        workers=3,
        train_worlds=4,
        validation_worlds=2,
        updates=1,
        batch_size=4,
        model_dim=8,
        heads=2,
        feedforward_dim=16,
    )
    spec, train, validation = relational_supervision_corpus(config)

    assert len(train) == 12
    assert len(validation) == 6
    assert spec.entity_dim == 7
    assert {item.split for item in train} == {"train"}
    assert {item.split for item in validation} == {"validation"}
    assert all(
        (item.observation.candidate_entity_indices >= 0).all()
        for item in train + validation
    )


def test_relational_benchmark_compares_one_architecture_change() -> None:
    result = run_relational_supervision_benchmark(
        RelationalSupervisionConfig(
            workers=2,
            train_worlds=8,
            validation_worlds=4,
            updates=2,
            batch_size=8,
            model_dim=8,
            heads=2,
            feedforward_dim=16,
        )
    )

    assert result["schema"] == "jormungandr.relational_supervision_benchmark.v1"
    assert set(result["arms"]) == {
        "pooled_context",
        "candidate_entity_attention",
    }
    assert result["arms"]["pooled_context"]["candidate_attention_layers"] == 0
    assert result["arms"]["candidate_entity_attention"][
        "candidate_attention_layers"
    ] == 1
    assert result["arms"]["candidate_entity_attention"][
        "trainable_parameters"
    ] > result["arms"]["pooled_context"]["trainable_parameters"]


def test_relational_benchmark_rejects_invalid_world_contract() -> None:
    with pytest.raises(ValueError, match="at least two workers"):
        RelationalSupervisionConfig(workers=1)
