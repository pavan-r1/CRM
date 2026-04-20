from app.ai.churn_model import train_and_score
from app.data.generate_synthetic_data import SeedConfig, generate_synthetic_dataset
from app.db.schema import initialize_database


def test_churn_training_outputs_metrics() -> None:
    initialize_database()
    generate_synthetic_dataset(SeedConfig(num_customers=220, random_seed=11))

    metrics, scored = train_and_score()

    assert 0 <= metrics.precision <= 1
    assert 0 <= metrics.recall <= 1
    assert 0 <= metrics.f1 <= 1
    assert len(scored) >= 200
