from app.ai.health_score import calculate_customer_health, refresh_all_health_scores
from app.data.generate_synthetic_data import SeedConfig, generate_synthetic_dataset
from app.db.schema import initialize_database


def test_health_scores_range() -> None:
    initialize_database()
    generate_synthetic_dataset(SeedConfig(num_customers=205, random_seed=9))
    refresh_all_health_scores()

    score, details = calculate_customer_health(1)
    assert 0 <= score <= 100
    assert set(details.keys()) == {"tickets", "contract", "usage", "nps"}
