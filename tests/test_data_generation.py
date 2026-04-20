from app.data.generate_synthetic_data import SeedConfig, data_quality_summary, generate_synthetic_dataset
from app.db.schema import initialize_database


def test_synthetic_data_volume() -> None:
    initialize_database()
    generate_synthetic_dataset(SeedConfig(num_customers=210, random_seed=7))
    summary = data_quality_summary()

    assert summary["customers"] >= 200
    assert summary["tickets"] > 0
    assert summary["devices"] > 0
    assert summary["usage_rows"] >= 200 * 12
    assert summary["invalid_contracts"] == 0
