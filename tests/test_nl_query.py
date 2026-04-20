from app.ai import churn_model, health_score, nl_query
from app.data.generate_synthetic_data import SeedConfig, generate_synthetic_dataset
from app.db.schema import initialize_database


def test_nl_query_with_follow_up_context() -> None:
    initialize_database()
    generate_synthetic_dataset(SeedConfig(num_customers=220, random_seed=13))
    health_score.refresh_all_health_scores()
    churn_model.train_and_score()

    session_id = "test-session"
    first = nl_query.run_query("Show all customers in EMEA", session_id)
    second = nl_query.run_query("only enterprise plans", session_id)

    assert not first.dataframe.empty
    assert "region" in first.context
    assert second.context["plan_tier"] == "Enterprise"
