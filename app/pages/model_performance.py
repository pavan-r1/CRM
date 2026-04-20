from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ai import churn_model


def render() -> None:
    st.subheader("Model Performance")
    st.caption("Churn model quality snapshot for demo credibility.")

    if st.button("Train / Refresh Churn Model"):
        metrics, _ = churn_model.train_and_score()
        st.session_state["latest_model_metrics"] = metrics

    metrics = st.session_state.get("latest_model_metrics")
    if metrics is None:
        st.info("Run model training to view accuracy, precision, recall, and confusion matrix.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics.accuracy:.3f}")
    c2.metric("Precision", f"{metrics.precision:.3f}")
    c3.metric("Recall", f"{metrics.recall:.3f}")
    c4.metric("F1", f"{metrics.f1:.3f}")

    matrix = pd.DataFrame(
        metrics.confusion_matrix,
        index=["Actual: Stay", "Actual: Churn"],
        columns=["Pred: Stay", "Pred: Churn"],
    )
    st.markdown("#### Confusion Matrix")
    st.dataframe(matrix, width="stretch")

    st.markdown("### Model Insights")
    st.markdown(
        "\n".join(
            [
                f"- Accuracy ({metrics.accuracy:.3f}) shows overall prediction correctness.",
                f"- Precision ({metrics.precision:.3f}) shows how often predicted churn is truly churn.",
                f"- Recall ({metrics.recall:.3f}) shows how many actual churners are captured.",
                f"- F1 ({metrics.f1:.3f}) balances precision and recall for reliable monitoring.",
            ]
        )
    )

    tn, fp = int(metrics.confusion_matrix[0][0]), int(metrics.confusion_matrix[0][1])
    fn, tp = int(metrics.confusion_matrix[1][0]), int(metrics.confusion_matrix[1][1])
    st.markdown("#### Confusion Matrix Interpretation")
    st.markdown(
        "\n".join(
            [
                f"- True Positives: {tp} accounts correctly identified as churn risk.",
                f"- False Negatives: {fn} churn-risk accounts missed by the model (highest business risk).",
                f"- False Positives: {fp} accounts flagged as risk but may stay (acceptable outreach overhead).",
                f"- True Negatives: {tn} stable accounts correctly recognized as low churn risk.",
            ]
        )
    )

    st.markdown("### Business Impact")
    st.markdown(
        "\n".join(
            [
                "- Improves retention planning by identifying vulnerable customers earlier.",
                "- Reduces reactive firefighting by prioritizing proactive outreach.",
                "- Focuses customer success effort on accounts with the highest downside risk.",
                "- Supports measurable churn reduction through model-driven interventions.",
            ]
        )
    )

    feature_importance = getattr(metrics, "feature_importance", {})
    if feature_importance:
        st.markdown("### Feature Importance")
        importance = (
            pd.Series(feature_importance)
            .sort_values(ascending=False)
            .head(8)
            .rename("importance")
        )
        st.bar_chart(importance)
