"""NovaPay Fraud Detection — Streamlit analyst console.

A thin client over the FastAPI service (api/main.py): scores single or
batch transactions, renders SHAP-based explanations, surfaces model
metadata, and gives fraud-ops a drift-monitoring view. This app holds no
modeling logic itself -- every score comes from the same `/score` endpoint
the production API serves, so what an analyst sees here is exactly what a
real caller gets, not a separate reimplementation that could drift out of
sync.

Run:
    streamlit run frontend/app.py

Requires the API to be running (default http://localhost:8000, configurable
in the sidebar). The Model Monitoring tab is the one exception: it imports
novapay_fraud.monitoring directly rather than going through the API, since
drift checking is an offline/batch analyst workflow, not a live scoring
endpoint (see api/main.py -- there is no /drift route on purpose).
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="NovaPay Fraud Detection Console",
    page_icon="🛡️",
    layout="wide",
)

DEFAULT_API_URL = os.environ.get("NOVAPAY_API_URL", "http://localhost:8000")
REQUIRED_COLUMNS = [
    "transaction_id", "timestamp", "home_country", "source_currency",
    "dest_currency", "channel", "kyc_tier", "ip_country", "amount_src",
    "amount_usd", "fee", "exchange_rate_src_to_dest", "new_device",
    "location_mismatch", "ip_risk_score", "account_age_days",
    "device_trust_score", "chargeback_history_count", "risk_score_internal",
    "txn_velocity_1h", "txn_velocity_24h", "corridor_risk",
]

# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------

def api_get(base_url: str, path: str, timeout: float = 5.0):
    resp = requests.get(f"{base_url}{path}", timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def api_post(base_url: str, path: str, payload: dict, timeout: float = 10.0):
    resp = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        raise RuntimeError(f"API error {resp.status_code}: {detail}")
    return resp.json()


def check_health(base_url: str) -> dict | None:
    try:
        return api_get(base_url, "/health", timeout=3.0)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar: API connection + health
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ NovaPay Fraud Console")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL).rstrip("/")

    health = check_health(api_url)
    if health and health.get("model_loaded"):
        st.success(f"API connected — model: {health['model_version']}")
    elif health:
        st.warning("API reachable but model not loaded")
    else:
        st.error("API unreachable. Start it with:\n\n`uvicorn api.main:app`")

    st.divider()
    st.caption(
        "This console calls the same `/score` endpoint any production "
        "caller uses. It does not score transactions itself."
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_single, tab_batch, tab_model, tab_monitor = st.tabs(
    ["🔎 Score a Transaction", "📦 Batch Scoring", "📊 Model Info", "📈 Drift Monitoring"]
)

# ---------------------------------------------------------------------------
# Tab 1: single transaction scoring
# ---------------------------------------------------------------------------
with tab_single:
    st.subheader("Score a single transaction")
    st.caption("Fill in a transaction's raw fields — engineered risk features are computed server-side.")

    with st.form("single_score_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            transaction_id = st.text_input("Transaction ID", value="txn_demo_1")
            home_country = st.selectbox("Home country", ["US", "CA", "UK"])
            source_currency = st.text_input("Source currency", value="USD")
            dest_currency = st.text_input("Destination currency", value="MXN")
            channel = st.selectbox("Channel", ["WEB", "MOBILE", "ATM"])
            kyc_tier = st.selectbox("KYC tier", ["LOW", "STANDARD", "ENHANCED"])
            ip_country = st.selectbox("IP country", ["US", "CA", "UK"])

        with c2:
            amount_usd = st.number_input("Amount (USD)", min_value=0.0, value=3200.0, step=50.0)
            amount_src = st.number_input("Amount (source currency)", min_value=0.0, value=3200.0, step=50.0)
            fee = st.number_input("Fee", min_value=0.0, value=12.5, step=1.0)
            exchange_rate = st.number_input("Exchange rate (src→dest)", min_value=0.0001, value=17.1, step=0.1)
            new_device = st.checkbox("New device", value=True)
            location_mismatch = st.checkbox("Location mismatch", value=True)

        with c3:
            ip_risk_score = st.slider("IP risk score", 0.0, 1.0, 0.82)
            account_age_days = st.number_input("Account age (days)", min_value=0, value=12)
            device_trust_score = st.slider("Device trust score", 0.0, 1.0, 0.21)
            chargeback_history_count = st.number_input("Chargeback history count", min_value=0, value=1)
            risk_score_internal = st.slider("Internal risk score", 0.0, 1.0, 0.70)
            txn_velocity_1h = st.number_input("Transactions in last 1h", min_value=0, value=4)
            txn_velocity_24h = st.number_input("Transactions in last 24h", min_value=0, value=9)
            corridor_risk = st.slider("Corridor risk", 0.0, 1.0, 0.55)

        submitted = st.form_submit_button("Score transaction", type="primary")

    if submitted:
        payload = {
            "transaction_id": transaction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "home_country": home_country,
            "source_currency": source_currency,
            "dest_currency": dest_currency,
            "channel": channel,
            "kyc_tier": kyc_tier,
            "ip_country": ip_country,
            "amount_src": amount_src,
            "amount_usd": amount_usd,
            "fee": fee,
            "exchange_rate_src_to_dest": exchange_rate,
            "new_device": new_device,
            "location_mismatch": location_mismatch,
            "ip_risk_score": ip_risk_score,
            "account_age_days": account_age_days,
            "device_trust_score": device_trust_score,
            "chargeback_history_count": chargeback_history_count,
            "risk_score_internal": risk_score_internal,
            "txn_velocity_1h": txn_velocity_1h,
            "txn_velocity_24h": txn_velocity_24h,
            "corridor_risk": corridor_risk,
        }
        try:
            result = api_post(api_url, "/score", payload)
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")
        else:
            proba = result["fraud_probability"]
            flagged = result["is_flagged"]

            col_gauge, col_summary = st.columns([1, 1])
            with col_gauge:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%"},
                    title={"text": "Fraud probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#C44E52" if flagged else "#55A868"},
                        "steps": [
                            {"range": [0, result["decision_threshold"] * 100], "color": "#eafaf1"},
                            {"range": [result["decision_threshold"] * 100, 100], "color": "#fdecea"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.8,
                            "value": result["decision_threshold"] * 100,
                        },
                    },
                ))
                fig.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
                st.plotly_chart(fig, width='stretch')

            with col_summary:
                st.metric("Decision", "🚩 FLAGGED FOR REVIEW" if flagged else "✅ Not flagged")
                st.metric("Model", result["model_version"])
                st.metric("Decision threshold", f"{result['decision_threshold']:.2f}")
                st.caption(f"Scored at {result['scored_at']}")

            st.markdown("#### Why this transaction was scored this way")
            reasons_df = pd.DataFrame(result["reasons"])
            reasons_df["abs_contribution"] = reasons_df["shap_contribution"].abs()
            reasons_df = reasons_df.sort_values("shap_contribution")

            fig2 = go.Figure(go.Bar(
                x=reasons_df["shap_contribution"],
                y=[f"{f} = {v}" for f, v in zip(reasons_df["feature"], reasons_df["value"])],
                orientation="h",
                marker_color=[
                    "#C44E52" if d == "increases_fraud_risk" else "#55A868"
                    for d in reasons_df["direction"]
                ],
            ))
            fig2.update_layout(
                height=280,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="SHAP contribution (→ higher = more fraud-like)",
            )
            st.plotly_chart(fig2, width='stretch')
            st.caption(
                "Red bars push the score toward fraud; green bars push it toward legitimate. "
                "Values shown are the transaction's actual (unscaled) field values."
            )

# ---------------------------------------------------------------------------
# Tab 2: batch scoring
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Score a batch of transactions")
    st.caption(
        "Upload a CSV with the raw transaction columns "
        f"({', '.join(REQUIRED_COLUMNS[:5])}, ...). Up to 500 rows per request."
    )

    uploaded = st.file_uploader("Transaction CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            batch_df = None

        if batch_df is not None:
            missing = [c for c in REQUIRED_COLUMNS if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                st.write(f"{len(batch_df)} transactions loaded.")
                st.dataframe(batch_df.head(10), width='stretch')

                if st.button("Score batch", type="primary"):
                    if len(batch_df) > 500:
                        st.warning("Only the first 500 rows will be scored (API limit).")
                        batch_df = batch_df.head(500)

                    records = json.loads(batch_df.to_json(orient="records", date_format="iso"))
                    try:
                        with st.spinner(f"Scoring {len(records)} transactions..."):
                            batch_result = api_post(
                                api_url, "/score/batch", {"transactions": records}, timeout=60.0
                            )
                    except Exception as exc:
                        st.error(f"Batch scoring failed: {exc}")
                    else:
                        results_df = pd.DataFrame(batch_result["results"])
                        results_df["reasons"] = results_df["reasons"].apply(
                            lambda rs: "; ".join(f"{r['feature']}={r['value']}" for r in rs[:3])
                        )

                        n_flagged = int(results_df["is_flagged"].sum())
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Transactions scored", len(results_df))
                        m2.metric("Flagged for review", n_flagged)
                        m3.metric("Flag rate", f"{n_flagged / len(results_df):.1%}")

                        st.dataframe(
                            results_df[
                                ["transaction_id", "fraud_probability", "is_flagged", "reasons"]
                            ].sort_values("fraud_probability", ascending=False),
                            width='stretch',
                        )

                        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download results as CSV",
                            data=csv_bytes,
                            file_name="novapay_batch_scores.csv",
                            mime="text/csv",
                        )

# ---------------------------------------------------------------------------
# Tab 3: model metadata
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Deployed model information")
    try:
        metadata = api_get(api_url, "/model/metadata")
    except Exception as exc:
        st.error(f"Could not load model metadata: {exc}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", metadata["best_model"])
        c2.metric("Decision threshold", f"{metadata['decision_threshold']:.2f}")
        c3.metric(
            "Recall uplift vs. rules baseline",
            f"{metadata['recall_uplift_vs_rules_baseline_pct']:.1f}%",
        )
        c4.metric(
            "Meets ≥15% requirement",
            "✅ Yes" if metadata["meets_min_recall_uplift_requirement"] else "❌ No",
        )

        st.markdown("#### Top SHAP features (global importance)")
        top_features = metadata.get("top_shap_features", [])
        if top_features:
            st.dataframe(
                pd.DataFrame({"rank": range(1, len(top_features) + 1), "feature": top_features}),
                width='stretch', hide_index=True,
            )

        with st.expander("Dataset summary"):
            st.json(metadata.get("dataset", {}))
        with st.expander("Library versions"):
            st.json(metadata.get("versions", {}))
        with st.expander("Full raw metadata"):
            st.json(metadata)

# ---------------------------------------------------------------------------
# Tab 4: drift monitoring (direct import, not via API — see module docstring)
# ---------------------------------------------------------------------------
with tab_monitor:
    st.subheader("Data & prediction drift")
    st.caption(
        "Compares a window of recently-scored transactions against the "
        "training-time reference distribution. Upload a CSV of scored "
        "transactions (must include a `fraud_probability` column, as "
        "produced by the Batch Scoring tab's download) to check for drift."
    )

    try:
        from novapay_fraud import config
        from novapay_fraud.monitoring.drift_check import check_drift
        monitoring_available = config.REFERENCE_DATA_PATH.exists()
    except Exception as exc:
        monitoring_available = False
        st.error(f"Monitoring module unavailable: {exc}")

    if monitoring_available:
        drift_upload = st.file_uploader("Recently-scored transactions CSV", type=["csv"], key="drift_upload")
        use_demo = st.checkbox("Use a synthetic demo window instead (perturbed reference sample)", value=drift_upload is None)

        if st.button("Run drift check", type="primary"):
            reference = pd.read_parquet(config.REFERENCE_DATA_PATH)
            if use_demo:
                current = reference.sample(min(300, len(reference)), random_state=7).copy()
                current["amount_usd"] = current["amount_usd"] * 1.4
                current["ip_risk_score"] = (current["ip_risk_score"] * 1.2).clip(0, 1)
                st.info("Using a synthetically perturbed sample of the reference window for this demo.")
            elif drift_upload is not None:
                current = pd.read_csv(drift_upload)
            else:
                st.warning("Upload a CSV or check the demo-window box.")
                current = None

            if current is not None:
                with st.spinner("Computing drift report..."):
                    result = check_drift(reference, current)

                m1, m2, m3 = st.columns(3)
                m1.metric("Drifted features", f"{result.drifted_feature_count} / {len(config.NUMERIC_FEATURES + config.BOOLEAN_FEATURES + config.CATEGORICAL_FEATURES)}")
                m2.metric("Prediction drift score", f"{result.prediction_drift_score:.3f}" if result.prediction_drift_score is not None else "n/a")
                m3.metric("Flag-rate change", f"{result.flag_rate_relative_change:+.0%}")

                if result.alerts:
                    for alert in result.alerts:
                        st.error(alert)
                else:
                    st.success("No drift alerts triggered.")

                if result.drifted_features:
                    st.markdown("**Drifted features:** " + ", ".join(result.drifted_features))

                with st.expander("Full drift report (JSON)"):
                    st.json(result.to_dict())
    else:
        st.warning(
            "Reference data not found. Run `python -m novapay_fraud.train` "
            "first to generate `artifacts/reference_data.parquet`."
        )
