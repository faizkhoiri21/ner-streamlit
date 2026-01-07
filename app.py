from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# === Bootstrap ===
load_dotenv()

try:
    from annotated_text import annotated_text
    HAS_ANNOTATED_TEXT = True
except Exception:
    HAS_ANNOTATED_TEXT = False


# === App Config ===
st.set_page_config(page_title="NER Inference UI", layout="wide")


@dataclass(frozen=True)
class AppConfig:
    timeout_sec: int = 60
    env_api_base_url: str = "NER_API_BASE_URL"
    default_api_base_url: str = ""


@dataclass(frozen=True)
class NerEntity:
    label: str
    text: str
    score: Optional[float]
    start: int
    end: int


LABEL_COLOR_MAP: Dict[str, str] = {
    "PER": "#ff4b4b",
    "ORG": "#ff8c00",
    "LOC": "#ffa500",
    "GPE": "#FFA39E",
    "EVENT": "#D4380D",
    "DATE": "#FFC069",
    "FAC": "#AD8B00",
    "LAW": "#D3F261",
    "MONEY": "#389E0D",
}
DEFAULT_HIGHLIGHT_COLOR = "#E5E7EB"


# === Pure Helpers ===
def normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def make_predict_endpoint(base_url: str) -> str:
    base = normalize_base_url(base_url)
    return f"{base}/predict" if base else ""


def parse_api_entities(payload: dict) -> pd.DataFrame:
    """
    Expected API shape:
      {"code":200,"message":"success","data":{"entities":[...],"count":1}}
    Returns a normalized DataFrame with cols: label,text,score,start,end
    """
    data = payload.get("data") or {}
    entities = data.get("entities") or []
    df = pd.DataFrame(entities)

    if df.empty:
        return pd.DataFrame(columns=["label", "text", "score", "start", "end"])

    # ensure columns exist
    for col in ["label", "text", "score", "start", "end"]:
        if col not in df.columns:
            df[col] = None

    df["label"] = df["label"].astype(str).str.upper()
    df["text"] = df["text"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")

    return df[["label", "text", "score", "start", "end"]]


def choose_longest_non_overlapping_spans(df: pd.DataFrame) -> pd.DataFrame:
    """
    For highlight display only:
    If spans overlap, keep the longer span (more specific).
    Output is non-overlapping spans.
    """
    if df.empty:
        return df

    d = df.dropna(subset=["start", "end"]).copy()
    if d.empty:
        return d

    d["start"] = d["start"].astype(int)
    d["end"] = d["end"].astype(int)
    d["span_len"] = (d["end"] - d["start"]).astype(int)

    # start asc, length desc
    d = d.sort_values(["start", "span_len", "end"], ascending=[True, False, False])

    kept_rows = []
    current_end = -1

    for _, row in d.iterrows():
        s, e = int(row["start"]), int(row["end"])
        if s >= current_end:
            kept_rows.append(row)
            current_end = e
            continue

        last = kept_rows[-1]
        last_len = int(last["end"]) - int(last["start"])
        cur_len = e - s
        if cur_len > last_len:
            kept_rows[-1] = row
            current_end = e

    out = pd.DataFrame(kept_rows).drop(columns=["span_len"], errors="ignore")
    return out


def compute_summary_metrics(df: pd.DataFrame) -> Tuple[int, int, Optional[float]]:
    total_entities = int(len(df))
    unique_labels = int(df["label"].nunique()) if total_entities else 0
    avg_conf = float(df["score"].mean()) if total_entities and df["score"].notna().any() else None
    return total_entities, unique_labels, avg_conf


def build_annotated_parts(
    text: str,
    df: pd.DataFrame,
    label_colors: Dict[str, str],
    default_color: str = DEFAULT_HIGHLIGHT_COLOR,
) -> List[object]:
    """
    Convert spans into annotated_text parts:
      ["prefix", ("entity", "LABEL", "#hex"), "suffix", ...]
    """
    if df.empty:
        return [text]

    d = choose_longest_non_overlapping_spans(df).dropna(subset=["start", "end"]).copy()
    if d.empty:
        return [text]

    d["start"] = d["start"].astype(int)
    d["end"] = d["end"].astype(int)
    d = d.sort_values(["start", "end"])

    parts: List[object] = []
    cursor = 0

    for _, row in d.iterrows():
        s, e = int(row["start"]), int(row["end"])
        if s < cursor:
            continue
        if cursor < s:
            parts.append(text[cursor:s])

        label = str(row["label"]).upper()
        color = label_colors.get(label, default_color)
        parts.append((text[s:e], label, color))
        cursor = e

    if cursor < len(text):
        parts.append(text[cursor:])

    return parts


# === IO / API ===
def request_ner(api_url: str, text: str, timeout_sec: int) -> dict:
    resp = requests.post(api_url, json={"text": text}, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.json()


def assert_api_success(payload: dict) -> None:
    code = int(payload.get("code", 200))
    if code != 200:
        msg = payload.get("message", "unknown error")
        raise RuntimeError(f"API returned code={code}: {msg}")


# === UI Components ===
def render_sidebar_settings(cfg: AppConfig) -> str:
    st.sidebar.header("Settings")

    default_base = os.getenv(cfg.env_api_base_url, cfg.default_api_base_url)
    base_url = st.sidebar.text_input(
        "API Base URL (without /predict)",
        value=default_base,
    )

    endpoint = make_predict_endpoint(base_url)
    if endpoint:
        st.sidebar.caption(f"Endpoint: {endpoint}")
    else:
        st.sidebar.warning("Please enter the API Base URL.")

    with st.sidebar.expander("Label Color Legend", expanded=False):
        render_label_legend(LABEL_COLOR_MAP)

    return endpoint


def render_label_legend(label_colors: Dict[str, str]) -> None:
    items = []
    for label, color in label_colors.items():
        items.append(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin:6px 0;">
              <div style="width:14px; height:14px; border-radius:3px;
                          background:{color};
                          border:1px solid rgba(0,0,0,0.15);"></div>
              <div style="font-size:14px;"><b>{label}</b></div>
            </div>
            """
        )
    st.markdown("".join(items), unsafe_allow_html=True)


def render_kpi_summary(df: pd.DataFrame) -> None:
    total, unique, avg_conf = compute_summary_metrics(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Entities", total)
    c2.metric("Unique Labels", unique)
    c3.metric("Avg Confidence", "-" if avg_conf is None else f"{avg_conf:.3f}")


def render_highlight(text: str, df: pd.DataFrame) -> None:
    if not HAS_ANNOTATED_TEXT:
        st.info("Install `st-annotated-text` to enable highlight rendering.")
        st.write(text)
        return

    parts = build_annotated_parts(text, df, LABEL_COLOR_MAP, DEFAULT_HIGHLIGHT_COLOR)
    annotated_text(*parts)


# === Main ===
def main() -> None:
    cfg = AppConfig()

    st.title("Named Entity Recognition")
    api_endpoint = render_sidebar_settings(cfg)

    default_text = "Jokowi merupakan seorang warga Solo."
    user_text = st.text_area("Input Text", value=default_text, height=200)

    run = st.button("Run NER", type="primary")

    if not run:
        return

    if not api_endpoint:
        st.error("API Base URL is not set.")
        return

    if not user_text.strip():
        st.warning("Input text is empty.")
        return

    with st.spinner("Calling /predict ..."):
        try:
            payload = request_ner(api_endpoint, user_text, timeout_sec=cfg.timeout_sec)
            assert_api_success(payload)
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
            return
        except Exception as e:
            st.error(str(e))
            return

    df_entities = parse_api_entities(payload)

    st.subheader("Summary")
    render_kpi_summary(df_entities)

    st.subheader("Highlights")
    render_highlight(user_text, df_entities)

    st.subheader("Entities")
    st.dataframe(df_entities, use_container_width=True)


if __name__ == "__main__":
    main()