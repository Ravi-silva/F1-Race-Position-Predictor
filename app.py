# -*- coding: utf-8 -*-
"""
F1 Race Position Predictor 2026 — app de demonstração (Streamlit).

O modelo é treinado com dados HISTÓRICOS (Kaggle, 1950-2024 — ver
notebook.ipynb). Este app usa esse modelo para prever corridas da
temporada 2026 ATUAL: o grid de pilotos, a equipe de cada um e a
nacionalidade são puxados automaticamente (não são mais escolhidos à mão),
e o grid/classificação de largada usados como ponto de partida vêm do
resultado REAL do último GP disputado (dados coletados da internet —
ver f1_2026_data.py). Você pode ajustar os controles para simular outros
cenários (ex.: "e se ele largasse em P3 em vez de P8?").

Rode com:  streamlit run app.py
Precisa do arquivo f1_model.pkl gerado pelo notebook.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from f1_2026_data import (
    CALENDAR_2026,
    CONSTRUCTOR_POINTS_2026,
    CONSTRUCTOR_REF,
    DRIVERS_2026,
    NEXT_ROUND,
    REFERENCE_DATE,
    REFERENCE_GP,
)

MODEL_PATH = Path(__file__).parent / "f1_model.pkl"
YEAR = 2026

BADGE = {
    "Podium":        ("#d4af37", "🏆 PÓDIO (P1–P3)"),
    "Points":        ("#4a90d9", "🔵 PONTOS (P4–P10)"),
    "Out of Points": ("#9aa0a6", "⚪ FORA DOS PONTOS (P11+)"),
}

DRIVERS_BY_NAME = {d["name"]: d for d in DRIVERS_2026}
CALENDAR_BY_LABEL = {
    f"Round {r['round']:02d} — {r['gp']}": r for r in CALENDAR_2026
}


@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


# ------------------------------------------------------------------ encoding
def encode_driver(enc, driver):
    """driverRef -> código. Estreantes sem histórico usam o proxy do piloto."""
    classes = enc.classes_
    ref = driver["driver_ref"]
    if ref in classes:
        return int(enc.transform([ref])[0]), None
    proxy = driver["driver_proxy"]
    note = (f"**{driver['name']}** estreou em 2025/2026 e não existe no "
            f"histórico 1950-2024 usado no treino — o modelo usa o piloto "
            f"**{proxy}** como referência aproximada só para essa variável "
            f"categórica (grid, quali e forma recente já usam os dados "
            f"reais do próprio {driver['name'].split()[-1]}).")
    return int(enc.transform([proxy])[0]), note


def encode_constructor(enc, team):
    """constructor_ref -> código. Equipes novas (Audi/Cadillac) usam proxy."""
    ref, note = CONSTRUCTOR_REF[team]
    return int(enc.transform([ref])[0]), note


def encode_nationality(enc, nationality):
    """nationality -> código, tolerando espaços/maiúsculas divergentes."""
    classes = list(enc.classes_)
    if nationality in classes:
        return int(enc.transform([nationality])[0])
    norm = {c.strip().lower(): c for c in classes}
    key = nationality.strip().lower()
    match = norm.get(key)
    if match is None:
        match = next((c for c in classes if key in c.strip().lower()), classes[0])
    return int(enc.transform([match])[0])


def build_row(bundle, inputs):
    """Monta o DataFrame de 1 linha na ordem exata de bundle['features']."""
    return pd.DataFrame([[inputs[f] for f in bundle["features"]]],
                        columns=bundle["features"])


def main():
    st.set_page_config(page_title="F1 Race Position Predictor 2026", page_icon="🏎️",
                       layout="wide")
    st.title("🏎️ F1 Race Position Predictor — Temporada 2026")
    st.caption("Previsão da faixa de resultado de um piloto **antes** da corrida "
               "— Podium / Points / Out of Points. Modelo treinado com dados "
               "históricos (1950-2024); pilotos, equipes e o resultado do "
               "último GP são os dados reais da temporada 2026 em andamento.")

    if not MODEL_PATH.exists():
        st.error("`f1_model.pkl` não encontrado. Rode o `notebook.ipynb` primeiro "
                 "(seção 13) para gerar o modelo.")
        st.stop()

    bundle = load_bundle()
    model = bundle["model"]
    enc = bundle["encoders"]
    labels = bundle["labels_order"]

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Variáveis de entrada")

        st.subheader("Piloto (grid oficial 2026)")
        driver_name = st.selectbox("Piloto", sorted(DRIVERS_BY_NAME),
                                    index=sorted(DRIVERS_BY_NAME).index("Max Verstappen")
                                    if "Max Verstappen" in DRIVERS_BY_NAME else 0)
        driver = DRIVERS_BY_NAME[driver_name]

        # equipe e nacionalidade não são mais escolhidas à mão — vêm do piloto
        st.markdown(f"🏁 **Equipe:** {driver['team']}")
        st.markdown(f"🌍 **Nacionalidade:** {driver['nationality']}")
        st.caption(f"Puxado automaticamente do grid 2026 (fonte: internet). "
                   f"{driver['season_points']} pts na temporada até o round "
                   f"{NEXT_ROUND - 1}.")

        st.subheader("Corrida a prever")
        race_label = st.selectbox("Etapa do calendário 2026", list(CALENDAR_BY_LABEL),
                                  index=[r["round"] for r in CALENDAR_2026].index(NEXT_ROUND))
        race = CALENDAR_BY_LABEL[race_label]
        st.caption(f"Ano fixo em **{YEAR}**. Etapa mais recente já disputada: "
                   f"{REFERENCE_GP} ({REFERENCE_DATE}).")

        st.subheader("Largada e classificação")
        st.caption("Pré-preenchido com o resultado REAL do último GP disputado "
                   f"por {driver_name.split()[-1]} ({REFERENCE_GP}). Ajuste os "
                   "sliders para simular outro cenário de largada.")
        grid = st.slider("Posição no grid", 1, 22, driver["grid"])
        quali_position = st.slider("Posição na classificação", 1, 22,
                                   driver["quali_position"])
        quali_gap = st.slider("Gap para a pole na quali (s)", 0.0, 3.0,
                              float(driver["quali_gap"]), 0.05)

        st.subheader("Forma e força")
        recent_form = st.slider(
            "Forma recente (posição no último GP disputado)",
            1.0, 22.0, float(driver["recent_result"]), 0.5)
        driver_experience = st.slider("Experiência (nº de corridas na carreira)",
                                      0, 450, driver["driver_experience"])
        cons_avg = CONSTRUCTOR_POINTS_2026[driver["team"]] / (NEXT_ROUND - 1)
        constructor_form = st.slider("Forma da equipe (pontos médios / últimas 5)",
                                     0.0, 44.0, round(cons_avg, 1), 1.0)
        constructor_season_pts = st.slider("Pontos médios da equipe na temporada",
                                           0.0, 44.0, round(cons_avg, 1), 1.0)

    # ------------------------------------------------------------------ encode
    driver_enc, driver_note = encode_driver(enc["driverRef"], driver)
    constructor_enc, constructor_note = encode_constructor(enc["constructor_ref"],
                                                            driver["team"])
    nationality_enc = encode_nationality(enc["driver_nationality"], driver["nationality"])
    circuit_enc = int(enc["circuitRef"].transform([race["circuit_ref"]])[0])

    notes = [n for n in (driver_note, constructor_note) if n]
    if notes:
        with st.expander("⚠️ Piloto/equipe estreante em 2026 — como o modelo lida com isso",
                          expanded=False):
            for n in notes:
                st.markdown(f"- {n}")

    inputs = {
        "grid": grid,
        "quali_position": quali_position,
        "quali_time_gap": quali_gap,
        "recent_form": recent_form,
        "driver_experience": driver_experience,
        "constructor_form": constructor_form,
        "constructor_season_pts": constructor_season_pts,
        "year": YEAR,
        "round": race["round"],
        "circuit_enc": circuit_enc,
        "driver_enc": driver_enc,
        "constructor_enc": constructor_enc,
        "nationality_enc": nationality_enc,
    }

    row = build_row(bundle, inputs)
    pred = model.predict(row)[0]
    proba = dict(zip(model.classes_, model.predict_proba(row)[0]))

    # ------------------------------------------------------------------ resultado
    st.markdown(f"### {driver_name} — {driver['team']} · {race['gp']} ({race['local']})")
    col1, col2 = st.columns([1, 1])

    with col1:
        color, text = BADGE[pred]
        st.markdown(
            f"""
            <div style="background:{color};padding:26px;border-radius:14px;
                        text-align:center;color:white;font-size:26px;font-weight:800;
                        letter-spacing:.5px;">{text}</div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.write("**Probabilidades:**")
        for lab in labels:
            st.progress(float(proba.get(lab, 0.0)),
                        text=f"{lab}: {proba.get(lab, 0.0):.0%}")

    with col2:
        # probabilidade de PÓDIO em função da posição de largada (varre o grid)
        grids = list(range(1, 23))
        rows = pd.concat([build_row(bundle, {**inputs, "grid": g,
                                             "quali_position": g}) for g in grids],
                         ignore_index=True)
        probs = model.predict_proba(rows)
        idx = {c: i for i, c in enumerate(model.classes_)}
        fig = go.Figure()
        for lab in labels:
            fig.add_trace(go.Scatter(x=grids, y=probs[:, idx[lab]], mode="lines+markers",
                                     name=lab, stackgroup="one"))
        fig.add_vline(x=grid, line_dash="dash", line_color="white")
        fig.update_layout(title="Probabilidade por posição de largada",
                          xaxis_title="grid", yaxis_title="probabilidade",
                          height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width='stretch')

    # ------------------------------------------------------------------ importâncias
    st.subheader("Feature importance do modelo")
    imp = (pd.Series(model.feature_importances_, index=bundle["features"])
           .sort_values())
    fig2 = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                            marker_color="#4a90d9"))
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="importância relativa (Gini)")
    st.plotly_chart(fig2, width='stretch')

    with st.expander("Ver vetor de entrada enviado ao modelo"):
        st.dataframe(row.T.rename(columns={0: "valor"}))

    st.caption(f"Modelo: RandomForest — acurácia de teste "
               f"{bundle['metrics']['rf_accuracy']:.1%} (treinado com dados de "
               "1950-2024). Grid/equipe/nacionalidade da temporada 2026 e o "
               f"resultado de referência ({REFERENCE_GP}) foram coletados da "
               "internet em 01/09/2026. O modelo só conhece o cenário "
               "pré-largada: não modela safety car, batidas, clima ou "
               "estratégia de pit stop.")


if __name__ == "__main__":
    main()
