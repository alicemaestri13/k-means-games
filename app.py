"""
=============================================================================
K-Means Explorer: Meccanica Operativa sul Dataset Videogiochi
=============================================================================
Applicazione didattica per la laurea magistrale.
Mostra il funzionamento INTERNO di K-means passo per passo.

Autore: generato con Claude (Anthropic)
Requisiti: streamlit, numpy, pandas, plotly, scikit-learn
Installazione: pip install streamlit numpy pandas plotly scikit-learn
Avvio:        streamlit run app.py
=============================================================================
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURAZIONE PAGINA
#    Deve essere la PRIMA chiamata Streamlit del file.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-Means Explorer · Videogiochi",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CSS CUSTOM — tema scuro stile "dashboard analitica premium"
#    Usiamo CSS iniettato via st.markdown per sovrascrivere il tema di default.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Font Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

/* Root variables */
:root {
    --bg-primary: #0d0f1a;
    --bg-card: #151825;
    --bg-sidebar: #10121e;
    --accent-1: #7c5cfc;   /* viola elettrico */
    --accent-2: #00d4aa;   /* teal neon */
    --accent-3: #ff6b6b;   /* coral */
    --accent-4: #ffd93d;   /* giallo */
    --accent-5: #6bcbff;   /* azzurro */
    --text-primary: #e8eaf6;
    --text-secondary: #8892b0;
    --border: rgba(124,92,252,0.25);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border);
}

/* Titoli */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: var(--text-primary) !important;
}

/* Metriche */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
}

/* Expander */
details {
    background: var(--bg-card);
    border: 1px solid var(--border) !important;
    border-radius: 10px;
}

/* Tabelle pandas */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
}

/* Bottone */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-1), #5a3de0);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
}

/* Formula math box */
.math-box {
    background: var(--bg-card);
    border-left: 4px solid var(--accent-1);
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-family: 'Space Mono', monospace;
}

/* Badge iterazione */
.badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

/* Info box */
.info-box {
    background: rgba(0, 212, 170, 0.08);
    border: 1px solid rgba(0, 212, 170, 0.3);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.92rem;
    color: var(--text-primary);
}

.warning-box {
    background: rgba(255, 107, 107, 0.08);
    border: 1px solid rgba(255, 107, 107, 0.3);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.92rem;
    color: var(--text-primary);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GENERAZIONE DATASET SINTETICO — Tema Videogiochi
#    Usiamo numpy con seed fisso per garantire riproducibilità.
#    Quattro archetipi chiari ma NON codificati nell'algoritmo:
#       - Consumo Intensivo:   alte ore, bassa completion (es. titoli con forte componente online)
#       - Consumo Esplorativo: ore medie, bassa completion (es. open world abbandonati a metà)
#       - Consumo Rapido:      poche ore, alta completion (titoli brevi e lineari)
#       - Consumo Narrativo:   ore medie, alta completion (forte coinvolgimento narrativo)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data  # Cache per non ricalcolare ad ogni interazione
def generate_dataset(seed: int = 42, n_per_cluster: int = 40) -> pd.DataFrame:
    """
    Genera un dataset sintetico di videogiochi con 4 archetipi latenti.
    Restituisce un DataFrame con colonne: nome, playtime, completion, archetype.
    """
    rng = np.random.default_rng(seed)

    # Centri (mu) e deviazioni standard (sigma) dei 4 archetipi
    clusters_params = [
        # (mu_playtime, sigma_pt, mu_completion, sigma_cr, label)
        (85,  18,  28,  8,  "RPG"),
        (35,  12,  22,  7,  "FPS"),
        (8,   3,   82,  9,  "Casual"),
        (42,  15,  74,  10, "Adventure"),
    ]

    nomi_prefissi = [
        "Shadow", "Cyber", "Neon", "Dark", "Ultra", "Pixel", "Hyper",
        "Astro", "Turbo", "Mega", "Vortex", "Storm", "Echo", "Void",
        "Blaze", "Frost", "Iron", "Quantum", "Nova", "Drift",
    ]
    nomi_suffissi = [
        "Quest", "Strike", "Rush", "Run", "Wars", "Legends", "Rising",
        "Chronicles", "Odyssey", "Arena", "Siege", "Clash", "Escape",
        "Origins", "Reborn", "Force", "Drift", "Realm", "Nexus", "Core",
    ]

    rows = []
    for mu_pt, sig_pt, mu_cr, sig_cr, label in clusters_params:
        pt_vals = rng.normal(mu_pt, sig_pt, n_per_cluster).clip(1, 150)
        cr_vals = rng.normal(mu_cr, sig_cr, n_per_cluster).clip(1, 99)
        for pt, cr in zip(pt_vals, cr_vals):
            nome = f"{rng.choice(nomi_prefissi)} {rng.choice(nomi_suffissi)}"
            rows.append({
                "Videogioco": nome,
                "Ore di Gioco (media)": round(float(pt), 1),
                "Completion Rate (%)": round(float(cr), 1),
                "Archetipo Reale": label,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. IMPLEMENTAZIONE K-MEANS PASSO-PASSO
#    NON usiamo sklearn per il clustering: lo implementiamo da zero
#    per poter mostrare ogni singola iterazione.
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distanza Euclidea tra due punti nello spazio 2D.
    Formula: d(a,b) = sqrt( (a1-b1)^2 + (a2-b2)^2 )
    """
    return float(np.sqrt(np.sum((a - b) ** 2)))


def run_kmeans_steps(X: np.ndarray, k: int, seed: int, max_iter: int = 10):
    """
    Esegue K-means dal principio e salva LO STATO di ogni iterazione.

    Parametri
    ---------
    X        : array (n, 2) — feature normalizzate
    k        : numero di cluster
    seed     : seed per l'inizializzazione random dei centroidi
    max_iter : numero massimo di iterazioni da registrare

    Ritorna
    -------
    steps : lista di dict, uno per iterazione (incluso step 0 = init)
        Ogni dict contiene:
          - 'centroids'   : array (k, 2) — posizione centroidi
          - 'labels'      : array (n,)   — cluster assegnato a ciascun punto
          - 'distances'   : array (n, k) — distanze da ogni punto a ogni centroide
          - 'converged'   : bool         — True se i centroidi non si sono mossi
          - 'inertia'     : float        — somma delle distanze quadratiche (WCSS)
    """
    rng = np.random.default_rng(seed)

    # ── Inizializzazione: k centroidi scelti a caso tra i punti ──────────────
    init_indices = rng.choice(len(X), size=k, replace=False)
    centroids = X[init_indices].copy()

    steps = []

    # Step 0: stato iniziale (solo centroidi, nessuna assegnazione ancora)
    # Calcoliamo già le distanze per mostrare il punto di partenza
    distances = np.array([[euclidean_distance(x, c) for c in centroids] for x in X])
    labels = np.argmin(distances, axis=1)
    inertia = float(np.sum(np.min(distances, axis=1) ** 2))

    steps.append({
        "centroids": centroids.copy(),
        "labels": labels.copy(),
        "distances": distances.copy(),
        "converged": False,
        "inertia": inertia,
        "description": "Inizializzazione: centroidi posizionati casualmente.",
    })

    # ── Iterazioni ──────────────────────────────────────────────────────────
    for i in range(1, max_iter + 1):
        prev_centroids = centroids.copy()

        # Passo A: ASSEGNAZIONE — ogni punto va al centroide più vicino
        distances = np.array([[euclidean_distance(x, c) for c in centroids] for x in X])
        labels = np.argmin(distances, axis=1)

        # Passo B: AGGIORNAMENTO — nuovo centroide = media dei punti assegnati
        new_centroids = np.zeros_like(centroids)
        for cluster_idx in range(k):
            members = X[labels == cluster_idx]
            if len(members) > 0:
                new_centroids[cluster_idx] = members.mean(axis=0)
            else:
                # Cluster vuoto: tieni il centroide precedente
                new_centroids[cluster_idx] = prev_centroids[cluster_idx]

        centroids = new_centroids
        inertia = float(np.sum(np.min(distances, axis=1) ** 2))

        # Convergenza: centroidi identici a prima? (tolleranza numerica)
        converged = np.allclose(prev_centroids, centroids, atol=1e-6)

        steps.append({
            "centroids": centroids.copy(),
            "labels": labels.copy(),
            "distances": distances.copy(),
            "converged": converged,
            "inertia": inertia,
            "description": (
                "✅ **Convergenza raggiunta!** I centroidi non si spostano più."
                if converged
                else f"Iterazione {i}: centroidi aggiornati, si continua."
            ),
        })

        if converged:
            break

    return steps


# ─────────────────────────────────────────────────────────────────────────────
# 4. PALETTE COLORI PER I CLUSTER
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_COLORS = ["#7c5cfc", "#00d4aa", "#ff6b6b", "#ffd93d", "#6bcbff"]
CLUSTER_NAMES_LABELS = ["Cluster A", "Cluster B", "Cluster C", "Cluster D", "Cluster E"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. FUNZIONE GRAFICO PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

def build_scatter_plot(
    df: pd.DataFrame,
    X_orig: np.ndarray,   # coordinate originali (non-scalate) per i tooltip
    X_scaled: np.ndarray, # coordinate scalate per posizionamento
    step_data: dict,
    k: int,
    scaler: StandardScaler,
) -> go.Figure:
    """
    Costruisce il grafico Plotly con:
      - scatter dei punti colorati per cluster
      - centroidi come simbolo "x" grandi
      - tooltip con nome del videogioco e feature reali
    """
    labels = step_data["labels"]
    centroids_scaled = step_data["centroids"]
    # De-scala i centroidi per mostrarli nello spazio originale
    centroids_orig = scaler.inverse_transform(centroids_scaled)

    fig = go.Figure()

    # ── Scatter per ogni cluster ───────────────────────────────────────────
    for c in range(k):
        mask = labels == c
        fig.add_trace(go.Scatter(
            x=X_orig[mask, 0],
            y=X_orig[mask, 1],
            mode="markers",
            name=CLUSTER_NAMES_LABELS[c],
            marker=dict(
                color=CLUSTER_COLORS[c],
                size=9,
                opacity=0.82,
                line=dict(width=0.8, color="rgba(255,255,255,0.3)"),
            ),
            text=[
                f"<b>{row['Videogioco']}</b><br>"
                f"Ore: {row['Ore di Gioco (media)']:.1f}h<br>"
                f"Completion: {row['Completion Rate (%)']:.1f}%<br>"
                f"Archetipo reale: {row['Archetipo Reale']}"
                for _, row in df[mask].iterrows()
            ],
            hovertemplate="%{text}<extra></extra>",
        ))

    # ── Centroidi come X grandi ────────────────────────────────────────────
    for c in range(k):
        fig.add_trace(go.Scatter(
            x=[centroids_orig[c, 0]],
            y=[centroids_orig[c, 1]],
            mode="markers+text",
            name=f"Centroide {CLUSTER_NAMES_LABELS[c]}",
            marker=dict(
                symbol="x",
                size=20,
                color=CLUSTER_COLORS[c],
                line=dict(width=4, color="white"),
            ),
            text=[f"C{c+1}"],
            textposition="top center",
            textfont=dict(size=13, color="white", family="Space Mono"),
            hovertemplate=(
                f"<b>Centroide {c+1}</b><br>"
                f"Ore: {centroids_orig[c, 0]:.2f}h<br>"
                f"Completion: {centroids_orig[c, 1]:.2f}%<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#10121e",
        font=dict(family="DM Sans", color="#e8eaf6"),
        legend=dict(
            bgcolor="rgba(21,24,37,0.9)",
            bordercolor="rgba(124,92,252,0.3)",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            title="Ore di Gioco Medie per Utente",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
            title_font=dict(size=13),
        ),
        yaxis=dict(
            title="Completion Rate (%)",
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.1)",
            title_font=dict(size=13),
        ),
        margin=dict(l=20, r=20, t=30, b=20),
        height=500,
        hoverlabel=dict(
            bgcolor="#1a1d2e",
            bordercolor="#7c5cfc",
            font=dict(family="DM Sans", size=12),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. FUNZIONE: TABELLA "DIETRO LE QUINTE"
#    Mostra il calcolo esplicito delle distanze per i primi N videogiochi.
# ─────────────────────────────────────────────────────────────────────────────

def build_distance_table(
    df: pd.DataFrame,
    step_data: dict,
    k: int,
    n_samples: int = 5,
) -> pd.DataFrame:
    """
    Costruisce un DataFrame che mostra, per i primi n_samples videogiochi:
      - Le distanze dal centroide 1, 2, … k
      - Il cluster assegnato (= indice del centroide con distanza minima)
    """
    distances = step_data["distances"]  # shape (n, k)
    labels = step_data["labels"]

    rows = []
    for i in range(min(n_samples, len(df))):
        row = {
            "Videogioco": df.iloc[i]["Videogioco"],
            "Ore": df.iloc[i]["Ore di Gioco (media)"],
            "Completion %": df.iloc[i]["Completion Rate (%)"],
        }
        for c in range(k):
            row[f"d(C{c+1})"] = round(distances[i, c], 4)
        row["Cluster Assegnato"] = f"C{labels[i]+1} ← min"
        rows.append(row)

    return pd.DataFrame(rows)



# ═════════════════════════════════════════════════════════════════════════════
#                              LAYOUT PRINCIPALE
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2.4rem;'>🎮</div>
        <div style='font-family: Space Mono, monospace; font-size: 1.05rem;
                    color: #7c5cfc; letter-spacing:2px; font-weight:700;'>
            K-MEANS EXPLORER
        </div>
        <div style='font-size:0.78rem; color:#8892b0; margin-top:4px;'>
            Videogiochi · Didattica ML
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Configurazione")

    # Slider: numero di cluster k
    k = st.slider(
        "Numero di Cluster (k)",
        min_value=2, max_value=5, value=4,
        help="K è il solo iperparametro di K-means: scegli quanti gruppi cercare."
    )

    # Slider: seed casuale (influenza l'inizializzazione)
    seed = st.slider(
        "Seed casuale (inizializzazione)",
        min_value=0, max_value=99, value=42,
        help="Cambia il seed per vedere come una diversa inizializzazione influisce sulla convergenza."
    )

    # Calcola i passi PRIMA di mostrare lo slider dell'iterazione,
    # così sappiamo quante iterazioni ci sono effettivamente.
    df_full = generate_dataset(seed=42, n_per_cluster=40)
    X_raw = df_full[["Ore di Gioco (media)", "Completion Rate (%)"]].values

    # Normalizzazione Z-score — necessaria per K-means (feature su scale diverse!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Esegui K-means custom e memorizza tutti gli step
    steps = run_kmeans_steps(X_scaled, k=k, seed=seed, max_iter=10)
    max_step = len(steps) - 1  # indice dell'ultimo step disponibile

    st.markdown("---")

    # Mini-guida nella sidebar
    st.markdown("""
    <div style='font-size:0.8rem; color:#8892b0; line-height:1.6;'>
    <b style='color:#7c5cfc;'>Come leggere l'app</b><br>
    1. Imposta <b>k</b> e il <b>seed</b><br>
    2. Scorri l'iterazione nella pagina principale<br>
    3. Osserva le <b>X</b> (poli/centroidi) muoversi<br>
    4. Leggi la tabella distanze<br>
    5. Confronta seed diversi 🔬
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE — Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='margin-bottom:0; font-size:1.9rem;'>
    K-Means: Meccanica Operativa
    <span style='color:#7c5cfc;'>·</span>
    Dataset Videogiochi 🎮
</h1>
<p style='color:#8892b0; margin-top:6px; font-size:0.95rem; font-family: DM Sans, sans-serif;'>
    Visualizzazione passo-passo del processo di clustering non supervisionato.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SLIDER ITERAZIONE — nella main page per massima visibilità
# ─────────────────────────────────────────────────────────────────────────────
col_sl_label, col_sl_widget = st.columns([1, 4])
with col_sl_label:
    st.markdown("""
    <div style='padding-top:30px; font-family: Space Mono, monospace;
                font-size:0.88rem; color:#7c5cfc; font-weight:700; letter-spacing:1px;'>
        ▶ ITERAZIONE
    </div>
    """, unsafe_allow_html=True)
with col_sl_widget:
    step_idx = st.slider(
        "Seleziona lo step",
        min_value=0, max_value=max_step, value=0,
        format="Step %d",
        label_visibility="collapsed",
        help="Step 0 = partizione iniziale (poli casuali). Aumenta per seguire lo spostamento dei centroidi.",
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE A — Status dell'iterazione corrente
# ─────────────────────────────────────────────────────────────────────────────
current_step = steps[step_idx]

# Badge iterazione + stato convergenza
conv_icon = "✅" if current_step["converged"] else "🔄"
badge_text = "INIZIALIZZAZIONE" if step_idx == 0 else f"ITERAZIONE {step_idx}"

st.markdown(f"""
<div style='display:flex; align-items:center; gap:12px; margin: 18px 0 10px 0;'>
    <span class='badge'>{badge_text}</span>
    <span style='color:#8892b0; font-size:0.9rem;'>
        {conv_icon} {current_step["description"]}
    </span>
</div>
""", unsafe_allow_html=True)

# Metriche rapide
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Cluster (k)", k)
col_m2.metric("Step corrente", f"{step_idx} / {max_step}")
col_m3.metric("Devianza Within (WCSS)", f"{current_step['inertia']:.2f}")
col_m4.metric("Convergenza", "Sì ✅" if current_step["converged"] else "No 🔄")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE B — Grafico interattivo + Spiegazione step corrente
# ─────────────────────────────────────────────────────────────────────────────
col_chart, col_explain = st.columns([2.2, 1])

with col_chart:
    st.markdown("#### 📊 Scatter Plot — Posizione Cluster & Centroidi")
    fig = build_scatter_plot(
        df=df_full,
        X_orig=X_raw,
        X_scaled=X_scaled,
        step_data=current_step,
        k=k,
        scaler=scaler,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_explain:
    st.markdown("#### 🔬 Meccanica di questo step")

    if step_idx == 0:
        st.markdown("""
        <div class='info-box'>
        <b>Step 0 — Partizione iniziale</b><br><br>
        Si scelgono <b>k poli iniziali</b> (semi iniziali) casualmente tra le unità
        del dataset. Si costruisce la <b>partizione iniziale</b>: ciascuna unità
        viene allocata al cluster il cui polo risulta il più vicino
        (distanza euclidea minima).
        </div>
        """, unsafe_allow_html=True)
    else:
        _conv_msg = (
            "<b style='color:#00d4aa;'>Nessuna modifica dei centroidi rispetto all'iterazione "
            "precedente: la <b>regola d'arresto</b> è soddisfatta — convergenza raggiunta.</b>"
            if current_step['converged']
            else "I centroidi si sono spostati → si ripete il ciclo."
        )
        st.markdown(f"""
        <div class='info-box'>
        <b>Step {step_idx} — Procedura iterativa</b><br><br>
        <b>1 — Calcolo centroidi:</b><br>
        Per ogni cluster si ricalcola il centroide come media delle unità assegnate.<br><br>
        <b>2 — Riassegnazione:</b><br>
        Per ogni unità si calcola la <b>distanza euclidea dai centroidi dei k cluster</b>.
        Se la distanza minima non è quella dal centroide del cluster di appartenenza,
        l'unità viene <b>riassegnata</b> al cluster più vicino
        (minimizzazione della <b>devianza within</b>).<br><br>
        {_conv_msg}
        </div>
        """, unsafe_allow_html=True)

    # Tabella posizioni centroidi correnti
    st.markdown("**📍 Poli / Centroidi (spazio originale)**")
    centroids_orig = scaler.inverse_transform(current_step["centroids"])
    df_centroids = pd.DataFrame({
        "Cluster": [f"C{i+1}" for i in range(k)],
        "Ore": centroids_orig[:, 0].round(2),
        "Completion %": centroids_orig[:, 1].round(2),
    })
    st.dataframe(df_centroids, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE C — Tabella "Dietro le Quinte" (il calcolo esplicito)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🔢 Tabella Calcolo Distanze — 'Dietro le Quinte' (primi 5 videogiochi)", expanded=True):

    st.markdown("""
    Questa tabella mostra il calcolo esplicito che K-means esegue **internamente**
    ad ogni iterazione per i primi 5 videogiochi del dataset.
    Per ogni punto vengono calcolate le distanze (nello **spazio normalizzato**)
    da tutti i k centroidi. Il cluster assegnato è quello con la **distanza minima**.
    """)

    df_dist = build_distance_table(df_full, current_step, k, n_samples=5)

    # Highlight la colonna del cluster assegnato — costruisce stile
    def highlight_min_distance(row):
        """Evidenzia in verde la distanza minima per ogni riga."""
        d_cols = [f"d(C{c+1})" for c in range(k)]
        styles = [""] * len(row)
        min_val = row[d_cols].min()
        for ci, col in enumerate(row.index):
            if col in d_cols and row[col] == min_val:
                styles[ci] = "background-color: rgba(0,212,170,0.2); color: #00d4aa; font-weight: bold;"
        return styles

    styled_df = df_dist.style.apply(highlight_min_distance, axis=1).format(
        {f"d(C{c+1})": "{:.4f}" for c in range(k)}
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("""
    > 💡 **Lettura:** Le distanze sono calcolate sulle **feature normalizzate** (Z-score).
    > La colonna evidenziata in <span style='color:#00d4aa;'>verde</span> è la distanza minima → determina l'assegnazione al cluster.
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE D — Dataset completo (espandibile)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📋 Dataset completo (160 videogiochi simulati)"):
    df_display = df_full.copy()
    df_display["Cluster Assegnato (step corrente)"] = [
        f"C{l+1}" for l in current_step["labels"]
    ]
    st.dataframe(df_display, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# SEZIONE E — Interpretazione dei cluster trovati
#   Mostra solo all'ultimo step (o quando converge) per non distrarre
#   durante la fase iterativa. I centroidi vengono letti nello spazio
#   originale e classificati automaticamente per quadrante.
# ─────────────────────────────────────────────────────────────────────────────

# Calcola la posizione mediana del dataset per definire i quadranti
ore_median     = float(np.median(X_raw[:, 0]))
comp_median    = float(np.median(X_raw[:, 1]))

def interpret_centroid(ore: float, comp: float, ore_med: float, comp_med: float) -> dict:
    """
    Classifica un centroide in uno dei 4 profili di consumo in base al quadrante
    (ore alte/basse × completion alta/bassa) rispetto alle mediane del dataset.
    Le etichette descrivono COME il gioco viene consumato dagli utenti,
    non il genere del gioco — coerente con le feature del dataset (medie aggregate).
    """
    high_ore  = ore  >= ore_med
    high_comp = comp >= comp_med

    if not high_ore and high_comp:
        return {
            "label": "Consumo Rapido",
            "emoji": "⚡",
            "color": "#7c5cfc",
            "ore_desc": f"poche ore investite ({ore:.0f}h in media)",
            "comp_desc": f"completion altissima ({comp:.0f}%)",
            "spiegazione": (
                "Titoli consumati in modo veloce e completo. "
                "L'utente medio arriva ai titoli di coda senza abbandonare — "
                "segnale di un'esperienza breve ma soddisfacente."
            ),
        }
    elif high_ore and high_comp:
        return {
            "label": "Consumo Narrativo",
            "emoji": "📖",
            "color": "#00d4aa",
            "ore_desc": f"ore elevate ({ore:.0f}h in media)",
            "comp_desc": f"completion medio-alta ({comp:.0f}%)",
            "spiegazione": (
                "L'utente investe tempo e porta a termine la storia. "
                "Alto coinvolgimento narrativo: il gioco trattiene e motiva "
                "a raggiungere il finale."
            ),
        }
    elif high_ore and not high_comp:
        return {
            "label": "Consumo Intensivo",
            "emoji": "🔥",
            "color": "#ff6b6b",
            "ore_desc": f"molte ore investite ({ore:.0f}h in media)",
            "comp_desc": f"completion bassa ({comp:.0f}%)",
            "spiegazione": (
                "L'utente gioca a lungo ma raramente completa la storia principale. "
                "Tipico di titoli con forte componente online o sandbox: "
                "il loop di gioco è il prodotto, non la narrativa."
            ),
        }
    else:  # not high_ore and not high_comp
        return {
            "label": "Consumo Esplorativo",
            "emoji": "🧭",
            "color": "#ffd93d",
            "ore_desc": f"ore contenute ({ore:.0f}h in media)",
            "comp_desc": f"completion bassa ({comp:.0f}%)",
            "spiegazione": (
                "L'utente esplora il gioco senza cercare il finale. "
                "Basso investimento di tempo e bassa completion suggeriscono "
                "un approccio libero, non orientato all'obiettivo."
            ),
        }

# Mostra la sezione solo all'ultimo step disponibile o a convergenza
is_last_step = (step_idx == max_step) or current_step["converged"]

st.markdown("---")
st.markdown("#### 🧩 Profili di consumo identificati")

if not is_last_step:
    st.markdown("""
    <div class='warning-box'>
    🔄 L'algoritmo è ancora in esecuzione. Porta lo slider all'ultimo step
    per leggere l'interpretazione dei cluster finali.
    </div>
    """, unsafe_allow_html=True)
else:
    centroids_final = scaler.inverse_transform(current_step["centroids"])

    # Costruisce una card per ogni cluster
    cols_interp = st.columns(k)
    for c in range(k):
        ore_c  = centroids_final[c, 0]
        comp_c = centroids_final[c, 1]
        interp = interpret_centroid(ore_c, comp_c, ore_median, comp_median)
        n_members = int(np.sum(current_step["labels"] == c))

        with cols_interp[c]:
            st.markdown(f"""
            <div style='
                background: var(--bg-card);
                border: 1px solid {interp["color"]}55;
                border-top: 3px solid {interp["color"]};
                border-radius: 10px;
                padding: 16px 14px;
                height: 100%;
            '>
                <div style='font-size:1.6rem; margin-bottom:6px;'>{interp["emoji"]}</div>
                <div style='font-family: Space Mono, monospace; font-size:0.75rem;
                            color:{interp["color"]}; font-weight:700; letter-spacing:1px;
                            margin-bottom:4px;'>C{c+1}</div>
                <div style='font-size:1rem; font-weight:700; color:#e8eaf6;
                            margin-bottom:10px;'>{interp["label"]}</div>
                <div style='font-size:0.8rem; color:#8892b0; line-height:1.7;'>
                    🕹️ {interp["ore_desc"]}<br>
                    ✅ {interp["comp_desc"]}<br>
                    👥 {n_members} titoli in questo cluster
                </div>
                <div style='margin-top:10px; font-size:0.78rem; color:#aab0c6;
                            line-height:1.6; border-top:1px solid rgba(255,255,255,0.07);
                            padding-top:10px;'>
                    {interp["spiegazione"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Nota metodologica finale
    st.markdown("""
    <div class='info-box' style='margin-top:18px;'>
    💡 <b>Nota metodologica:</b> l'algoritmo non ha mai "visto" questi profili.
    Li ha <b>ricostruiti autonomamente</b> minimizzando la devianza within —
    calcolando distanze euclidee e spostando i centroidi iterazione dopo iterazione.
    Le feature (ore di gioco medie e completion rate) descrivono <b>come i titoli
    vengono consumati dagli utenti</b>: i cluster emergono da pattern di consumo,
    non da categorie predefinite. L'interpretazione è nostra — K-Means consegna solo i numeri.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:0.8rem; color:#8892b0; padding:10px 0 20px 0;
            font-family: Space Mono, monospace;'>
    K-Means Explorer · IULM Milano · Laurea Magistrale AI for Business and Society<br>
    <span style='color:#7c5cfc;'>numpy · pandas · plotly · streamlit</span>
</div>
""", unsafe_allow_html=True)
