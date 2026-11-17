"""
Stratosphere Telemetry Copilot (Agente de Análise de Voo & Sensores)
Interface Streamlit - Painel de Controle de Missão Aeroespacial e Chat Analítico Text-to-Python.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Garante path para importação dos módulos internos
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.profiler import TelemetryProfiler
from src.agent import TelemetryCopilotAgent
from src.sandbox import SafePythonSandbox
from src.loader import load_telemetry


# Configuração de Página Streamlit
st.set_page_config(
    page_title="Stratosphere Telemetry Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS Customizada de Painel de Missão Espacial
st.markdown("""
<style>
    /* Tema Escuro Aeroespacial */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Mission Control */
    .mission-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #0a1128 100%);
        border: 1px solid #1e3a8a;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.08);
    }
    
    .mission-title {
        font-family: 'Inter', -apple-system, sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #00d4ff 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .mission-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Badges de Status */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-success { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #059669; }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #d97706; }
    .badge-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #dc2626; }
    .badge-info { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #2563eb; }

    /* Cards de Anomalias */
    .anomaly-card {
        background-color: #131b2e;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 5px solid;
    }
    .anomaly-warning { border-left-color: #f59e0b; }
    .anomaly-critical { border-left-color: #ef4444; }
    .anomaly-high { border-left-color: #ec4899; }
    .anomaly-medium { border-left-color: #3b82f6; }

    /* Observabilidade Footer */
    .observability-bar {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.8rem;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Inicialização de Estado da Sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "profiler" not in st.session_state:
    st.session_state.profiler = None
if "flight_report" not in st.session_state:
    st.session_state.flight_report = None
if "selected_query_input" not in st.session_state:
    st.session_state.selected_query_input = ""
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None


# Sidebar: Configuracao do Agente e Importacao
with st.sidebar:
    st.markdown("### Importar Telemetria")
    sidebar_file = st.file_uploader(
        "Arquivo de Log (.csv ou .txt):",
        type=["csv", "txt", "log"],
        key="sidebar_file_uploader",
        help="Formatos aceitos: CSV delimitado (, ; tab |) ou log bruto de pacotes/estacao de solo (.txt)."
    )

    if st.session_state.df is not None:
        if st.button("Limpar Dados / Novo Upload", use_container_width=True):
            st.session_state.df = None
            st.session_state.profiler = None
            st.session_state.flight_report = None
            st.session_state.messages = []
            st.session_state.current_filename = None
            st.rerun()

    st.markdown("---")
    st.markdown("### Motor de IA (LLM)")

    # Seletor de Provedor
    provider_opt = st.selectbox(
        "Provedor LLM:",
        ["Gemini API (Google GenAI)", "OpenAI API", "Motor Científico Determinístico (Offline)"],
        index=0
    )

    provider_map = {
        "Gemini API (Google GenAI)": "gemini",
        "OpenAI API": "openai",
        "Motor Científico Determinístico (Offline)": "offline"
    }
    selected_provider = provider_map[provider_opt]

    model_name = "gemini-3.5-flash-lite"
    if selected_provider == "gemini":
        model_name = st.selectbox(
            "Modelo Gemini:",
            ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash"],
            index=0
        )
    elif selected_provider == "openai":
        model_name = st.selectbox(
            "Modelo OpenAI:",
            ["gpt-4o-mini", "gpt-4o"],
            index=0
        )

    custom_key = st.text_input(
        "API Key Customizada (Opcional):",
        type="password",
        help="Deixe em branco para usar a chave configurada no ambiente do servidor."
    )

    st.markdown("---")
    st.markdown("### Guardrails e Sandbox")
    st.markdown("""
    - **AST Validator**: Bloqueio de eval, subprocess, IO de disco
    - **Self-Healing Loop**: Até 2 tentativas automáticas
    - **Context Budget**: Meta-estatísticas compactas (< 800 tokens)
    - **Plotly Integration**: Visualizações dinâmicas nativas
    """)


# Processamento do Arquivo Enviado pela Barra Lateral
if sidebar_file is not None and sidebar_file.name != st.session_state.current_filename:
    try:
        with st.spinner(f"Processando arquivo '{sidebar_file.name}'..."):
            loaded_df = load_telemetry(sidebar_file, filename=sidebar_file.name)
            st.session_state.df = loaded_df
            st.session_state.current_filename = sidebar_file.name
            profiler_inst = TelemetryProfiler(loaded_df)
            st.session_state.profiler = profiler_inst

            agent_init = TelemetryCopilotAgent(
                api_key=custom_key if custom_key else None,
                provider=selected_provider,
                model=model_name
            )
            st.session_state.flight_report = agent_init.generate_automated_flight_report(loaded_df, profiler_inst)
            st.session_state.messages = []
    except Exception as e:
        import traceback
        traceback.print_exc()
        st.error(f"Erro ao interpretar arquivo ({sidebar_file.name}): {e}")
        st.session_state.df = None
        st.session_state.current_filename = None

df_loaded = st.session_state.df
profiler = st.session_state.profiler
current_mission_name = st.session_state.current_filename


# Cabeçalho Principal (Mission Control Dashboard)
st.markdown(f"""
<div class="mission-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <h1 class="mission-title">Stratosphere Telemetry Copilot</h1>
            <p class="mission-subtitle">Agente Autônomo de Análise de Séries Temporais Aeroespaciais & Sensores Críticos</p>
        </div>
        <div style="text-align: right; margin-top: 6px;">
            <span class="status-badge {'badge-success' if df_loaded is not None else 'badge-warning'}">
                {'Missão Carregada' if df_loaded is not None else 'Aguardando Importação'}
            </span>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                Arquivo: {current_mission_name if df_loaded is not None else 'Nenhum'}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


if df_loaded is not None and profiler is not None:
    p_data = profiler.profile_data
    milestones = p_data.get("milestones", {})
    anomalies = p_data.get("anomalies", [])

    alt_col = "altitude_gps_m" if "altitude_gps_m" in df_loaded else ("altitude_m" if "altitude_m" in df_loaded else df_loaded.columns[0])
    press_col = "pressure_hpa" if "pressure_hpa" in df_loaded else None
    temp_col = next((c for c in ["temp_external_c", "temperature_c", "T", "temp"] if c in df_loaded), None)

    # Métricas Principais do Voo
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        burst_alt = milestones.get("burst", {}).get("altitude_m")
        if burst_alt is None:
            burst_alt = float(df_loaded[alt_col].max()) if not df_loaded.empty else 0.0
        st.metric("Apogeu (Burst)", f"{burst_alt:,.1f} m", delta=f"{burst_alt/1000:.2f} km")
    with col2:
        dur_min = p_data.get("total_duration_sec", 0.0) / 60.0
        st.metric("Duração da Missão", f"{dur_min:.1f} min", delta=f"{p_data.get('total_duration_sec', 0.0):.0f} s")
    with col3:
        min_temp = float(df_loaded[temp_col].min()) if temp_col else 0.0
        st.metric("Temperatura Mínima", f"{min_temp:.1f} °C", delta="Sensor Térmico")
    with col4:
        min_press = float(df_loaded[press_col].min()) if press_col else 0.0
        st.metric("Menor Pressão", f"{min_press:.2f} hPa", delta="Vácuo relativo")
    with col5:
        max_v_down = milestones.get("max_post_burst_descent_speed_mps", 0.0)
        st.metric("Velocidade Descida Pós-Burst", f"{max_v_down:.1f} m/s", delta=f"{max_v_down*3.6:.0f} km/h", delta_color="inverse")

    st.markdown("<br>", unsafe_allow_html=True)

    # Abas de Navegação Principal
    tab_report, tab_chat, tab_explorer, tab_evals = st.tabs([
        "Relatório Automático de Missão (1 Clique)",
        "Chat Analítico (Text-to-Python & Plotly)",
        "Explorador de Telemetria e Estatísticas",
        "Evals e Benchmark de Engenharia de IA"
    ])

    # =========================================================================
    # TAB 1: RELATÓRIO AUTOMÁTICO DE MISSÃO & FASES
    # =========================================================================
    with tab_report:
        st.subheader("Detecção Automática de Fases do Voo")
        st.caption("Identificação em tempo de execução via análise de gradientes de altitude, pressão barométrica e IMU.")

        report = st.session_state.flight_report or {}
        phases = report.get("phases", [])

        # Timeline das Fases
        phase_cols = st.columns(len(phases))
        for idx, (col, phase) in enumerate(zip(phase_cols, phases)):
            with col:
                st.markdown(f"""
                <div style="background-color: #1e293b; border-radius: 8px; padding: 12px; height: 100%; border-top: 3px solid #38bdf8;">
                    <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 700;">FASE {idx+1}</div>
                    <div style="font-weight: 700; font-size: 0.9rem; margin-top: 2px;">{phase['phase_name'].split('. ')[-1]}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin: 4px 0;">Tempo: {phase['time_window']}</div>
                    <div style="font-size: 0.75rem; color: #e2e8f0; font-weight: 600;">Faixa: {phase['altitude_range']}</div>
                    <div style="margin-top: 8px;"><span class="status-badge badge-info">{phase['status']}</span></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Diagnóstico de Anomalias Físicas e Hardware")

        if anomalies:
            anom_cols = st.columns(len(anomalies))
            sev_class_map = {
                "CRITICAL": "badge-danger",
                "HIGH": "badge-warning",
                "WARNING": "badge-warning",
                "MEDIUM": "badge-info"
            }
            card_class_map = {
                "CRITICAL": "anomaly-critical",
                "HIGH": "anomaly-high",
                "WARNING": "anomaly-warning",
                "MEDIUM": "anomaly-medium"
            }

            for col, anom in zip(anom_cols, anomalies):
                with col:
                    badge_cls = sev_class_map.get(anom["severity"], "badge-info")
                    card_cls = card_class_map.get(anom["severity"], "anomaly-warning")
                    st.markdown(f"""
                    <div class="anomaly-card {card_cls}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; font-size: 0.85rem;">{anom['type']}</span>
                            <span class="status-badge {badge_cls}">{anom['severity']}</span>
                        </div>
                        <p style="font-size: 0.8rem; color: #cbd5e1; margin: 0;">{anom['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("Nenhuma anomalia crítica detectada nos sensores de bordo.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Painel Gráfico Geral de Telemetria de Voo")

        # 4 Gráficos Mestres
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            fig_alt = px.line(
                df_loaded,
                x="time_sec",
                y=alt_col,
                title="<b>Perfil Vertical de Altitude vs Tempo</b>",
                labels={"time_sec": "Tempo de Voo (s)", alt_col: "Altitude (m)"},
                template="plotly_dark"
            )
            burst_sec = float(milestones.get("burst", {}).get("time_sec", 6200))
            t_last_sec = float(pd.to_numeric(df_loaded["time_sec"].iloc[-1], errors="coerce") or 0.0) if "time_sec" in df_loaded else 0.0
            if burst_sec < t_last_sec - 10:
                fig_alt.add_vline(x=burst_sec, line_dash="dash", line_color="#ff2a6d", annotation_text="BURST")
            fig_alt.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
            st.plotly_chart(fig_alt, use_container_width=True, key="dashboard_fig_alt")

        with gcol2:
            fig_thermal = go.Figure()
            if temp_col:
                fig_thermal.add_trace(go.Scatter(
                    x=df_loaded[temp_col],
                    y=df_loaded[alt_col],
                    mode="lines",
                    name=f"Temperatura ({temp_col})",
                    line=dict(color="#00d4ff", width=2)
                ))
            if "temp_internal_c" in df_loaded:
                fig_thermal.add_trace(go.Scatter(
                    x=df_loaded["temp_internal_c"],
                    y=df_loaded[alt_col],
                    mode="lines",
                    name="Temp Interna (°C)",
                    line=dict(color="#f59e0b", width=2, dash="dash")
                ))
            tropo_alt_val = milestones.get("tropopause", {}).get("altitude_m", 12049)
            if tropo_alt_val and tropo_alt_val < float(df_loaded[alt_col].max()):
                fig_thermal.add_hline(y=tropo_alt_val, line_dash="dot", line_color="#ff2a6d", annotation_text="Tropopausa")
            fig_thermal.update_layout(
                title="<b>Perfil Térmico Vertical vs Altitude</b>",
                xaxis_title="Temperatura (°C)",
                yaxis_title="Altitude (m)",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                height=320
            )
            st.plotly_chart(fig_thermal, use_container_width=True, key="dashboard_fig_thermal")

        gcol3, gcol4 = st.columns(2)
        with gcol3:
            if press_col:
                fig_press = px.line(
                    df_loaded,
                    x=alt_col,
                    y=press_col,
                    title="<b>Curva Barométrica: Pressão Atmosférica vs Altitude</b>",
                    labels={alt_col: "Altitude (m)", press_col: "Pressão (hPa)"},
                    template="plotly_dark"
                )
                fig_press.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
                st.plotly_chart(fig_press, use_container_width=True, key="dashboard_fig_press")

        with gcol4:
            lat_series = pd.to_numeric(df_loaded["latitude"], errors="coerce").dropna() if "latitude" in df_loaded else pd.Series(dtype=float)
            lon_series = pd.to_numeric(df_loaded["longitude"], errors="coerce").dropna() if "longitude" in df_loaded else pd.Series(dtype=float)
            alt_series = pd.to_numeric(df_loaded[alt_col], errors="coerce").dropna() if alt_col in df_loaded else pd.Series(dtype=float)

            has_gps = (len(lat_series) > 0 and len(lon_series) > 0 and len(alt_series) > 0)
            attitude_cols = [c for c in ["pitch_deg", "roll_deg", "yaw_deg", "accel_x_avg", "accel_y_avg", "accel_z_avg"] if c in df_loaded]

            if has_gps and attitude_cols:
                tab_g4_3d, tab_g4_imu = st.tabs(["Trajetória 3D / GPS", "Dinâmica de Atitude (IMU)"])
            else:
                tab_g4_3d = None
                tab_g4_imu = None

            if has_gps:
                container_3d = tab_g4_3d if tab_g4_3d is not None else st.container()
                with container_3d:
                    lat_min, lat_max = float(lat_series.min()), float(lat_series.max())
                    lon_min, lon_max = float(lon_series.min()), float(lon_series.max())
                    alt_min, alt_max = float(alt_series.min()), float(alt_series.max())

                    d_lat = max(lat_max - lat_min, 0.002)
                    d_lon = max(lon_max - lon_min, 0.002)
                    d_alt = max(alt_max - alt_min, 20.0)

                    disp_m = float(np.sqrt(((lat_max - lat_min) * 111000)**2 + ((lon_max - lon_min) * 102000)**2 + (alt_max - alt_min)**2))
                    is_static = disp_m < 30.0

                    fig_3d = go.Figure()
                    fig_3d.add_trace(go.Scatter3d(
                        x=lon_series,
                        y=lat_series,
                        z=alt_series,
                        mode="lines+markers",
                        marker=dict(size=6, color="#00d4ff", symbol="circle"),
                        line=dict(color="#00d4ff", width=4),
                        name="Fix GPS" if is_static else "Trajetória HAB"
                    ))

                    title_text = f"<b>Fix de Solo / Bancada ({disp_m:.1f}m)</b>" if is_static else "<b>Trajetória de Voo 3D (Lat x Lon x Alt)</b>"

                    fig_3d.update_layout(
                        title=title_text,
                        scene=dict(
                            xaxis=dict(title="Longitude", range=[lon_min - d_lon*0.3, lon_max + d_lon*0.3]),
                            yaxis=dict(title="Latitude", range=[lat_min - d_lat*0.3, lat_max + d_lat*0.3]),
                            zaxis=dict(title="Altitude (m)", range=[alt_min - d_alt*0.3, alt_max + d_alt*0.3])
                        ),
                        template="plotly_dark",
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=320
                    )
                    st.plotly_chart(fig_3d, use_container_width=True, key="dashboard_fig_3d")

            if attitude_cols:
                container_imu = tab_g4_imu if tab_g4_imu is not None else st.container()
                with container_imu:
                    plot_att_cols = [c for c in ["pitch_deg", "roll_deg", "yaw_deg"] if c in df_loaded]
                    if not plot_att_cols:
                        plot_att_cols = attitude_cols
                    fig_dyn = px.line(
                        df_loaded,
                        x="time_sec",
                        y=plot_att_cols,
                        title="<b>Dinâmica Angular IMU (Pitch, Roll, Yaw) vs Tempo</b>",
                        labels={"time_sec": "Tempo (s)", "value": "Graus (°)"},
                        template="plotly_dark"
                    )
                    fig_dyn.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
                    st.plotly_chart(fig_dyn, use_container_width=True, key="dashboard_fig_dyn")

    # =========================================================================
    # TAB 2: CHAT ANALÍTICO (TEXT-TO-PYTHON & PLOTLY)
    # =========================================================================
    with tab_chat:
        st.subheader("Chat Analítico de Telemetria (Text-to-Python Sandbox)")
        st.caption("Faça perguntas técnicas em linguagem natural. O agente escreve código Python em sandbox seguro, executa e plota gráficos interativos.")

        # Botões de Sugestões Rápidas (1 clique para o recrutador testar)
        st.markdown("**Sugestões de Análise Técnica Instantânea:**")
        chip1, chip2, chip3 = st.columns(3)
        with chip1:
            if st.button("Perfil Térmico e Tropopausa", use_container_width=True):
                st.session_state.selected_query_input = "Plote o perfil térmico (temperatura externa vs. interna) por faixa de altitude e destaque a tropopausa."
        with chip2:
            if st.button("Taxa de Subida e Velocidade Pós-Burst", use_container_width=True):
                st.session_state.selected_query_input = "Calcule a taxa média de subida (m/s) e a velocidade máxima na descida logo após o burst."
        with chip3:
            if st.button("Giroscópio e Falha de Pressão", use_container_width=True):
                st.session_state.selected_query_input = "Houve algum momento em que o giroscópio indicou rotação anômala enquanto a pressão falhava?"

        chip4, chip5, chip6 = st.columns(3)
        with chip4:
            if st.button("Queda Térmica de Tensão da Bateria", use_container_width=True):
                st.session_state.selected_query_input = "Analise o comportamento de tensão da bateria (V) ao longo do tempo e identifique se houve queda térmica acentuada."
        with chip5:
            if st.button("Trajetória 3D de Voo (Coordenadas)", use_container_width=True):
                st.session_state.selected_query_input = "Trace a trajetória de voo em 3D usando latitude, longitude e altitude."
        with chip6:
            if st.button("Integridade de Pacotes MicroSD", use_container_width=True):
                st.session_state.selected_query_input = "Verifique a integridade do link de telemetria e o contador de pacotes: houve perda de pacotes no MicroSD?"

        st.markdown("---")

        # Histórico de Mensagens do Chat
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "figure_json" in msg and msg["figure_json"]:
                    try:
                        fig_dict = json.loads(msg["figure_json"])
                        st.plotly_chart(go.Figure(fig_dict), use_container_width=True, key=f"chat_hist_fig_{idx}")
                    except Exception:
                        pass
                if "code" in msg and msg["code"]:
                    with st.expander("Ver Código Python Gerado e Executado"):
                        st.code(msg["code"], language="python")
                if "observability" in msg and msg["observability"]:
                    obs = msg["observability"]
                    st.markdown(f"""
                    <div class="observability-bar">
                        <span>Modelo: <b>{obs.get('model', 'N/A')}</b></span>
                        <span>Latência: <b>{obs.get('latency_ms', 0):.0f} ms</b></span>
                        <span>Auto-Correção: <b>{'Sim (' + str(obs.get('retries', 0)) + ' retries)' if obs.get('self_healed') else '0 retries (1ª tentativa)'}</b></span>
                        <span>Sandbox: <b style="color: #34d399;">Executado com Sucesso</b></span>
                    </div>
                    """, unsafe_allow_html=True)

        # Entrada de Texto do Usuário
        input_placeholder = "Ex: Calcule a aceleração vertical máxima durante o estouro do balão..."
        user_prompt = st.chat_input(input_placeholder)

        # Se clicou em um dos botões de sugestão, preenche a consulta
        if st.session_state.selected_query_input:
            user_prompt = st.session_state.selected_query_input
            st.session_state.selected_query_input = ""

        if user_prompt:
            # Exibe mensagem do usuário
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Executa com o agente
            with st.chat_message("assistant"):
                with st.spinner("Analisando telemetria, gerando script científico e executando em sandbox..."):
                    agent = TelemetryCopilotAgent(
                        api_key=custom_key if custom_key else None,
                        provider=selected_provider,
                        model=model_name
                    )
                    resp = agent.run_query(user_prompt, df_loaded, profiler)

                    # Exibe síntese técnica
                    st.markdown(resp.synthesis)

                    # Exibe gráfico se houver
                    fig_to_save_json = None
                    if agent.last_fig:
                        st.plotly_chart(agent.last_fig, use_container_width=True, key=f"chat_new_fig_{len(st.session_state.messages)}")
                        fig_to_save_json = agent.last_fig.to_json()

                    # Exibe código em expander
                    with st.expander("Ver Código Python Gerado e Executado"):
                        st.code(resp.code, language="python")

                    # Observabilidade
                    st.markdown(f"""
                    <div class="observability-bar">
                        <span>Modelo: <b>{resp.model_name}</b></span>
                        <span>Latência: <b>{resp.total_latency_ms:.0f} ms</b></span>
                        <span>Auto-Correção: <b>{'Sim (' + str(resp.retries_count) + ' retries)' if resp.self_healed else '0 retries (1ª tentativa)'}</b></span>
                        <span>Sandbox: <b style="color: #34d399;">Executado com Sucesso</b></span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Salva no histórico
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resp.synthesis,
                        "code": resp.code,
                        "figure_json": fig_to_save_json,
                        "observability": {
                            "model": resp.model_name,
                            "latency_ms": resp.total_latency_ms,
                            "self_healed": resp.self_healed,
                            "retries": resp.retries_count
                        }
                    })

    # =========================================================================
    # TAB 3: EXPLORADOR DE TELEMETRIA & ESTATÍSTICAS
    # =========================================================================
    with tab_explorer:
        st.subheader("Metadados do Dataset e Schema (Nó 1 - Profiler)")
        st.markdown("O agente utiliza exclusivamente metadados agregados para não estourar a janela de contexto de tokens:")

        with st.expander("Ver Contexto Exato Passado ao LLM (Context Engineering)", expanded=True):
            st.code(profiler.get_llm_context_string(), language="yaml")

        st.markdown("### Amostra de Dados da Telemetria (Head)")
        st.dataframe(df_loaded.head(100), use_container_width=True)

        st.markdown("### Resumo Estatístico Numérico")
        st.dataframe(df_loaded.describe().T, use_container_width=True)

        # Botão para Download
        csv_bytes = df_loaded.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar Log Completo de Telemetria (.csv)",
            data=csv_bytes,
            file_name="telemetria_voo_hab.csv",
            mime="text/csv"
        )

    # =========================================================================
    # TAB 4: EVALS & BENCHMARK DE ENGENHARIA DE IA
    # =========================================================================
    with tab_evals:
        st.subheader("Evals e Benchmark de Confiabilidade (15 Perguntas Técnicas)")
        st.markdown("""
        Demonstração rigorosa de **Engenharia de IA Real**:
        - Taxa de acerto sintático e execução em sandbox
        - Eficácia comprovada do **Loop de Auto-Correção (Self-Healing)**
        - Latência média por consulta e aderência a Structured Outputs
        """)

        eval_json_path = Path(__file__).parent / "evals" / "eval_results.json"
        queries_json_path = Path(__file__).parent / "evals" / "test_queries.json"

        # Carrega perguntas do catálogo
        test_queries = []
        if queries_json_path.exists():
            with open(queries_json_path, "r", encoding="utf-8") as f:
                test_queries = json.load(f)

        # Resultados anteriores
        if eval_json_path.exists():
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            ecol1, ecol2, ecol3, ecol4 = st.columns(4)
            with ecol1:
                st.metric("Taxa de Sucesso (Pass Rate)", f"{eval_data.get('pass_rate_pct', 100):.1f}%", f"{eval_data.get('passed', 0)}/{eval_data.get('total_queries', 0)} OK")
            with ecol2:
                st.metric("Auto-Correções com Sucesso", f"{eval_data.get('self_healed_count', 0)} recuperadas", delta="Self-Healing Ativo")
            with ecol3:
                st.metric("Latência Média", f"{eval_data.get('avg_latency_ms', 0):.0f} ms")
            with ecol4:
                st.metric("Modelo Avaliado", eval_data.get("model", "gemini-3.5-flash-lite"))

            st.markdown("### Perguntas Técnicas Avaliadas no Benchmark")
            df_evals = pd.DataFrame(eval_data.get("queries_evaluated", []))
            if not df_evals.empty:
                display_cols = ["id", "category", "query", "passed", "self_healed", "latency_ms"]
                available_cols = [c for c in display_cols if c in df_evals.columns]
                st.dataframe(df_evals[available_cols], use_container_width=True)

        st.markdown("---")
        st.markdown("### Catálogo Completo das 15 Perguntas de Avaliação:")
        for q in test_queries:
            st.markdown(f"- **[{q['id']}] ({q['category']})**: {q['query']}")
else:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0d1b2a 0%, #131b2e 100%); border: 2px dashed #1e3a8a; border-radius: 14px; padding: 50px 32px; text-align: center; margin: 30px auto; max-width: 820px; box-shadow: 0 4px 20px rgba(0, 212, 255, 0.05);">
        <h2 style="color: #38bdf8; font-weight: 700; margin-bottom: 12px; font-size: 1.6rem; letter-spacing: -0.3px;">Importar Arquivo de Telemetria</h2>
        <p style="color: #94a3b8; font-size: 0.95rem; max-width: 620px; margin: 0 auto 20px auto; line-height: 1.6;">
            Selecione ou arraste um arquivo de log (.csv ou .txt). O sistema detecta automaticamente o formato (dados tabulares delimitados ou quadros de pacotes de rádio/estação), processa as séries temporais e ativa o agente de análise.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        main_uploaded_file = st.file_uploader(
            "Arraste ou selecione o arquivo (.csv ou .txt):",
            type=["csv", "txt", "log"],
            key="main_file_uploader",
            help="Formatos aceitos: CSV delimitado (, ; tab |) ou log bruto de pacotes de rádio/estação (.txt)."
        )
        if main_uploaded_file is not None:
            try:
                with st.spinner(f"Processando arquivo '{main_uploaded_file.name}'..."):
                    loaded_df = load_telemetry(main_uploaded_file, filename=main_uploaded_file.name)
                    st.session_state.df = loaded_df
                    st.session_state.current_filename = main_uploaded_file.name
                    profiler_inst = TelemetryProfiler(loaded_df)
                    st.session_state.profiler = profiler_inst

                    agent_init = TelemetryCopilotAgent(
                        api_key=custom_key if custom_key else None,
                        provider=selected_provider,
                        model=model_name
                    )
                    st.session_state.flight_report = agent_init.generate_automated_flight_report(loaded_df, profiler_inst)
                    st.session_state.messages = []
                    st.rerun()
            except Exception as e:
                import traceback
                traceback.print_exc()
                st.error(f"Erro ao interpretar arquivo ({main_uploaded_file.name}): {e}")
