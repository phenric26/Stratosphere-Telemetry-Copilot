"""
Telemetry Profiler (Nó 1 da Arquitetura do Agente)
Lê metadados, schema, taxa de amostragem, resumo estatístico e detecta
marcos preliminares de voo e anomalias sem despejar o CSV bruto no contexto do LLM.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class TelemetryProfiler:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.profile_data: Dict[str, Any] = {}
        self._analyze()

    def _classify_columns(self) -> Dict[str, List[str]]:
        cols = list(self.df.columns)
        classified = {
            "time": [],
            "altitude": [],
            "pressure": [],
            "temperature": [],
            "battery": [],
            "gyro": [],
            "accel": [],
            "gps": [],
            "packet": [],
            "other": []
        }
        for col in cols:
            c = col.lower()
            if any(k in c for k in ["time", "timestamp", "datetime", "sec"]):
                classified["time"].append(col)
            elif "alt" in c:
                classified["altitude"].append(col)
            elif c == "p" or "press" in c or ("baro" in c and "alt" not in c):
                classified["pressure"].append(col)
            elif c == "t" or "temp" in c or "t_ext" in c or "t_int" in c:
                classified["temperature"].append(col)
            elif c in ["bat", "vbat", "v_bat"] or "batt" in c or "volt" in c or "vbat" in c:
                classified["battery"].append(col)
            elif "gyro" in c or "dps" in c or "deg_s" in c or "pitch" in c or "roll" in c or "yaw" in c:
                classified["gyro"].append(col)
            elif "acc" in c or "grav" in c or c in ["ax", "ay", "az"]:
                classified["accel"].append(col)
            elif any(k in c for k in ["lat", "lon", "coord", "rssi", "snr", "sat", "fix"]):
                classified["gps"].append(col)
            elif "packet" in c or "frame" in c or "seq" in c or "ack" in c:
                classified["packet"].append(col)
            else:
                classified["other"].append(col)
        return classified

    def _analyze(self):
        df = self.df
        row_count, col_count = df.shape
        classified = self._classify_columns()

        # 1. Taxa de amostragem e tempo
        # Prefere coluna de tempo numerica (ex: time_sec)
        time_col = None
        for tc in classified["time"]:
            if pd.api.types.is_numeric_dtype(df[tc]):
                time_col = tc
                break
        if not time_col and classified["time"]:
            time_col = classified["time"][0]

        dt_median = 1.0
        sampling_rate_hz = 1.0
        total_duration_sec = 0.0

        if time_col and pd.api.types.is_numeric_dtype(df[time_col]):
            time_series = pd.to_numeric(df[time_col], errors="coerce").dropna()
            diffs = time_series.diff().dropna()
            diffs = diffs[diffs > 0]
            if not diffs.empty:
                dt_median = float(diffs.median())
                sampling_rate_hz = round(1.0 / dt_median, 3) if dt_median > 0 else 1.0
                total_duration_sec = float(time_series.iloc[-1] - time_series.iloc[0])
        elif time_col:
            try:
                # Tenta formato de horario simples HH:MM:SS ou ISO
                ts = pd.to_datetime(df[time_col], format="%H:%M:%S", errors="coerce")
                if ts.isna().all():
                    ts = pd.to_datetime(df[time_col], errors="coerce")
                diffs = ts.diff().dt.total_seconds().dropna()
                diffs = diffs[diffs > 0]
                if not diffs.empty:
                    dt_median = float(diffs.median())
                    sampling_rate_hz = round(1.0 / dt_median, 3) if dt_median > 0 else 1.0
                    total_duration_sec = float((ts.dropna().iloc[-1] - ts.dropna().iloc[0]).total_seconds())
            except Exception:
                pass

        # 2. Resumo estatístico
        num_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        for col in num_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                stats[col] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "25%": float(s.quantile(0.25)),
                    "50%": float(s.quantile(0.50)),
                    "75%": float(s.quantile(0.75)),
                    "null_count": int(df[col].isna().sum()),
                    "null_pct": round(float(df[col].isna().mean() * 100), 2)
                }

        # 3. Marcos Físicos Preliminares
        alt_col = classified["altitude"][0] if classified["altitude"] else None
        milestones = {}
        anomalies = []

        if alt_col and alt_col in df:
            alt_series = pd.to_numeric(df[alt_col], errors="coerce").fillna(0.0)
            max_alt = float(alt_series.max())
            max_alt_idx = int(alt_series.idxmax())
            min_alt = float(alt_series.min())
            
            # Burst point
            burst_time = float(pd.to_numeric(df["time_sec"].iloc[max_alt_idx], errors="coerce") or float(max_alt_idx)) if "time_sec" in df else float(max_alt_idx)
            milestones["burst"] = {
                "altitude_m": max_alt,
                "row_index": max_alt_idx,
                "time_sec": burst_time
            }

            # Taxas de subida e descida
            if "time_sec" in df:
                ascent_mask = df.index <= max_alt_idx
                descent_mask = df.index > max_alt_idx

                if ascent_mask.sum() > 1:
                    ascent_dh = max_alt - float(alt_series.iloc[0])
                    t0 = float(pd.to_numeric(df["time_sec"].iloc[0], errors="coerce") or 0.0)
                    ascent_dt = max(1.0, burst_time - t0)
                    milestones["avg_ascent_rate_mps"] = round(ascent_dh / ascent_dt, 2)
                
                if descent_mask.sum() > 1:
                    land_alt = float(alt_series.iloc[-1])
                    land_time = float(pd.to_numeric(df["time_sec"].iloc[-1], errors="coerce") or 0.0)
                    descent_dh = max_alt - land_alt
                    descent_dt = max(1.0, land_time - burst_time)
                    milestones["avg_descent_rate_mps"] = round(descent_dh / descent_dt, 2)

                # Velocidade instantânea máxima na descida (logo após o burst)
                post_burst = df.loc[max_alt_idx:min(max_alt_idx + 150, len(df) - 1)].copy()
                if len(post_burst) > 2:
                    dz = -pd.to_numeric(post_burst[alt_col], errors="coerce").diff()
                    dt = pd.to_numeric(post_burst["time_sec"], errors="coerce").diff()
                    descent_speed = (dz / dt).dropna()
                    if not descent_speed.empty:
                        milestones["max_post_burst_descent_speed_mps"] = round(float(descent_speed.max()), 2)

        # 4. Análise Térmica & Tropopausa
        temp_ext_col = next((c for c in classified["temperature"] if "ext" in c.lower()), None)
        if not temp_ext_col and classified["temperature"]:
            temp_ext_col = classified["temperature"][0]

        if temp_ext_col and temp_ext_col in df:
            temp_series = pd.to_numeric(df[temp_ext_col], errors="coerce").dropna()
            if not temp_series.empty:
                min_temp = float(temp_series.min())
                min_temp_idx = int(temp_series.idxmin())
                min_temp_alt = float(pd.to_numeric(df[alt_col].iloc[min_temp_idx], errors="coerce") or 0.0) if alt_col else None
                milestones["tropopause"] = {
                    "min_temp_c": min_temp,
                    "altitude_m": min_temp_alt,
                    "row_index": min_temp_idx
                }

        # 5. Detecção de Anomalias Físicas e Hardware
        # a) Queda de bateria por frio extremo / sub-tensão (Detecção automática 1S vs 2S)
        batt_col = classified["battery"][0] if classified["battery"] else None
        if batt_col and batt_col in df:
            bat_series = pd.to_numeric(df[batt_col], errors="coerce").dropna()
            if not bat_series.empty:
                min_v = float(bat_series.min())
                max_v = float(bat_series.max())
                drop = max_v - min_v
                # Identifica número de células LiPo (1S vs 2S vs 3S)
                is_1s = max_v < 4.5
                crit_threshold = 3.35 if is_1s else 7.15
                warn_drop = 0.45 if is_1s else 0.85

                if min_v < crit_threshold or drop > warn_drop:
                    min_v_idx = int(bat_series.idxmin())
                    v_time = float(pd.to_numeric(df["time_sec"].iloc[min_v_idx], errors="coerce") or min_v_idx) if "time_sec" in df else min_v_idx
                    anomalies.append({
                        "type": "BATTERY_THERMAL_SAG",
                        "severity": "WARNING",
                        "description": f"Queda acentuada de tensão na bateria ({min_v:.2f}V) registrada em t={v_time}s.",
                        "min_value": min_v
                    })

        # b) Congelamento / Saturação de Barômetro (< 10 hPa com variância zero)
        press_col = classified["pressure"][0] if classified["pressure"] else None
        if press_col and press_col in df:
            press_series = pd.to_numeric(df[press_col], errors="coerce").dropna()
            low_press = press_series[press_series < 12.0]
            if len(low_press) > 10:
                diffs_zero = (low_press.diff().abs() < 0.03).sum()
                if diffs_zero > 15:
                    anomalies.append({
                        "type": "BAROMETER_SATURATION_OR_FREEZE",
                        "severity": "CRITICAL",
                        "description": f"Barômetro ({press_col}) estagnou/congelou em baixa pressão ({low_press.min():.2f} hPa) perto do apogeu.",
                        "min_pressure_hpa": float(low_press.min())
                    })

        # c) Picos de rotação no giroscópio (Turbulência na Tropopausa ou Burst)
        gyro_cols = classified["gyro"]
        if gyro_cols:
            max_rates = {}
            for gc in gyro_cols:
                g_series = pd.to_numeric(df[gc], errors="coerce").abs().dropna()
                if not g_series.empty:
                    max_rates[gc] = float(g_series.max())
            if max_rates:
                extreme_gyro = max(max_rates.values())
                if extreme_gyro > 50.0:
                    anomalies.append({
                        "type": "IMU_GYRO_HIGH_TUMBLE",
                        "severity": "HIGH",
                        "description": f"Pico severo de rotação angular ({extreme_gyro:.1f} dps) detectado durante evento de burst/cisalhamento de vento na tropopausa.",
                        "max_dps": extreme_gyro
                    })

        # d) Falhas de pacote / perda de dados no MicroSD
        pkt_col = None
        for cand in classified["packet"]:
            c = cand.lower()
            if any(k in c for k in ["seq", "packet", "frame", "num", "count", "idx"]) and "ack" not in c:
                pkt_col = cand
                break
        if not pkt_col and classified["packet"]:
            cand = classified["packet"][0]
            if "ack" not in cand.lower():
                pkt_col = cand

        if pkt_col and pkt_col in df:
            pkt_num = pd.to_numeric(df[pkt_col], errors="coerce").dropna()
            if len(pkt_num) > 1:
                pkt_diffs = pkt_num.diff().dropna()
                pos_diffs = pkt_diffs[pkt_diffs > 1]
                if not pos_diffs.empty:
                    dropped = int((pos_diffs - 1).sum())
                    if dropped > 0:
                        anomalies.append({
                            "type": "TELEMETRY_PACKET_LOSS",
                            "severity": "MEDIUM",
                            "description": f"Perda ou corrupção de pacotes detectada: total de {dropped} quadros perdidos no MicroSD/link RF.",
                            "dropped_packets": dropped
                        })

        self.profile_data = {
            "row_count": row_count,
            "col_count": col_count,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "classified_columns": classified,
            "sampling_rate_hz": sampling_rate_hz,
            "median_dt_sec": dt_median,
            "total_duration_sec": total_duration_sec,
            "summary_stats": stats,
            "milestones": milestones,
            "anomalies": anomalies
        }
        return self.profile_data

    def get_llm_context_string(self) -> str:
        """
        Retorna string compacta de telemetria (< 800 tokens) para injeção no prompt do LLM.
        Garante zero vazamento de linhas brutas e economia total de contexto.
        """
        d = self.profile_data
        lines = [
            "=== RESUMO DO DATASET DE TELEMETRIA ===",
            f"Dimensões: {d['row_count']} linhas, {d['col_count']} colunas",
            f"Amostragem: ~{d['sampling_rate_hz']} Hz (dt mediano = {d['median_dt_sec']}s)",
            f"Duração Total: {d['total_duration_sec']:.1f} segundos ({d['total_duration_sec']/60:.1f} min)",
            "",
            "--- COLUNAS E TIPOS DISPONÍVEIS ---",
        ]
        for col, dt in d["dtypes"].items():
            st = d["summary_stats"].get(col)
            if st:
                lines.append(f"- '{col}' ({dt}): min={st['min']:.2f}, max={st['max']:.2f}, mean={st['mean']:.2f}, std={st['std']:.2f}, nulls={st['null_count']}")
            else:
                lines.append(f"- '{col}' ({dt})")

        lines.append("")
        lines.append("--- MARCOS FÍSICOS IDENTIFICADOS ---")
        for k, v in d["milestones"].items():
            lines.append(f"- {k}: {v}")

        if d["anomalies"]:
            lines.append("")
            lines.append("--- ANOMALIAS PRELIMINARES DETECTADAS ---")
            for a in d["anomalies"]:
                lines.append(f"- [{a['severity']}] {a['type']}: {a['description']}")

        return "\n".join(lines)
