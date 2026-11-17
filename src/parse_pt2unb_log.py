"""
Parser e Ingestor do Log Real de Telemetria PT2UNB (Universidade de Brasília)
Converte o log bruto de pacotes de rádio LoRa/TDM em DataFrame limpo e arquivo CSV estruturado.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np


def parse_pt2unb_text(text: str) -> pd.DataFrame:
    # Quebra o texto por blocos de frame
    # Cada frame comeca com '------- PT2UNB -------' ou contem os campos
    lines = text.splitlines()
    frames = []
    current_frame: Dict[str, Any] = {}
    
    in_frame = False
    last_event = None
    crc_error_flag = False

    for line in lines:
        line_clean = line.strip()
        
        # Detecta eventos de comando e erro LoRa
        if "COMANDO [1020] ENFILEIRADO" in line_clean or "Disparando 'CMD:1020'" in line_clean:
            last_event = "CMD:1020_DISPARADO"
        elif "COMANDO 1020 CONFIRMADO" in line_clean:
            last_event = "CMD:1020_CONFIRMADO_ACK1021"
        elif "COMANDO [2500] ENFILEIRADO" in line_clean or "Disparando 'CMD:2500'" in line_clean:
            last_event = "CMD:2500_DISPARADO"
        elif "COMANDO 2500 CONFIRMADO" in line_clean:
            last_event = "CMD:2500_CONFIRMADO_ACK2501"
        elif "Recepcao falhou (CRC/timeout)" in line_clean:
            crc_error_flag = True

        if line_clean.startswith("------- PT2UNB -------"):
            if current_frame and "time" in current_frame:
                if last_event:
                    current_frame["event"] = last_event
                    last_event = None
                current_frame["lora_crc_error"] = crc_error_flag
                crc_error_flag = False
                frames.append(current_frame)
            current_frame = {"callsign": "PT2UNB"}
            in_frame = True
            continue

        if line_clean.startswith("----------------------"):
            if current_frame and "time" in current_frame:
                if last_event:
                    current_frame["event"] = last_event
                    last_event = None
                current_frame["lora_crc_error"] = crc_error_flag
                crc_error_flag = False
                frames.append(current_frame)
            current_frame = {}
            in_frame = False
            continue

        if in_frame and ":" in line_clean:
            # Trata 'RSSI:-71 dBm | SNR:12 dB'
            if "RSSI:" in line_clean and "SNR:" in line_clean:
                parts = line_clean.split("|")
                m_rssi = re.search(r"RSSI:\s*([-\d\.]+)", parts[0])
                m_snr = re.search(r"SNR:\s*([-\d\.]+)", parts[1])
                if m_rssi:
                    current_frame["rssi_dbm"] = float(m_rssi.group(1))
                if m_snr:
                    current_frame["snr_db"] = float(m_snr.group(1))
                continue

            k, v = line_clean.split(":", 1)
            k = k.strip()
            v = v.strip()

            key_map = {
                "Lat": ("latitude", float),
                "Lon": ("longitude", float),
                "Alt": ("altitude_gps_m", float),
                "AltB": ("altitude_baro_m", float),
                "Sat": ("satellites", int),
                "Fix": ("gps_fix", int),
                "T": ("temperature_c", float),
                "P": ("pressure_hpa", float),
                "Time": ("time", str),
                "Pitch": ("pitch_deg", float),
                "Roll": ("roll_deg", float),
                "Yaw": ("yaw_deg", float),
                "AXavg": ("accel_x_avg", float),
                "AYavg": ("accel_y_avg", float),
                "AZavg": ("accel_z_avg", float),
                "Bat": ("battery_v", float),
                "Ack": ("ack", int)
            }

            if k in key_map:
                field_name, field_type = key_map[k]
                try:
                    current_frame[field_name] = field_type(v)
                except Exception:
                    current_frame[field_name] = v

    # Último frame se restou
    if current_frame and "time" in current_frame:
        frames.append(current_frame)

    df = pd.DataFrame(frames)
    if df.empty:
        return df

    # Calcula tempo decorrido em segundos a partir do primeiro frame
    # Formato Time: HH:MM:SS
    def time_to_sec(t_str):
        parts = [int(p) for p in t_str.split(":")]
        return parts[0] * 3600 + parts[1] * 60 + parts[2]

    t_sec_raw = df["time"].apply(time_to_sec)
    t0 = t_sec_raw.iloc[0]
    # Trata possível virada de meia-noite se houver
    df["time_sec"] = t_sec_raw - t0
    df["time_sec"] = df["time_sec"].apply(lambda s: s if s >= 0 else s + 86400)

    # Cria timestamp ISO padrão para o voo
    # Ex: Data do voo 2026-08-15 + Time
    df["timestamp"] = "2026-08-15T" + df["time"] + "Z"

    # Ordena colunas de forma limpa e padronizada
    ordered_cols = [
        "time", "time_sec", "timestamp",
        "altitude_gps_m", "altitude_baro_m", "pressure_hpa", "temperature_c",
        "battery_v", "rssi_dbm", "snr_db", "satellites", "gps_fix",
        "pitch_deg", "roll_deg", "yaw_deg",
        "accel_x_avg", "accel_y_avg", "accel_z_avg",
        "latitude", "longitude", "ack"
    ]
    present_cols = [c for c in ordered_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in present_cols]
    df = df[present_cols + remaining_cols]

    return df


if __name__ == "__main__":
    import sys
    transcript_path = "/home/phenric/.gemini/antigravity-cli/brain/89c44d18-e75a-4e4d-bee0-74f240623244/.system_generated/logs/transcript_full.jsonl"
    print("Lendo transcript...")
    target_content = ""
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "USER_INPUT":
                c = obj.get("content", "")
                if "PT2UNB" in c and len(c) > len(target_content):
                    target_content = c

    print(f"Conteudo PT2UNB encontrado com {len(target_content)} caracteres.")
    df = parse_pt2unb_text(target_content)
    print(f"DataFrame gerado com {len(df)} linhas e {len(df.columns)} colunas.")
    print("Colunas:", list(df.columns))
    print(df.head())

    out_csv = "/home/phenric/IA-ifood/strato-telemetry-copilot/data/pt2unb_real_flight.csv"
    df.to_csv(out_csv, index=False)
    print(f"Salvo em: {out_csv}")
