"""
Carregador e Parser Universal de Telemetria Aeroespacial (CSV e TXT).
Suporta logs brutos de estacoes de solo (PT2UNB, LoRa, APRS),
arquivos delimitados (virgula, ponto-e-virgula, tabulacao, espaco) com extensao .csv ou .txt,
alem de fluxos de telemetria baseados em chave-valor e JSON Lines.
"""

import io
import re
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import pandas as pd
import numpy as np


def _decode_bytes(content: bytes) -> str:
    """Decodifica bytes tentando utf-8, latin-1 e cp1252."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def parse_packet_log(text: str) -> pd.DataFrame:
    """
    Parser para logs de pacotes estruturados por blocos (como PT2UNB, LoRa e APRS).
    Detecta delimitadores de frame (ex: '------- CALLSIGN -------' ou '----------------------'),
    linhas com 'RSSI:... | SNR:...', metadados de estacao e telecomandos (CMD/ACK).
    """
    lines = text.splitlines()
    frames: List[Dict[str, Any]] = []
    current_frame: Dict[str, Any] = {}
    
    in_frame = False
    last_event = None
    crc_error_flag = False

    # Mapeamento canonico de chaves para tipos padrao de engenharia
    key_map = {
        "lat": ("latitude", float),
        "latitude": ("latitude", float),
        "lon": ("longitude", float),
        "longitude": ("longitude", float),
        "alt": ("altitude_gps_m", float),
        "alt_gps": ("altitude_gps_m", float),
        "altitude": ("altitude_gps_m", float),
        "altb": ("altitude_baro_m", float),
        "alt_baro": ("altitude_baro_m", float),
        "sat": ("satellites", int),
        "sats": ("satellites", int),
        "fix": ("gps_fix", int),
        "t": ("temperature_c", float),
        "temp": ("temperature_c", float),
        "p": ("pressure_hpa", float),
        "press": ("pressure_hpa", float),
        "time": ("time", str),
        "pitch": ("pitch_deg", float),
        "roll": ("roll_deg", float),
        "yaw": ("yaw_deg", float),
        "axavg": ("accel_x_avg", float),
        "ayavg": ("accel_y_avg", float),
        "azavg": ("accel_z_avg", float),
        "ax": ("accel_x_avg", float),
        "ay": ("accel_y_avg", float),
        "az": ("accel_z_avg", float),
        "bat": ("battery_v", float),
        "vbat": ("battery_v", float),
        "battery": ("battery_v", float),
        "ack": ("ack", int),
        "rssi": ("rssi_dbm", float),
        "snr": ("snr_db", float),
    }

    callsign = "TELEMETRY"

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        # Ignora linhas de cabecalho geral de estacao que nao sao telemetria
        if line_clean.startswith("[ESTACAO]") or line_clean.startswith("[STATION]"):
            continue

        # Deteccao de telecomandos terrestres e eventos LoRa
        if "COMANDO [1020]" in line_clean or "CMD:1020" in line_clean:
            last_event = "CMD:1020_CALIBRACAO_BARO_GPS"
        elif "COMANDO 1020 CONFIRMADO" in line_clean or "ACK:1021" in line_clean.upper():
            last_event = "CMD:1020_CONFIRMADO_ACK1021"
        elif "COMANDO [2500]" in line_clean or "CMD:2500" in line_clean:
            last_event = "CMD:2500_CALIBRACAO_IMU_TARA"
        elif "COMANDO 2500 CONFIRMADO" in line_clean or "ACK:2501" in line_clean.upper():
            last_event = "CMD:2500_CONFIRMADO_ACK2501"
        elif "CRC/timeout" in line_clean or "Recepcao falhou" in line_clean:
            crc_error_flag = True

        # Deteccao de delimitadores de frame
        frame_header_match = re.match(r"^[-=]{3,}\s*([A-Za-z0-9_-]+)?\s*[-=]{3,}$", line_clean)
        if frame_header_match:
            matched_name = frame_header_match.group(1)
            # Se for linha de fechamento '----------------------'
            if not matched_name:
                if current_frame and ("time" in current_frame or "altitude_gps_m" in current_frame or len(current_frame) >= 2):
                    if last_event:
                        current_frame["event"] = last_event
                        last_event = None
                    current_frame["lora_crc_error"] = crc_error_flag
                    crc_error_flag = False
                    frames.append(current_frame)
                current_frame = {}
                in_frame = False
                continue
            else:
                # Inicio de novo frame
                if current_frame and ("time" in current_frame or "altitude_gps_m" in current_frame or len(current_frame) >= 2):
                    if last_event:
                        current_frame["event"] = last_event
                        last_event = None
                    current_frame["lora_crc_error"] = crc_error_flag
                    crc_error_flag = False
                    frames.append(current_frame)
                callsign = matched_name
                current_frame = {"callsign": callsign}
                in_frame = True
                continue

        # Linhas de sinal de radio: RSSI:-71 dBm | SNR:12 dB
        if "RSSI:" in line_clean and "SNR:" in line_clean:
            parts = line_clean.split("|")
            m_rssi = re.search(r"RSSI:\s*([-\d\.]+)", parts[0], re.IGNORECASE)
            m_snr = re.search(r"SNR:\s*([-\d\.]+)", parts[1], re.IGNORECASE)
            if m_rssi:
                current_frame["rssi_dbm"] = float(m_rssi.group(1))
            if m_snr:
                current_frame["snr_db"] = float(m_snr.group(1))
            continue

        # Linhas no formato Chave: Valor ou Chave=Valor
        sep = ":" if ":" in line_clean else ("=" if "=" in line_clean else None)
        if sep:
            parts = line_clean.split(sep, 1)
            k_raw = parts[0].strip().lower()
            v_raw = parts[1].strip()

            # Remove unidades comuns (dBm, dB, m, hPa, C, V, deg)
            v_clean = re.sub(r"\s*(?:dBm|dB|hPa|deg|m|V|C)\s*$", "", v_raw, flags=re.IGNORECASE).strip()

            if k_raw in key_map:
                std_key, val_type = key_map[k_raw]
                try:
                    current_frame[std_key] = val_type(v_clean)
                except Exception:
                    if val_type in (float, int):
                        current_frame[std_key] = None
                    else:
                        current_frame[std_key] = v_raw
            else:
                try:
                    if "." in v_clean:
                        current_frame[k_raw] = float(v_clean)
                    else:
                        current_frame[k_raw] = int(v_clean)
                except Exception:
                    if v_clean in ["-", "--", "nan", "null", "none", ""]:
                        current_frame[k_raw] = None
                    else:
                        current_frame[k_raw] = v_raw

    # Salva ultimo frame pendente se houver dados
    if current_frame and ("time" in current_frame or "altitude_gps_m" in current_frame or len(current_frame) >= 2):
        if last_event:
            current_frame["event"] = last_event
        current_frame["lora_crc_error"] = crc_error_flag
        frames.append(current_frame)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    return df


def parse_generic_key_values(text: str) -> pd.DataFrame:
    """
    Parser para arquivos de texto onde linhas contem multiplos pares chave:valor ou chave=valor
    (ex: 'Time: 12:00:00 Alt: 500 P: 950') ou blocos sequenciais de linhas chave-valor.
    """
    lines = text.splitlines()
    records: List[Dict[str, Any]] = []
    current_rec: Dict[str, Any] = {}
    seen_keys = set()

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            if current_rec:
                records.append(current_rec)
                current_rec = {}
                seen_keys = set()
            continue

        pairs = re.findall(r"([A-Za-z0-9_]+)\s*[:=]\s*([^\s,;]+)", line_clean)
        if not pairs:
            continue

        # Se a linha contem 2 ou mais pares (ex: Time:... Alt:... P:...)
        if len(pairs) >= 2 and not current_rec:
            line_dict = {}
            for k, v in pairs:
                try:
                    line_dict[k] = float(v) if "." in v else int(v)
                except ValueError:
                    if v in ["-", "--", "nan", "null", "none", ""]:
                        line_dict[k] = None
                    else:
                        line_dict[k] = v
            records.append(line_dict)
            continue

        for k, v in pairs:
            k_lower = k.lower()
            if k_lower in seen_keys and len(current_rec) >= 2:
                records.append(current_rec)
                current_rec = {}
                seen_keys = set()

            seen_keys.add(k_lower)
            try:
                if "." in v:
                    current_rec[k] = float(v)
                else:
                    current_rec[k] = int(v)
            except ValueError:
                if v in ["-", "--", "nan", "null", "none", ""]:
                    current_rec[k] = None
                else:
                    current_rec[k] = v

    if current_rec:
        records.append(current_rec)

    return pd.DataFrame(records)


def standardize_telemetry_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza colunas essenciais, assegura calculo de time_sec e cria aliases
    convenientes para analises de missao aeroespacial sem perder dados originais.
    Garante que altitude_gps_m, altitude_m, pressure_hpa, temperature_c, battery_v e time_sec
    estejam SEMPRE disponiveis para evitar KeyErrors no dashboard e no agente.
    """
    if df.empty:
        return df

    # Limpeza de nomes de colunas e reset do indice
    df = df.reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]

    # Mapeamento para nomes padronizados em minusculas
    col_lower_map = {c.lower(): c for c in df.columns}

    # 1. Alias e Garantia de Altitude (altitude_gps_m E altitude_m)
    alt_col_found = None
    for candidate in [
        "altitude_gps_m", "altitude_m", "alt", "altitude",
        "altitude_baro_m", "altb", "height", "alt_gps", "alt_baro"
    ]:
        if candidate in col_lower_map:
            alt_col_found = col_lower_map[candidate]
            break

    if alt_col_found:
        alt_series = pd.to_numeric(df[alt_col_found], errors="coerce")
        if alt_series.isna().all():
            alt_series = alt_series.fillna(0.0)
        else:
            alt_series = alt_series.interpolate().ffill().bfill().fillna(0.0)
        df["altitude_gps_m"] = alt_series
        df["altitude_m"] = alt_series
    else:
        df["altitude_gps_m"] = pd.Series(0.0, index=df.index, dtype=float)
        df["altitude_m"] = pd.Series(0.0, index=df.index, dtype=float)

    # 2. Alias de Altitude Barometrica
    baro_col_found = None
    for candidate in ["altitude_baro_m", "altb", "alt_baro", "altitude_baro"]:
        if candidate in col_lower_map:
            baro_col_found = col_lower_map[candidate]
            break
    if baro_col_found:
        baro_series = pd.to_numeric(df[baro_col_found], errors="coerce").interpolate().ffill().bfill().fillna(df["altitude_gps_m"])
        df["altitude_baro_m"] = baro_series
    else:
        df["altitude_baro_m"] = df["altitude_gps_m"].copy()

    # 3. Alias de Pressao (pressure_hpa)
    press_col_found = None
    for candidate in ["pressure_hpa", "pressure", "press", "baro", "p"]:
        if candidate in col_lower_map:
            press_col_found = col_lower_map[candidate]
            break
    if press_col_found:
        df["pressure_hpa"] = pd.to_numeric(df[press_col_found], errors="coerce").interpolate().ffill().bfill().fillna(1013.25)
    else:
        df["pressure_hpa"] = pd.Series(1013.25, index=df.index, dtype=float)

    # 4. Alias de Temperatura (temperature_c)
    temp_col_found = None
    for candidate in ["temperature_c", "temp_external_c", "t", "temp", "temperature", "t_ext"]:
        if candidate in col_lower_map:
            temp_col_found = col_lower_map[candidate]
            break
    if temp_col_found:
        df["temperature_c"] = pd.to_numeric(df[temp_col_found], errors="coerce").interpolate().ffill().bfill().fillna(25.0)
    else:
        df["temperature_c"] = pd.Series(25.0, index=df.index, dtype=float)

    # 5. Alias de Bateria (battery_v)
    bat_col_found = None
    for candidate in ["battery_v", "bat", "vbat", "battery", "v_bat", "volt", "voltage"]:
        if candidate in col_lower_map:
            bat_col_found = col_lower_map[candidate]
            break
    if bat_col_found:
        df["battery_v"] = pd.to_numeric(df[bat_col_found], errors="coerce").interpolate().ffill().bfill().fillna(4.2)
    else:
        df["battery_v"] = pd.Series(4.2, index=df.index, dtype=float)

    # 6. Alias de Tempo de Voo (time_sec)
    if "time_sec" not in df.columns:
        time_calculated = False
        if "time" in col_lower_map:
            actual_time_col = col_lower_map["time"]
            s = df[actual_time_col].astype(str)
            def hms_to_sec(val):
                try:
                    p = str(val).split(":")
                    if len(p) >= 3:
                        return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
                    elif len(p) == 2:
                        return int(p[0]) * 60 + float(p[1])
                    return np.nan
                except Exception:
                    return np.nan
            secs = s.apply(hms_to_sec)
            valid_secs = secs.dropna()
            if not valid_secs.empty:
                t0 = float(valid_secs.iloc[0])
                df["time_sec"] = secs - t0
                df["time_sec"] = df["time_sec"].apply(lambda v: v if (pd.isna(v) or v >= 0) else v + 86400)
                df["time_sec"] = df["time_sec"].interpolate().ffill().bfill().fillna(0.0)
                time_calculated = True
        elif "timestamp" in col_lower_map:
            actual_ts_col = col_lower_map["timestamp"]
            try:
                dt_series = pd.to_datetime(df[actual_ts_col], errors="coerce")
                valid_dt = dt_series.dropna()
                if not valid_dt.empty:
                    t0 = valid_dt.iloc[0]
                    df["time_sec"] = (dt_series - t0).dt.total_seconds().interpolate().ffill().bfill().fillna(0.0)
                    time_calculated = True
            except Exception:
                pass

        if not time_calculated or "time_sec" not in df.columns or df["time_sec"].isna().all():
            df["time_sec"] = np.arange(len(df), dtype=float)
    else:
        df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce").interpolate().ffill().bfill().fillna(0.0)

    # 7. Alias de GPS (latitude, longitude)
    for candidate in ["lat", "latitude"]:
        if candidate in col_lower_map:
            df["latitude"] = pd.to_numeric(df[col_lower_map[candidate]], errors="coerce")
            break
    for candidate in ["lon", "longitude", "lng"]:
        if candidate in col_lower_map:
            df["longitude"] = pd.to_numeric(df[col_lower_map[candidate]], errors="coerce")
            break

    # Conversao numerica geral com errors='coerce' para colunas de telemetria
    for col in df.columns:
        if col not in ["time", "timestamp", "callsign", "event", "note"]:
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].dropna()
            if not non_null.empty:
                valid_num_ratio = converted.dropna().shape[0] / non_null.shape[0]
                if valid_num_ratio >= 0.5:
                    df[col] = converted
            else:
                df[col] = converted

    return df


def load_telemetry(
    source: Any,
    filename: Optional[str] = None
) -> pd.DataFrame:
    """
    Funcao principal de ingestao universal.
    Aceita arquivo Streamlit UploadedFile, caminho em disco (str ou Path),
    string bruta com conteudo de log ou bytes.
    Identifica automaticamente se o arquivo e CSV delimitado, log de pacotes TXT ou chave-valor.
    """
    file_name = ""
    raw_text = ""

    # Caso Streamlit UploadedFile ou objeto com atributos name e read/getvalue
    if hasattr(source, "name") and (hasattr(source, "getvalue") or hasattr(source, "read")):
        file_name = str(source.name)
        if hasattr(source, "getvalue"):
            data_bytes = source.getvalue()
        else:
            data_bytes = source.read()
        raw_text = _decode_bytes(data_bytes)
    # Caso caminho de arquivo Path ou str existente em disco
    elif isinstance(source, (str, Path)) and "\n" not in str(source) and len(str(source)) < 260 and Path(source).is_file():
        path_obj = Path(source)
        file_name = path_obj.name
        with open(path_obj, "rb") as f:
            raw_text = _decode_bytes(f.read())
    # Caso string direta com conteudo do arquivo
    elif isinstance(source, str):
        file_name = filename or "log_telemetria.txt"
        raw_text = source
    # Caso bytes diretos
    elif isinstance(source, bytes):
        file_name = filename or "log_telemetria.txt"
        raw_text = _decode_bytes(source)
    else:
        raise ValueError(f"Origem de dados nao suportada: {type(source)}")

    if not raw_text.strip():
        raise ValueError("O arquivo de log fornecido esta vazio.")

    head_sample = raw_text[:3000]
    
    # Marcadores de log de pacotes de estacao (ex: PT2UNB, LoRa, APRS)
    is_packet_log = (
        "-------" in head_sample or
        "[ESTACAO]" in head_sample or
        "[STATION]" in head_sample or
        ("RSSI:" in head_sample and "SNR:" in head_sample) or
        ("Alt:" in head_sample and "\n" in head_sample and "Lat:" in head_sample)
    )

    df: Optional[pd.DataFrame] = None

    if is_packet_log:
        try:
            df = parse_packet_log(raw_text)
        except Exception:
            df = None

    # Se nao for log de pacotes ou nao tiver retornado dados suficientes
    if df is None or df.empty or len(df) < 1:
        # Tenta delimitadores comuns (, ; tab |)
        for sep_candidate in [None, ",", ";", "\t", "|"]:
            try:
                sio = io.StringIO(raw_text)
                if sep_candidate is None:
                    candidate_df = pd.read_csv(sio, sep=None, engine="python")
                else:
                    candidate_df = pd.read_csv(sio, sep=sep_candidate, engine="python")

                cols_have_syntax_chars = any(":" in str(c) or "=" in str(c) for c in candidate_df.columns)
                if candidate_df.shape[1] > 1 and len(candidate_df) > 0 and not cols_have_syntax_chars:
                    df = candidate_df
                    break
            except Exception:
                continue

    # Se ainda nao resolveu, tenta parser de chave-valor generico
    if df is None or df.empty or df.shape[1] <= 1:
        try:
            df = parse_generic_key_values(raw_text)
        except Exception:
            pass

    # Se ainda assim nao carregou, lanca excecao com diagnostico
    if df is None or df.empty:
        raise ValueError(
            f"Nao foi possivel interpretar a estrutura do arquivo '{file_name}'. "
            "Certifique-se de que o arquivo seja um CSV delimitado (, ; tab) ou log de pacotes TXT com telemetria."
        )

    # Padronizacao e aliases essenciais
    df = standardize_telemetry_dataframe(df)

    return df
