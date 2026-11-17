import datetime
import numpy as np
import pandas as pd

def generate_flight_data(output_path: str):
    np.random.seed(42)

    # Time setup: 0 to 9000 seconds with step of 2s (4501 points) ~ 2.5 hours
    t = np.arange(0, 9002, 2)
    n = len(t)

    t_launch = 120   # launch at 2 min (120s)
    t_burst = 6200   # burst at ~1h43m (6200s)
    t_land = 8900    # landing at ~2h28m (8900s)

    alt_ground = 645.0
    alt_burst = 31842.0
    alt_land = 582.0

    # 1. Altitude Profile (GPS)
    altitude_gps = np.zeros(n)
    for i, time in enumerate(t):
        if time < t_launch:
            altitude_gps[i] = alt_ground + np.random.normal(0, 0.4)
        elif time <= t_burst:
            # Ascent with realistic aerodynamic slight non-linearity
            progress = (time - t_launch) / (t_burst - t_launch)
            altitude_gps[i] = alt_ground + (alt_burst - alt_ground) * (progress ** 1.04) + np.random.normal(0, 1.2)
        elif time <= t_land:
            # Descent: high velocity in low density atmosphere, parachute decelerates exponentially
            progress = (time - t_burst) / (t_land - t_burst)
            altitude_gps[i] = alt_burst - (alt_burst - alt_land) * (1.0 - np.exp(-3.2 * progress)) / (1.0 - np.exp(-3.2)) + np.random.normal(0, 1.5)
        else:
            altitude_gps[i] = alt_land + np.random.normal(0, 0.3)

    # 2. Atmospheric Pressure (Barometric formula with scale height H ~ 7250m)
    p_ground = 938.5
    pressure = p_ground * np.exp(-altitude_gps / 7250.0)
    # Simulated hardware anomaly: Barometer stalling / freezing near apogee (<10 hPa)
    for i, time in enumerate(t):
        if 5920 <= time <= 6180:
            # Sensor plateau / stalling anomaly due to extreme vacuum / cold
            pressure[i] = 8.62 + np.random.normal(0, 0.015)
        else:
            pressure[i] += np.random.normal(0, 0.12)
    pressure = np.clip(pressure, 8.4, 1013.25)

    # Barometric altitude derived from pressure
    altitude_baro = 44330.0 * (1.0 - (pressure / 1013.25) ** 0.1903)

    # 3. Thermal Profile (External vs Internal)
    temp_ext = np.zeros(n)
    for i, h in enumerate(altitude_gps):
        if h < 11000:
            # Troposphere lapse rate: -6.5 C / 1000m
            temp_ext[i] = 23.5 - 0.0065 * h + np.random.normal(0, 0.4)
        elif h < 20000:
            # Tropopause minimum (-58.5 C)
            dip = -58.5 + 4.2 * ((h - 13500) / 6500) ** 2
            temp_ext[i] = dip + np.random.normal(0, 0.5)
        else:
            # Stratosphere inversion warming (ozone layer)
            temp_ext[i] = -52.0 + 0.0024 * (h - 20000) + np.random.normal(0, 0.6)

    # Internal temperature (payload enclosure with foam insulation + electronic self-heating)
    temp_int = np.zeros(n)
    curr_tint = 24.5
    for i, time in enumerate(t):
        if i == 0:
            temp_int[i] = curr_tint
        else:
            dt = 2.0
            q_gen = 0.014  # internal heat dissipation
            k_loss = 0.00038  # thermal transmission coefficient
            curr_tint += (-k_loss * (curr_tint - temp_ext[i]) + q_gen) * dt
            temp_int[i] = curr_tint + np.random.normal(0, 0.08)

    # 4. Battery Voltage (2S LiPo nominal 8.4V with cold sag anomaly)
    battery_v = np.zeros(n)
    v_nominal = 8.38
    for i, time in enumerate(t):
        v_base = v_nominal - 0.72 * (time / 9000.0)
        # Thermal battery sag during extreme cold soak
        temp_sag = 0.0
        if temp_int[i] < 4.0:
            temp_sag = 0.09 * (4.0 - temp_int[i])
        battery_v[i] = v_base - temp_sag + np.random.normal(0, 0.012)

    # 5. IMU: Gyroscope (DPS) & Accelerometer (G)
    gyro_x = np.random.normal(0, 1.1, n)
    gyro_y = np.random.normal(0, 1.1, n)
    gyro_z = np.random.normal(1.2, 1.4, n)  # slight natural tether twisting

    accel_x = np.random.normal(0, 0.04, n)
    accel_y = np.random.normal(0, 0.04, n)
    accel_z = np.random.normal(1.0, 0.03, n)  # 1g static

    for i, time in enumerate(t):
        # Tropopause turbulence (jet stream wind shear between 11km and 14km)
        if 1800 <= time <= 2550:
            shear_x = np.sin((time - 1800) / 90.0) * 28.0
            shear_y = np.cos((time - 1800) / 75.0) * 24.0
            gyro_x[i] += shear_x + np.random.normal(0, 9.0)
            gyro_y[i] += shear_y + np.random.normal(0, 8.0)
            gyro_z[i] += np.random.normal(0, 16.0)
            accel_z[i] += np.random.normal(0, 0.28)

        # Burst moment (t = 6200s): violent balloon rupture and tumble
        if 6196 <= time <= 6224:
            gyro_x[i] = np.random.uniform(-310, 320)
            gyro_y[i] = np.random.uniform(-330, 340)
            gyro_z[i] = np.random.uniform(-270, 360)
            accel_z[i] = np.random.uniform(-2.9, -1.3)  # Freefall weightlessness
        elif 6224 < time <= 6750:
            # Post-burst turbulent descent damping
            decay = np.exp(-(time - 6224) / 280.0)
            gyro_x[i] += np.random.normal(0, 70.0 * decay)
            gyro_y[i] += np.random.normal(0, 70.0 * decay)
            gyro_z[i] += np.random.normal(0, 90.0 * decay)
            accel_z[i] += np.random.normal(0, 0.45 * decay)

        # Landing touchdown impact shock
        if 8896 <= time <= 8904:
            accel_z[i] = 3.82 + np.random.normal(0, 0.25)
            gyro_x[i] = np.random.normal(0, 210)
        elif time > t_land:
            gyro_x[i] = np.random.normal(0, 0.04)
            gyro_y[i] = np.random.normal(0, 0.04)
            gyro_z[i] = np.random.normal(0, 0.04)
            accel_z[i] = 1.0 + np.random.normal(0, 0.01)

    # 6. Trajectory: Coordinates (Campinas -> Mogi Mirim / Itapira region)
    lat0, lon0 = -22.9056, -47.0608
    lat = np.zeros(n)
    lon = np.zeros(n)
    curr_lat, curr_lon = lat0, lon0
    for i, time in enumerate(t):
        h = altitude_gps[i]
        if time < t_launch:
            pass
        elif time <= t_burst:
            curr_lat += 0.000088 * (1.0 + 0.25 * np.sin(time / 600))
            curr_lon += 0.000180 * (1.0 + 0.45 * (h / 30000))
        elif time <= t_land:
            curr_lat += 0.000062
            curr_lon += 0.000115
        lat[i] = curr_lat + np.random.normal(0, 0.00004)
        lon[i] = curr_lon + np.random.normal(0, 0.00004)

    # 7. Timestamps and Packets (with simulated MicroSD buffer lag / packet drop)
    base_time = datetime.datetime(2026, 8, 15, 8, 30, 0)
    timestamps = [(base_time + datetime.timedelta(seconds=int(sec))).isoformat() + "Z" for sec in t]

    packet_id = np.zeros(n, dtype=int)
    pid = 1000
    for i, time in enumerate(t):
        if 3420 <= time <= 3432:
            # 6 packets dropped during high write latency
            pid += 2
        else:
            pid += 1
        packet_id[i] = pid

    # 8. Radio Telemetry RSSI (dBm)
    ground_dist_km = np.sqrt(((lat - lat0) * 111) ** 2 + ((lon - lon0) * 102) ** 2)
    los_dist_km = np.sqrt(ground_dist_km ** 2 + (altitude_gps / 1000.0) ** 2)
    rssi_dbm = -52.0 - 19.5 * np.log10(np.maximum(1.0, los_dist_km)) + np.random.normal(0, 2.2)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "time_sec": t,
        "packet_id": packet_id,
        "altitude_gps_m": np.round(altitude_gps, 1),
        "altitude_baro_m": np.round(altitude_baro, 1),
        "pressure_hpa": np.round(pressure, 2),
        "temp_external_c": np.round(temp_ext, 1),
        "temp_internal_c": np.round(temp_int, 1),
        "battery_v": np.round(battery_v, 2),
        "gyro_x_dps": np.round(gyro_x, 2),
        "gyro_y_dps": np.round(gyro_y, 2),
        "gyro_z_dps": np.round(gyro_z, 2),
        "accel_x_g": np.round(accel_x, 3),
        "accel_y_g": np.round(accel_y, 3),
        "accel_z_g": np.round(accel_z, 3),
        "latitude": np.round(lat, 5),
        "longitude": np.round(lon, 5),
        "rssi_dbm": np.round(rssi_dbm, 1)
    })

    df.to_csv(output_path, index=False)
    print(f"File created: {output_path} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/sample_flight.csv"
    generate_flight_data(out)
