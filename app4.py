# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time
import queue
import threading
from datetime import datetime, timezone, timedelta
import plotly.graph_objs as go
import paho.mqtt.client as mqtt

# Optional: lightweight auto-refresh helper (install in requirements). If you don't want it, remove next import and the st_autorefresh call below.
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ---------------------------
# Config (edit if needed)
# ---------------------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_SENSOR = "perseus/iot/sensor/data"
TOPIC_OUTPUT = "perseus/iot/sensor/output"
MODEL_PATH = "model_SUBANDI.pkl"   # put the .pkl in same repo

# timezone GMT+7 helper
TZ = timezone(timedelta(hours=7))
def now_str():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# module-level queue used by MQTT thread (do NOT replace this with st.session_state inside callbacks)
# ---------------------------
GLOBAL_MQ = queue.Queue()

# ---------------------------
# Streamlit page setup
# ---------------------------

st.set_page_config(page_title="SUBANDI", layout="wide")
st.title("SUBANDI - Sistem Uji Banjir dan Deteksi Intensitas")

# ---------------------------
# session_state init (must be done before starting worker)
# ---------------------------
if "msg_queue" not in st.session_state:
    # expose the global queue in session_state so UI can read it
    st.session_state.msg_queue = GLOBAL_MQ

if "logs" not in st.session_state:
    st.session_state.logs = []         # list of dict rows

if "last" not in st.session_state:
    st.session_state.last = None

if "mqtt_thread_started" not in st.session_state:
    st.session_state.mqtt_thread_started = False

if "ml_model" not in st.session_state:
    st.session_state.ml_model = None

# ---------------------------
# Load Model (safe)
# ---------------------------
@st.cache_resource
def load_ml_model(path):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        # don't fail the app; just return None and show a warning in UI
        st.warning(f"Could not load ML model from {path}: {e}")
        return None

if st.session_state.ml_model is None:
    st.session_state.ml_model = load_ml_model(MODEL_PATH)
if st.session_state.ml_model:
    st.success(f"Model loaded: {MODEL_PATH}")
else:
    st.info("No ML model loaded. Upload iot_temp_model.pkl in repo to enable predictions.")

# ---------------------------
# MQTT callbacks (use GLOBAL_MQ, NOT st.session_state inside callbacks)
# ---------------------------
def _on_connect(client, userdata, flags, rc):
    try:
        client.subscribe(TOPIC_SENSOR)
    except Exception:
        pass
    # push connection status into queue
    GLOBAL_MQ.put({"_type": "status", "connected": (rc == 0), "ts": time.time()})

def _on_message(client, userdata, msg):
    payload = msg.payload.decode(errors="ignore")
    try:
        data = json.loads(payload)
    except Exception:
        # push raw payload if JSON parse fails
        GLOBAL_MQ.put({"_type": "raw", "payload": payload, "ts": time.time()})
        return

    # push structured sensor message
    GLOBAL_MQ.put({"_type": "sensor", "data": data, "ts": time.time(), "topic": msg.topic})

# ---------------------------
# Start MQTT thread (worker)
# ---------------------------
def start_mqtt_thread_once():
    def worker():
        client = mqtt.Client()
        client.on_connect = _on_connect
        client.on_message = _on_message
        # optional: configure username/password if needed:
        # client.username_pw_set(USER, PASS)
        while True:
            try:
                client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
                client.loop_forever()
            except Exception as e:
                # push error into queue so UI can show it
                GLOBAL_MQ.put({"_type": "error", "msg": f"MQTT worker error: {e}", "ts": time.time()})
                time.sleep(5)  # backoff then retry

    if not st.session_state.mqtt_thread_started:
        t = threading.Thread(target=worker, daemon=True, name="mqtt_worker")
        t.start()
        st.session_state.mqtt_thread_started = True
        time.sleep(0.05)

# start thread
start_mqtt_thread_once()

# ---------------------------
# Helper: model predict
# ---------------------------
def model_predict_status_and_conf(hum, suhu,jarak,beda):
    model = st.session_state.ml_model
    if model is None:
        return ("N/A", None)
    X = [[float(hum), float(suhu),float(jarak),float(beda)]]
    try:
        status = model.predict(X)[0]
    except Exception:
        status = "ERR"
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(np.max(model.predict_proba(X)))
        except Exception:
            prob = None
    return (status, prob)

# ---------------------------
# Helper: publish message
# ---------------------------
def publish_message(topic, message):
    """Helper function to publish messages reliably"""
    try:
        pubc = mqtt.Client()
        pubc.connect(MQTT_BROKER, MQTT_PORT, 60)
        # Start the network loop to handle connection
        pubc.loop_start()
        # Give it a moment to connect
        time.sleep(0.1)
        # Publish with QoS=1 for reliable delivery
        pubc.publish(topic, message, qos=1)
        # Give it a moment to send
        time.sleep(0.1)
        # Clean up
        pubc.loop_stop()
        pubc.disconnect()
        return True
    except Exception as e:
        st.error(f"Publish failed: {e}")
        return False

# ---------------------------
# Drain queue (process incoming msgs)
# ---------------------------
def process_queue():
    updated = False
    q = st.session_state.msg_queue
    while not q.empty():
        item = q.get()
        ttype = item.get("_type")
        if ttype == "status":
            # status - connection
            st.session_state.last_status = item.get("connected", False)
            updated = True
        elif ttype == "error":
            # show error
            st.error(item.get("msg"))
            updated = True
        elif ttype == "raw":
            row = {"ts": now_str(), "raw": item.get("payload")}
            st.session_state.logs.append(row)
            st.session_state.last = row
            updated = True
        elif ttype == "sensor":
            d = item.get("data", {})
            try:
                hum = float(d.get("hum"))
            except Exception:
                hum = None
            try:
                suhu = float(d.get("suhu"))
            except Exception:
                suhu = None
            try:
                jarak = float(d.get("jarak"))
            except Exception:
                jarak = None
            try:
                beda = float(d.get("beda"))
            except Exception:
                beda = None


            row = {
                "ts": datetime.fromtimestamp(item.get("ts", time.time()), TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "hum": hum,
                "suhu": suhu,
                "jarak":jarak,
                "beda":beda
            }

            # ML prediction
            if hum is not None and suhu is not None and jarak is not None:
                status, conf = model_predict_status_and_conf(hum, suhu, jarak, beda)
            else:
                status, conf = ("N/A", None)

            row["pred"] = status
            row["conf"] = conf

            

            # simple anomaly: low confidence or z-score on latest window
            anomaly = False
            if conf is not None and conf < 0.6:
                anomaly = True

            # z-score on temp using recent window
            suhus = [r["suhu"] for r in st.session_state.logs if r.get("suhu") is not None]
            window = suhus[-30:] if len(suhus) > 0 else []
            if len(window) >= 5 and suhu is not None:
                mean = float(np.mean(window))
                std = float(np.std(window, ddof=0))
                if std > 0:
                    z = abs((suhu - mean) / std)
                    if z >= 3.0:
                        anomaly = True

            row["anomaly"] = anomaly
            st.session_state.last = row
            st.session_state.logs.append(row)
            # keep bounded
            if len(st.session_state.logs) > 5000:
                st.session_state.logs = st.session_state.logs[-5000:]
            updated = True

            # Auto-publish alert back to ESP32
            if status == "banjir":
                print("banjir")
                publish_message(TOPIC_OUTPUT,"banjir2")
            elif status == "surut":
                print("surut")
                publish_message(TOPIC_OUTPUT,"hujan")
            elif status == "normal":
                print("normal")
                publish_message(TOPIC_OUTPUT,"normal")
            else:
                print("anomaly")
                publish_message(TOPIC_OUTPUT,"normal")



            
    return updated

# run once here to pick up immediately available messages
_ = process_queue()

# ---------------------------
# UI layout
# ---------------------------
# optionally auto refresh UI; requires streamlit-autorefresh in requirements
if HAS_AUTOREFRESH:
    st_autorefresh(interval=2000, limit=None, key="autorefresh")  # 2s refresh

left, right = st.columns([1, 2])

with left:
    st.header("Connection Status")
    st.write("Broker:", f"{MQTT_BROKER}:{MQTT_PORT}")
    connected = getattr(st.session_state, "last_status", None)
    st.metric("MQTT Connected", "Yes" if connected else "No")
    st.write("Topic:", TOPIC_SENSOR)
    st.markdown("---")

    st.header("Last Reading")
    if st.session_state.last:
        last = st.session_state.last
        st.write(f"Time: {last.get('ts')}")
        st.write(f"Hum : {last.get('hum')} %")
        st.write(f"Suhu: {last.get('suhu')} °C")
        st.write(f"Jarak: {last.get('jarak')} cm")
        st.write(f"Prediction: {last.get('pred')}")
        st.write(f"Confidence: {last.get('conf')}")
        st.write(f"Anomaly flag: {last.get('anomaly')}")
    else:
        st.info("Waiting for data...")

    st.markdown("---")
    #st.image("subandi.jpg")
with right:
    st.header("Live Chart (last 200 points)")
    df_plot = pd.DataFrame(st.session_state.logs[-200:])
    if (not df_plot.empty) and {"hum", "suhu","jarak"}.issubset(df_plot.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["hum"], mode="lines+markers", name="Hum (%)"))
        fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["suhu"], mode="lines+markers", name="Suhu (°C)", yaxis="y2"))
        #fig.add_trace(go.Scatter(x=df_plot["ts"], y=df_plot["jarak"], mode="lines+markers", name="Jarak (cm)", yaxis="y3"))
        fig.update_layout(
            yaxis=dict(title="Humidity (%)"),
            yaxis2=dict(title="Suhu (°C)", overlaying="y", side="right", showgrid=False),
            #yaxis3=dict(title="Jarak (cm)", overlaying="y", side="right", showgrid=False),
            height=520
        )
        # color markers by anomaly / label
        colors = []
        for _, r in df_plot.iterrows():
            if r.get("anomaly"):
                colors.append("magenta")
            else:
                lab = r.get("pred", "")
                if lab == "Banjir2":
                    colors.append("red")
                elif lab == "Hujan":
                    colors.append("green")
                elif lab == "Banjir":
                    colors.append("blue")
                else:
                    colors.append("gray")
        fig.update_traces(marker=dict(size=8, color=colors), selector=dict(mode="lines+markers"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Make sure ESP32 publishes to correct topic.")

    st.markdown("### Recent Logs")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs)[::-1].head(100))
    else:
        st.write("—")

# after UI render, drain queue (so next rerun shows fresh data)
process_queue()