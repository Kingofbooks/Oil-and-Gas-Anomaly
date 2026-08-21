# ⛽ Oil & Gas Real-Time Anomaly Detection System

An end-to-end real-time anomaly detection system for oil & gas sensor data.

The system ingests sensor readings through MQTT or REST APIs, processes the data using a **TranAD-based deep learning anomaly detection model**, stores predictions in PostgreSQL, generates operational alerts, and exposes the results through a FastAPI backend and interactive React dashboard.

---

# 🚀 Project Overview

Industrial oil and gas systems generate continuous streams of sensor data from wells and equipment.

Detecting abnormal behavior early is important because anomalies may indicate:

- Pressure abnormalities
- Equipment malfunction
- Valve failures
- Unexpected operational behavior
- Potential safety risks

This project implements a complete anomaly detection pipeline capable of processing streaming sensor data and identifying abnormal patterns in near real time.

---

# 🏗️ System Architecture

<img width="1368" height="945" alt="Image" src="https://github.com/user-attachments/assets/3d830edd-2f43-4496-a9b7-a74fa61bea48" />

---

# ✨ Features

## 📡 Real-Time Sensor Data Ingestion

Sensor readings can be ingested through:

- MQTT
- FastAPI REST API

Each sensor reading contains data from **22 industrial features**.

### Sensor Features

```text
ABER_CKGL
ABER_CKP
ESTADO_DHSV
ESTADO_M1
ESTADO_M2
ESTADO_PXO
ESTADO_SDV_GL
ESTADO_SDV_P
ESTADO_W1
ESTADO_W2
ESTADO_XO
P_ANULAR
P_JUS_CKGL
P_JUS_CKP
P_MON_CKP
P_PDG
P_TPT
QGL
T_JUS_CKP
T_MON_CKP
T_PDG
T_TPT
```

---

# 🧠 Machine Learning Anomaly Detection

The project uses a **TranAD-based anomaly detection model**.

The ML pipeline performs:

```text
Sensor Readings
       ↓
Feature Selection
       ↓
Preprocessing / Scaling
       ↓
Sliding Window Creation
       ↓
TranAD Model
       ↓
Anomaly Score
       ↓
Threshold Comparison
       ↓
Normal / Anomaly
```

## Model Configuration

| Property | Value |
|---|---|
| Model | TranAD |
| Model Version | demo-v1 |
| Features | 22 |
| Window Size | 120 |
| Threshold | 0.005 |

The system requires a sequence of sensor readings equal to the configured window size before performing inference.

---

# 📊 Ground Truth Dataset

The demo dataset contains:

- **Total Rows:** 30,000
- **Normal Samples:** 28,860
- **Anomaly Samples:** 1,140

The dataset contains five ground-truth anomaly events.

| Event | Row Range | Duration |
|---|---|---:|
| Event 1 | 6000 → 6179 | 180 rows |
| Event 2 | 12000 → 12239 | 240 rows |
| Event 3 | 18000 → 18299 | 300 rows |
| Event 4 | 24000 → 24119 | 120 rows |
| Event 5 | 27000 → 27299 | 300 rows |

---

# 📈 Exploratory Data Analysis

The sensor dataset contains periods of normal operation and several injected anomaly events.

The following visualization shows the distribution and location of anomaly events within the dataset.

## Ground Truth Anomaly Events

Add your EDA / anomaly visualization image here.

Place the image at:

```text
assets/
└── anomaly_events.png
```

Then add the image to the README:

```markdown
![Ground Truth Anomaly Events](assets/anomaly_events.png)
```

### Example

```text
Normal Data
███████████████████   Anomaly Event   ███████████████████
                            ▲
                            │
                     Abnormal Pattern
```

This visualization helps demonstrate where the known anomaly events occur and provides a comparison point for the ML model predictions.

---

# 🚨 Anomaly Detection Results

After running the complete dataset through the system:

- **Total ML Results:** 30,124
- **Normal Predictions:** 28,485
- **Anomaly Predictions:** 1,639
- **Normal Percentage:** 94.56%
- **Anomaly Percentage:** 5.44%
- **Highest Anomaly Score:** 51.78

## Example Anomaly Result

```text
Model: TranAD
Version: demo-v1
Anomaly Score: 1.739989
Threshold: 0.005
Prediction: ANOMALY
```

## Example Normal Prediction

```text
Model: TranAD
Version: demo-v1
Anomaly Score: 0.002687
Threshold: 0.005
Prediction: NORMAL
```

---

# 🚨 Alert Management

When an anomaly is detected, the system evaluates the anomaly score and creates an operational alert.

The alert system includes:

- Severity classification
- Open / resolved status
- Duplicate alert prevention
- Well identification
- Associated anomaly result
- Timestamp tracking

## Example Alert

```text
Well: WELL-001
Severity: MEDIUM
Status: OPEN

Message:
Anomaly detected for WELL-001

Score: 0.007567
Model: TranAD
```

## Duplicate Alert Prevention

The system intentionally does **not** create thousands of duplicate alerts when a continuous anomaly event occurs.

Instead:

```text
Anomaly Detected
       ↓
Check Existing OPEN Alert
       ↓
 ┌───────────────┐
 │ Alert Exists? │
 └───────┬───────┘
         │
     Yes │ No
       │   │
       ▼   ▼
Reuse Alert   Create Alert
```

This is why a large number of anomaly detections can result in only one active alert for a well.

For example:

```text
Anomaly Results: 1,639
Active Alerts: 1
```

This behavior prevents alert flooding during continuous anomaly periods.

---

# 🗄️ Database Architecture

The system uses **PostgreSQL**.

## `sensor_readings`

Stores incoming sensor data.

```text
sensor_readings
│
├── id
├── well_id
├── timestamp
├── received_at
├── source
└── 22 sensor features
```

## `anomaly_results`

Stores ML predictions.

```text
anomaly_results
│
├── id
├── reading_id
├── timestamp
├── model_name
├── model_version
├── anomaly_score
├── is_anomaly
└── processed_at
```

## `alerts`

Stores operational alerts.

```text
alerts
│
├── id
├── anomaly_result_id
├── well_id
├── created_at
├── severity
├── message
├── status
└── resolved_at
```

---

# ⚡ FastAPI Backend

The backend exposes REST APIs for interacting with the system.

## Dashboard Summary

```http
GET /dashboard/summary
```

Provides summary statistics for:

- Total sensor readings
- Total anomalies
- Normal readings
- Active alerts
- Latest anomaly information

---

## List Wells

```http
GET /wells
```

Returns available wells.

### Example Response

```json
[
  {
    "well_id": "WELL-001"
  }
]
```

---

## Submit Sensor Reading

```http
POST /readings
```

Allows sensor data to be submitted directly through the REST API.

### Example Request

```json
{
  "well_id": "WELL-001",
  "timestamp": "2026-08-19T14:20:00Z",
  "received_at": "2026-08-19T14:20:00Z",
  "source": "api-test",
  "ABER_CKGL": 0,
  "ABER_CKP": 0
}
```

---

## Direct ML Prediction

```http
POST /predict
```

Runs the TranAD model directly without requiring MQTT ingestion.

### Example Response

```json
{
  "model": "TranAD",
  "version": "demo-v1",
  "window_size": 120,
  "feature_count": 22,
  "anomaly_score": 1.739989,
  "threshold": 0.005,
  "is_anomaly": true,
  "predicted_at": "2026-08-19T18:38:03"
}
```

---

# 📡 MQTT Pipeline

The system supports real-time MQTT ingestion.

## Example Topic

```text
oilgas/WELL-001/sensors
```

## Pipeline

```text
MQTT Publisher
       ↓
MQTT Broker
       ↓
MQTT Subscriber
       ↓
Database
       ↓
ML Buffer
       ↓
TranAD
       ↓
Anomaly Result
       ↓
Alert System
```

The ML system uses a rolling buffer:

```text
Reading 1
Reading 2
Reading 3
...
Reading 120
       ↓
TranAD Inference
```

After the initial window is filled, new readings continue to be processed using a sliding window.

---

# 🖥️ Frontend Dashboard

The frontend is built using:

- React
- Vite
- Axios
- Recharts
- Lucide React

The dashboard visualizes:

- Total sensor readings
- Total anomalies
- Normal readings
- Active alerts
- Anomaly scores
- Recent anomaly events
- Well information

---

# 🛠️ Tech Stack

## Backend

- Python
- FastAPI
- SQLAlchemy
- PostgreSQL
- Uvicorn

## Machine Learning

- PyTorch
- TranAD
- NumPy
- Pandas
- Scikit-learn

## Streaming

- MQTT
- Paho MQTT

## Frontend

- React
- Vite
- Axios
- Recharts
- Lucide React

---

# 📂 Project Structure

```text
oil-gas-anomaly/
│
├── api/
│   ├── routes/
│   │   ├── wells.py
│   │   ├── readings.py
│   │   ├── dashboard.py
│   │   └── predict.py
│   └── main.py
│
├── database/
│   ├── connection.py
│   ├── models.py
│   └── repository.py
│
├── ml/
│   ├── tranad_detector.py
│   ├── preprocessor.py
│   └── predict.py
│
├── mqtt/
│   ├── publisher.py
│   └── subscriber.py
│
├── artifacts/
│   └── model artifacts
│
├── assets/
│   └── anomaly_events.png
│
├── oil-gas-dashboard/
│   ├── src/
│   ├── components/
│   └── services/
│
└── README.md
```

---

# ▶️ Running the Backend

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run FastAPI:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# ▶️ Running the Frontend

Navigate to the frontend directory:

```bash
cd oil-gas-dashboard
```

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

---

# 🧪 Testing the ML API

Example PowerShell request:

```powershell
$body = Get-Content .\normal_predict.json -Raw

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/predict" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

## Example Anomaly Response

```text
model         : TranAD
version       : demo-v1
window_size   : 120
feature_count : 22
anomaly_score : 1.739989
threshold     : 0.005
is_anomaly    : True
```

---

# 🔄 Complete Data Flow

```text
                         ┌──────────────┐
                         │ Sensor Data  │
                         └──────┬───────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ MQTT / FastAPI Input  │
                    └───────────┬───────────┘
                                │
                                ▼
                       ┌────────────────┐
                       │   PostgreSQL   │
                       │ Sensor Readings│
                       └────────┬───────┘
                                │
                                ▼
                       ┌────────────────┐
                       │ Sliding Window │
                       │      120       │
                       └────────┬───────┘
                                │
                                ▼
                       ┌────────────────┐
                       │     TranAD     │
                       │ Anomaly Model  │
                       └────────┬───────┘
                                │
                                ▼
                       ┌────────────────┐
                       │ Anomaly Result │
                       └────────┬───────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │ Is Anomaly Detected?   │
                    └───────────┬────────────┘
                                │
                        ┌───────┴───────┐
                        │               │
                      NORMAL         ANOMALY
                        │               │
                        ▼               ▼
                  Store Result      Check Alerts
                                        │
                                        ▼
                                  Create / Reuse
                                      Alert
                                        │
                                        ▼
                                    Dashboard
```

---

# 🎯 Key Learning Outcomes

This project demonstrates how to build a complete AI-powered monitoring system combining:

- Real-time data streaming
- Deep learning anomaly detection
- Time-series processing
- Sliding window inference
- PostgreSQL database design
- REST APIs
- MQTT communication
- Alert management
- React data visualization
- End-to-end ML system integration

---

# 🔮 Future Improvements

Possible improvements include:

- Multiple well support
- Automatic alert resolution
- WebSocket real-time dashboard updates
- Role-based authentication
- Historical anomaly analytics
- Model monitoring
- Dynamic anomaly thresholds
- Retraining pipeline
- Docker containerization
- Cloud deployment
- Kubernetes deployment
- Grafana monitoring integration

---

# 👨‍💻 Author

**Aryan Sharma**

AI/ML Engineer | Machine Learning | Deep Learning | Agentic AI

---

# ⭐ Project Status

🚧 **Active Development**

The core pipeline is functional:

- [x] Sensor Data Ingestion
- [x] MQTT Streaming
- [x] PostgreSQL Storage
- [x] TranAD Anomaly Detection
- [x] Sliding Window Processing
- [x] Anomaly Result Storage
- [x] Alert Management
- [x] FastAPI Backend
- [x] REST API Testing
- [x] React Dashboard

---
