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
