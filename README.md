# 🚧 AI-Based Road Crack Detection & Severity Assessment System

An AI-powered road inspection system using YOLO11, computer vision, and MLOps tools to detect and classify road cracks and support data-driven road maintenance.

## 📌 Project Overview

Road cracks are often detected late through manual inspections, resulting in worsening pavement conditions and expensive repairs. This project aims to identify road cracks from images, display prediction results, and support severity assessment.

## 🎯 Objectives

- Detect road cracks using a pretrained YOLO11 model.
- Classify four crack types: alligator, edge, longitudinal, and transverse.
- Display detection results through a Streamlit web interface.
- Monitor application metrics and inference performance.
- Support experiment tracking, testing, and monitoring through MLOps tools.

## 🛠️ Technology Stack

- **Python** — Application development
- **YOLO11 (Ultralytics)** — Object detection
- **Streamlit** — Web interface
- **Prometheus** — Metrics collection
- **Grafana** — Monitoring dashboards
- **MLflow** — Experiment and run tracking
- **Docker** — Containerized supporting services
- **Pytest** — Automated testing
- **Alertmanager** — Monitoring alerts

## 🖥️ Application Workflow

1. Launch the Streamlit application.
2. Upload a road image.
3. Run the crack detection model.
4. View the detected crack regions and prediction results.
5. Review the severity assessment provided by the application.
6. Monitor application metrics through Prometheus and Grafana.

## ⚙️ Installation and Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ensure the trained model `best.pt` is available at the path expected by the application.

## ▶️ Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, typically `http://localhost:8501`.

## 📊 Monitoring and MLOps

The project includes Prometheus metrics collection and Grafana visualization, along with MLflow experiment tracking. Docker configurations manage supporting services, and Pytest is used to test project components.

Follow the monitoring configuration and setup instructions included in the repository.

## 📈 Business Impact

The project aims to support earlier crack detection, more consistent severity assessment, and better maintenance prioritization. Targets include at least 85% precision, 25% fewer major resurfacing interventions, and 15–25% lower rehabilitation costs per kilometer. These are intended targets, not verified outcomes.

## 🔮 Future Scope

- Evaluate detection performance on diverse road conditions.
- Improve severity assessment validation.
- Expand monitoring and alerting.
- Verify end-to-end data lineage integration.
- Explore deployment for road inspection workflows.

## 👨‍💻 Author

**Sanket Jayant Dhawade**

## ⚠️ Disclaimer

This project is a prototype intended to support road inspection. Its predictions should not replace professional engineering inspections or independently determine road safety.

