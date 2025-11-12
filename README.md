
# SentinelAI MVP (Cyber Intelligence) - Flask Project

This MVP demonstrates a simple cyber threat detection pipeline using a Logistic Regression model and a Flask dashboard.
It can run locally and is structured to be deployable on hosting platforms like Render or Railway.

## Files
- app.py - Flask application (loads/trains model, serves dashboard, API and alerting)
- data.csv - example dataset used to train the model
- model.joblib - created after first run (auto-generated)
- templates/index.html - dashboard page
- static/style.css - styling
- requirements.txt - Python dependencies

## Quick start (local)
1. Create a python virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate    # Windows (powershell)
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open http://127.0.0.1:5000 in your browser and upload `data.csv` as a test input.

## Email alerts
Email alerts are disabled by default. To enable them, edit `app.py` and set `EMAIL_ALERTS = True`, and fill in `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` and `ALERT_TO` with valid values.

## Deployment
- For Render: create a new web service, set the start command `gunicorn app:app`, add necessary environment variables and push files.
- For Railway/Heroku: similar steps (Procfile with web: gunicorn app:app).

