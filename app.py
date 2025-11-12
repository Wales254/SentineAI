from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import smtplib
from email.message import EmailMessage

# Configuration
MODEL_PATH = "model.joblib"
DATA_PATH = "data.csv"

# Email alert config (fill these with real values when deploying)
EMAIL_ALERTS = False  # set to True to enable email alerts
SMTP_HOST = "smtp.example.com"
SMTP_PORT = 587
SMTP_USER = "you@example.com"
SMTP_PASSWORD = "yourpassword"
ALERT_TO = "recipient@example.com"

app = Flask(__name__)
app.secret_key = "sentinel-secret-key"  # replace with a secure key in production

def train_and_save_model():
    # Simple dataset loader and model trainer
    df = pd.read_csv(DATA_PATH)
    # assume the last column is 'label' with 0 = normal, 1 = attack
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Training complete. Model report:")
    print(classification_report(y_test, preds))
    joblib.dump(model, MODEL_PATH)
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save_model()

model = load_model()

def send_alert(subject, body):
    # Sends an email alert if EMAIL_ALERTS True; otherwise prints to console
    if EMAIL_ALERTS:
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = SMTP_USER
            msg["To"] = ALERT_TO
            msg.set_content(body)
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls()
                s.login(SMTP_USER, SMTP_PASSWORD)
                s.send_message(msg)
            print("Email alert sent.")
        except Exception as e:
            print("Failed to send alert:", e)
    else:
        # console alert
        print("ALERT:", subject)
        print(body)


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    if request.method == "POST":
        # handle CSV upload
        f = request.files.get("file")
        if not f:
            flash("No file uploaded", "danger")
            return redirect(url_for("index"))
        try:
            df = pd.read_csv(f)
        except Exception as e:
            flash(f"Failed to read CSV: {e}", "danger")
            return redirect(url_for("index"))
        # Expect same features as training data
        needed_cols = list(pd.read_csv(DATA_PATH).drop(columns=["label"]).columns)
        if not all(col in df.columns for col in needed_cols):
            flash(f"Uploaded CSV must contain columns: {needed_cols}", "danger")
            return redirect(url_for("index"))
        X = df[needed_cols]
        preds = model.predict(X)
        df["prediction"] = preds
        # If any attack predictions, trigger alert
        attacks = df[df["prediction"] == 1]
        if len(attacks) > 0:
            subject = f"SentinelAI Alert: {len(attacks)} suspicious event(s) detected"
            body = f"Detected {len(attacks)} suspicious rows. Example rows:\\n\\n{attacks.head().to_csv(index=False)}"
            send_alert(subject, body)
        results = df.to_dict(orient="records")
    return render_template("index.html", results=results)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST JSON with list of records or a single record.
    Example: {"data": [{"feature1": 1, "feature2": 2, ...}, {...}]}
    """
    payload = request.get_json(force=True)
    if "data" not in payload:
        return jsonify({"error": "missing 'data' field"}), 400
    df = pd.DataFrame(payload["data"])
    needed_cols = list(pd.read_csv(DATA_PATH).drop(columns=["label"]).columns)
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        return jsonify({"error": f"missing columns: {missing}"}), 400
    X = df[needed_cols]
    preds = model.predict(X)
    df["prediction"] = preds
    # send alert if attack found
    if df["prediction"].sum() > 0:
        subject = f"SentinelAI Alert: {int(df['prediction'].sum())} suspicious event(s) detected via API"
        body = df[df["prediction"] == 1].to_csv(index=False)
        send_alert(subject, body)
    return jsonify({"predictions": df.to_dict(orient="records")})

if __name__ == "__main__":
    # Run local dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
