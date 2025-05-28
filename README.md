# 🫁 LungCareAI

[Presentation](https://docs.google.com/presentation/d/1hbeTMh2d3SL4N1DGmcxghpmxNQgUBqrXrL0yUacq3kc/edit?usp=sharing)
🔗 **View the project pitch deck**

[Use webapp](https://gibbon-clever-bream.ngrok-free.app/LungCareAI)
🌐 **View web site**

# 📈 Architectural diagram

![LungCareAI Architecture Diagram](https://github.com/Jerryblessed/lungcareai/blob/main/static/AI%20Lung%20Cancer%20Diagnosis%20Flowchart.png?raw=true)

# ⭐️ Web screen

![LungCareAI Landing Page](https://github.com/Jerryblessed/LungCareAI/blob/main/static/landingpage.png?raw=true)

**LungCareAI** is a lightweight Flask application that democratizes sonar AI-powered LLM and lung cancer diagnosis using both deep learning and no-code tools. Built for low-resource settings, the platform enables non-technical health workers to upload CT or histopathology images via a simple web UI for instant predictions and guidance.

---

## 🚀 Features

* 🖼️ **Dual Image Upload**

  * Supports both CT and histopathology images.
  * Upload via drag-and-drop interface on the web app.
* 🧠 **High-Fidelity Deep Learning**

  * Backend runs a DenseNet121 model trained on **16,000+ images** for accurate, offline-ready inference.
* 💡 **No-Code Azure Option**

  * For clinics with limited technical capacity, 800 pre-labeled lung images were trained on **Azure Custom Vision** for an easy plug-and-play interface.
* 🤖 **AI Chatbot Assistant**

  * Integrated with sonar model from plerplexity to explain results, answer lung-health questions, and guide users through uploads.
* 🔐 **Secure Viewer Access**

  * Azure Custom Vision access managed via viewer lists—no exposed secrets.

---

## 🎯 Uniqueness

1. **Dual Training Paths**

   * A powerful deep learning model (DenseNet121) trained locally on 16K lung images.
   * A parallel no-code model trained on 800 curated samples using Azure Custom Vision for accessible cloud-based use.
2. **Two Modalities, One Platform**

   * Handles both CT scans and histopathology slides with equal ease.
3. **Offline-Ready Architecture**

   * Local model designed for containerized inference in remote clinics.
4. **Embedded AI Agent**

   * Helps interpret model outputs and provides clinical context in plain English.

---

## 🌍 Social Good

* **Bridging diagnostic gaps** in rural clinics (currently <15% imaging coverage).
* **Advancing SDG 3.4**: Early detection could help save over 600,000 lives annually.
* **Empowering non-specialists**: AI helps community health workers participate in diagnosis and triage.
* **Fostering collaboration**: Viewer access and open-source codebase encourage transparency and feedback.

---

## 🛠️ Getting Started

### Step 1: Download and install files

```bash
# Clone the repo
git clone https://github.com/Jerryblessed/breathsafe.git
cd LungCareAI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Download models from Google Drive

[Download models here](https://drive.google.com/drive/folders/17am-HyZ2R7SoCi9Rpfu5uK71XAWWI1ff?usp=drive_link)
🔗 **Download both CT Scan and Histopathology Models**

```bash
# Place both models in the root directory of the Flask app (same level as app.py)
```

### Step 3: Run the Flask app

```bash
# Run the app
python app.py
```

---

## 📁 Project Structure

Make sure your folder looks like this:

```
📁 LungCareAI/
│
├── app.py                          # Flask main application
├── ctscan_densenet121.keras       # Trained CT scan model
├── histo_densenet121_model.keras  # Trained histopathology model
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── static/                        # Static files (e.g., images, CSS)
├── templates/                     # HTML templates for Flask
└── train/                         # Model training scripts
```

Visit `http://localhost:5000` in your browser to explore.

---

## ✅ Training Models for LungCareAI

[🧠 Model Training Guide](https://github.com/Jerryblessed/breathsafe/tree/main/train)
🧩 **Learn how to train your own model for this project**

---

© 2025 Breath Safe Initiative
