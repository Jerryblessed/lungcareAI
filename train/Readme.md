## 🚀 Train Your Own Model from Scratch

You can train your own image classification models using **Google Colab** or your **local machine**.

![Training Process](https://github.com/Jerryblessed/LungCareAI/blob/main/static/train.png?raw=true)

### 📌 On Google Colab

1. **Upload training scripts** (e.g., `ct_train.py` or `histo_train.py`) to Colab.

2. **Run the files individually**

   * Each script will automatically download the required dataset from **Kaggle** and train a model.
   * Models will be saved in `.keras` format.

   Example:

   ```bash
   !python ct_train.py
   !python histo_train.py
   ```

3. **Get the trained models**

   * `ct_train.py` → Outputs: `densenet121.keras`
   * `histo_train.py` → Outputs:

     * `densenet121.keras`
     * `mobilenetv2.keras`
     * `inceptionv3.keras`

4. **Rename models for compatibility with this project**

   * Use `densenet121.keras` from both scripts
   * Rename them as:

     * `ctscan_densenet121.keras`
     * `histo_densenet121_model.keras`

### 🖥️ On Your Local Computer

* **Log in** securely via a role-based viewer list.
* **Upload** CT or histopathology images (`Benign`, `Adenocarcinoma`, `Squamous Cell Carcinoma`).
* **Receive** AI predictions and confidence scores instantly.
* **Interact** with an embedded chatbot for explanations and clinical guidance.

All model training, evaluation, and predictions can be managed via **Azure’s portal** — no programming experience needed for clinicians.

---

