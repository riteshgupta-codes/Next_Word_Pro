# 🚀 Next_Word_Pro — AI-Powered Text Generation Engine

> **Generate intelligent, context-aware text in real time using LSTM-based deep learning.**

🔴 **Live Demo:** https://nextwordpro.streamlit.app/


---

## ⚡ What This Project Does

**Next_Word_Pro** is an end-to-end **NLP system** that predicts and generates the next sequence of words based on user input.

It simulates how real-world systems like **smart keyboards, email autocomplete, and AI writing tools** work — built from scratch using deep learning.

---

## 🎯 Why This Project Matters

* Demonstrates **real-world NLP pipeline (not just model training)**
* Covers **data → model → deployment → UI**
* Shows ability to build **production-ready AI applications**
* Directly relevant to **AI/ML internship roles**

---

## ✨ Core Features

* 🧠 **Context-Aware Text Generation**
  LSTM captures long-term dependencies for meaningful predictions

* 📊 **Large Vocabulary Support**
  Trained on ~**9K unique tokens** for diverse outputs

* ⚡ **Real-Time Inference**
  Instant predictions via Streamlit UI

* 🎛️ **Dynamic Output Control**
  Generate 1 → N words using an interactive slider

* 🔁 **Recursive Generation Engine**
  Continuously feeds predictions back into input for sequence building

---

## 🧠 Model Architecture (Simplified)

```
Input Text → Tokenizer → Padded Sequence → LSTM → Dense → Softmax → Next Word
```

* **Sequence Length:** 745
* **Architecture:** LSTM + Dense
* **Loss:** Categorical Crossentropy
* **Optimizer:** Adam

---

## 🛠️ Tech Stack

* **Deep Learning:** TensorFlow, Keras
* **Frontend/UI:** Streamlit
* **Data Processing:** NumPy, Pandas
* **Model Storage:** Pickle / HDF5
* **Language:** Python 3.11

---

## 🚀 Run Locally

```bash
git clone https://github.com/riteshgupta-codes/Next_Word_Pro.git
cd Next_Word_Pro
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚙️ How It Works (Pipeline)

1. **Text Preprocessing** → Cleaning + Tokenization
2. **Sequence Creation** → Convert text → numerical format
3. **Padding** → Fixed-length input (745 tokens)
4. **Model Prediction** → LSTM learns context
5. **Softmax Output** → Probability over vocabulary
6. **Recursive Loop** → Builds multi-word output

---

## 📸 Demo Preview

<img width="1912" height="1019" alt="image" src="https://github.com/user-attachments/assets/a6aafa1a-1958-4b7c-b8d7-a2bb19c5ef28" />
<img width="958" height="881" alt="image" src="https://github.com/user-attachments/assets/d62919d2-e853-4d71-977e-d6e06f3c7b2e" />


---

## 📈 Future Improvements

* 🚀 Replace LSTM with **Transformer (GPT-style)**
* 🔍 Add **Beam Search decoding**
* 🌍 Multi-language support
* ☁️ Deploy with **API backend (FastAPI)**

---

## 💼 Skills Demonstrated

* Sequence Modeling (LSTM)
* NLP Preprocessing (Tokenization, Padding)
* Model Deployment (Streamlit)
* End-to-End ML System Design

---

## 📬 Contact

* 📧 [riteshgupta.eng@gmail.com](mailto:riteshgupta.eng@gmail.com)
* 💻 https://github.com/riteshgupta-codes

---



⭐ If you found this useful, consider giving it a star!
