# 🚀 HireSense AI — Intelligent Resume Screening System

> **AI-powered resume analysis platform that automates candidate-job matching using NLP, semantic similarity, and ATS optimization.**

---

## 🌟 Overview

HireSense AI is an advanced **NLP-driven recruitment intelligence system** designed to streamline the hiring process by automatically evaluating resumes against job descriptions.

It leverages **BERT-based semantic similarity, keyword gap analysis, and ATS scoring** to deliver actionable insights for both recruiters and candidates.

---

## 🎯 Key Features

* 🔍 **Semantic Resume Matching**
  Uses **Sentence Transformers (BERT)** to evaluate contextual similarity between resumes and job descriptions.

* 📊 **ATS Scoring System**
  Generates a quantitative score based on keyword alignment and relevance.

* 🧠 **Keyword Gap Analysis**
  Identifies missing skills and suggests improvements for better job alignment.

* ⚡ **Real-time Resume Evaluation**
  Instant feedback on resume effectiveness and match strength.

* 📈 **Interactive Visual Insights**
  Displays matching scores and analysis using dynamic visualizations.

---

## 🏗️ System Architecture

```bash
User Input (Resume + JD)
        ↓
Text Preprocessing (NLP)
        ↓
Embedding Generation (BERT)
        ↓
Semantic Similarity Scoring
        ↓
ATS Scoring + Keyword Analysis
        ↓
Insights & Recommendations (UI)
```

---

## 🧪 Tech Stack

| Category        | Tools                                     |
| --------------- | ----------------------------------------- |
| Language        | Python                                    |
| NLP             | spaCy, Sentence Transformers (BERT)       |
| Data Processing | Pandas, NumPy                             |
| Visualization   | Plotly                                    |
| Frontend        | Streamlit                                 |
| ML Concepts     | Semantic Similarity, NLP, Text Embeddings |

---

## 📸 Demo

🔗 **Live App:** https://tinyurl.com/hiresense-ai

> ⚠️ *Note: App may take ~15–20 seconds to load (hosted on free tier).*

---

## 📂 Project Structure

```bash
HireSense-AI/
│
├── app.py                # Streamlit app entry point
├── model/               # NLP models & embeddings
├── utils/               # Helper functions
├── data/                # Sample resumes / job descriptions
├── requirements.txt     # Dependencies
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/MayankPandey09/HireSense-AI.git
cd HireSense-AI
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📊 How It Works

1. Upload Resume & Job Description
2. System processes text using NLP techniques
3. Generates embeddings using BERT
4. Computes semantic similarity score
5. Outputs:

   * ATS Score
   * Match Percentage
   * Keyword Suggestions

---

## 📈 Results & Impact

* ✅ Improved resume-job matching accuracy
* ⏱️ Reduced manual screening effort
* 📊 Enabled data-driven hiring decisions
* 🤖 Automated candidate evaluation pipeline

---

## 🚀 Future Enhancements

* 🔹 Multi-resume batch processing
* 🔹 Recruiter dashboard with analytics
* 🔹 Integration with job portals (LinkedIn APIs)
* 🔹 Fine-tuned domain-specific NLP models

---

## 👨‍💻 Author

**Mayank Shekhar Pandey**
📧 [mayankpandey3009@gmail.com](mailto:mayankpandey3009@gmail.com)
🔗 LinkedIn: https://linkedin.com/in/mayankpandey01
💻 GitHub: https://github.com/MayankPandey09

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!

---
