import streamlit as st
import plotly.graph_objects as pgo
from PyPDF2 import PdfReader
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
from collections import Counter
import math
import time
import base64
import requests
from streamlit_lottie import st_lottie

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HireSense AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.sidebar.markdown('### ⚙️ AI Settings')
if 'gemini_key' not in st.session_state:
    st.session_state.gemini_key = ""
st.session_state.gemini_key = st.sidebar.text_input("Gemini API Key (Optional)", value=st.session_state.gemini_key, type="password", help="Enter key to unlock Deep AI Feedback")

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")
nlp = load_nlp()

# ══════════════════════════════════════════════════════════
# SKILL TAXONOMY
# ══════════════════════════════════════════════════════════
SKILL_CATEGORIES = {
    "Programming & Query": [
        "python", "r", "sql", "java", "scala", "julia", "matlab",
        "bash", "shell", "c++", "javascript", "typescript", "go",
        "pyspark", "hive", "spark sql", "nosql", "t-sql", "pl/sql",
    ],
    "Machine Learning & AI": [
        "machine learning", "deep learning", "supervised learning",
        "unsupervised learning", "semi-supervised", "reinforcement learning",
        "transfer learning", "few-shot learning", "zero-shot learning",
        "natural language processing", "nlp", "computer vision",
        "generative ai", "large language models", "llm", "gpt",
        "regression", "classification", "clustering", "neural networks",
        "random forest", "gradient boosting", "xgboost", "lightgbm", "catboost",
        "svm", "support vector machine", "decision tree", "ensemble methods",
        "bagging", "boosting", "logistic regression", "linear regression",
        "naive bayes", "knn", "k-nearest", "anomaly detection",
        "time series", "forecasting", "arima", "lstm", "transformer",
        "bert", "attention mechanism", "word2vec", "embeddings",
        "object detection", "image classification", "segmentation",
        "recommendation system", "collaborative filtering",
        "dimensionality reduction", "pca", "t-sne", "umap",
        "hyperparameter tuning", "cross validation", "overfitting",
        "regularization", "dropout", "batch normalization",
        "model evaluation", "roc auc", "f1 score", "precision", "recall",
        "confusion matrix", "feature importance", "shap", "lime",
        "explainable ai", "xai", "model interpretability",
        "a/b testing", "hypothesis testing", "statistical inference",
        "bayesian", "markov", "monte carlo",
    ],
    "Data Science & Analysis": [
        "data science", "data analysis", "exploratory data analysis", "eda",
        "statistical analysis", "data wrangling", "data cleaning",
        "feature engineering", "feature selection", "data preprocessing",
        "data transformation", "data imputation", "outlier detection",
        "correlation analysis", "regression analysis", "predictive modeling",
        "descriptive statistics", "inferential statistics",
        "probability", "distributions", "central limit theorem",
        "anova", "chi-square", "t-test", "p-value", "confidence interval",
        "data storytelling", "insight generation", "kpi",
        "cohort analysis", "funnel analysis", "segmentation",
        "customer analytics", "churn prediction", "lifetime value",
        "demand forecasting", "fraud detection", "risk modeling",
    ],
    "Frameworks & Libraries": [
        "pandas", "numpy", "scikit-learn", "sklearn", "scipy",
        "tensorflow", "keras", "pytorch", "jax",
        "hugging face", "transformers", "langchain", "openai",
        "xgboost", "lightgbm", "catboost", "optuna", "ray tune",
        "mlflow", "wandb", "weights and biases", "neptune",
        "fastapi", "flask", "django", "streamlit", "gradio",
        "spark", "pyspark", "dask", "polars", "vaex",
        "nltk", "spacy", "gensim", "textblob",
        "opencv", "pillow", "torchvision",
        "statsmodels", "pingouin", "lifelines",
    ],
    "Data Engineering & Databases": [
        "sql", "mysql", "postgresql", "sqlite", "oracle",
        "mongodb", "cassandra", "redis", "elasticsearch", "neo4j",
        "snowflake", "databricks", "bigquery", "redshift", "synapse",
        "dbt", "apache airflow", "airflow", "prefect", "luigi",
        "kafka", "rabbitmq", "kinesis", "pubsub",
        "hadoop", "hive", "hbase", "presto", "trino",
        "etl", "elt", "data pipeline", "data warehouse", "data lake",
        "data lakehouse", "delta lake", "iceberg",
        "data modeling", "star schema", "dimensional modeling",
        "data governance", "data quality", "data catalog",
        "real-time data", "streaming data", "batch processing",
    ],
    "Visualization & BI Tools": [
        "tableau", "power bi", "looker", "metabase", "superset",
        "matplotlib", "seaborn", "plotly", "bokeh", "altair",
        "ggplot", "d3.js", "grafana", "kibana",
        "google data studio", "google looker studio",
        "excel", "google sheets", "pivot table",
        "dashboard", "data visualization", "reporting",
        "storytelling with data",
    ],
    "Cloud & MLOps": [
        "aws", "amazon web services", "azure", "gcp", "google cloud",
        "sagemaker", "azure ml", "vertex ai", "databricks",
        "docker", "kubernetes", "containerization",
        "ci/cd", "github actions", "jenkins", "gitlab ci",
        "terraform", "ansible", "infrastructure as code",
        "mlflow", "kubeflow", "bentoml", "seldon", "torchserve",
        "model serving", "model monitoring", "data drift",
        "feature store", "model registry",
        "lambda", "ec2", "s3", "gcs", "blob storage",
        "linux", "git", "github", "version control",
    ],
}

ALL_SKILLS_FLAT = [s for cat in SKILL_CATEGORIES.values() for s in cat]

SECTION_PATTERNS = {
    "Contact":        r"(email|phone|mobile|linkedin|github|address|contact|portfolio)",
    "Summary":        r"(summary|objective|profile|about me|career goal|overview)",
    "Experience":     r"(experience|employment|work history|internship|intern|position|role|job)",
    "Education":      r"(education|academic|degree|bachelor|master|b\.tech|m\.tech|b\.sc|m\.sc|phd|university|college|institute)",
    "Skills":         r"(skills|technical skills|competencies|tools|technologies|stack|proficiency)",
    "Projects":       r"(projects|portfolio|case stud|capstone|personal project|academic project)",
    "Certifications": r"(certif|credential|course|training|nanodegree|bootcamp|udemy|coursera|edx|deeplearning\.ai)",
    "Achievements":   r"(achievement|award|honor|recognition|publication|paper|research|hackathon|competition|kaggle|rank)",
}

# ══════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════
def extract_text(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text.strip()

def normalize(text):
    t = text.lower()
    t = re.sub(r'[-]', ' ', t)
    replacements = {
        r'\bml\b': 'machine learning ml',
        r'\bai\b': 'artificial intelligence ai',
        r'\bds\b': 'data science ds',
        r'\bnlp\b': 'natural language processing nlp',
        r'\bcv\b': 'computer vision cv',
        r'\bllm\b': 'large language models llm',
        r'\bsklearn\b': 'scikit learn sklearn',
        r'\bsk-learn\b': 'scikit learn sklearn',
        r'\bscikit-learn\b': 'scikit learn',
        r'\btf\b': 'tensorflow tf',
        r'\bxgb\b': 'xgboost xgb',
        r'\blgbm\b': 'lightgbm lgbm',
        r'\brnn\b': 'recurrent neural network rnn',
        r'\bcnn\b': 'convolutional neural network cnn',
        r'\bgbm\b': 'gradient boosting gbm',
        r'\brf\b': 'random forest rf',
        r'\bpca\b': 'principal component analysis pca',
        r'\beda\b': 'exploratory data analysis eda',
        r'\bpowerbi\b': 'power bi',
        r'\bhuggingface\b': 'hugging face',
    }
    for pattern, replacement in replacements.items():
        t = re.sub(pattern, replacement, t)
    return t

def preprocess_for_tfidf(text):
    text = normalize(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc
              if not token.is_stop and token.is_alpha and len(token) > 2]
    return " ".join(tokens)

def tfidf_score(resume_raw, jd_raw):
    resume = preprocess_for_tfidf(resume_raw)
    jd     = preprocess_for_tfidf(jd_raw)
    scores = []
    weights = [0.2, 0.5, 0.3]
    for ngram in [(1,1),(1,2),(1,3)]:
        vec = TfidfVectorizer(ngram_range=ngram, min_df=1)
        try:
            v = vec.fit_transform([resume, jd])
            s = cosine_similarity(v[0], v[1])[0][0]
            scores.append(s)
        except Exception:
            pass
    if not scores:
        return 0.0
    final = sum(s * w for s, w in zip(scores, weights[:len(scores)]))
    return round(final * 100, 2)

def skill_in_text(skill, text_norm):
    if skill in text_norm:
        return True
    if skill.replace(' ', '-') in text_norm:
        return True
    ns = skill.replace(' ', '')
    if len(ns) > 4 and ns in text_norm.replace(' ', ''):
        return True
    return False

def skill_analysis(resume_raw, jd_raw):
    resume_norm = normalize(resume_raw)
    jd_norm     = normalize(jd_raw)
    category_results = {}
    all_jd, all_matched, all_missing = [], [], []
    for category, skills in SKILL_CATEGORIES.items():
        jd_skills = [s for s in skills if skill_in_text(s, jd_norm)]
        matched   = [s for s in jd_skills if skill_in_text(s, resume_norm)]
        missing   = [s for s in jd_skills if s not in matched]
        all_jd.extend(jd_skills)
        all_matched.extend(matched)
        all_missing.extend(missing)
        if jd_skills:
            category_results[category] = {
                "matched": matched, "missing": missing,
                "score": round((len(matched) / len(jd_skills)) * 100, 1),
            }
    bonus = [s for s in ALL_SKILLS_FLAT if skill_in_text(s, resume_norm) and s not in all_jd]
    overall = round((len(all_matched) / max(len(all_jd), 1)) * 100, 2)
    return overall, category_results, all_matched, all_missing, bonus

DS_PROJECT_SIGNALS = [
    r'kaggle', r'github\.com', r'notebook', r'jupyter',
    r'dataset', r'trained\s+a?\s*model', r'built\s+a?\s*model',
    r'accuracy\s+of\s+\d+', r'\d+%\s*accuracy', r'f1\s*score',
    r'deployed', r'end.to.end', r'capstone', r'thesis',
    r'research\s+paper', r'published', r'ieee', r'arxiv',
    r'hackathon', r'competition', r'top\s+\d+%', r'winner',
    r'dashboard', r'visualization', r'insight', r'\beda\b',
    r'exploratory', r'predicted', r'forecasted', r'classified',
    r'clustered', r'segmented', r'analyzed', r'processed',
    r'\d+[,\d]*\s*(rows|records|samples|data points|observations)',
    r'improved\s+by\s+\d+', r'reduced\s+by\s+\d+',
    r'increased\s+by\s+\d+', r'achieved\s+\d+',
]

def ds_project_bonus(resume_text):
    text_lower = resume_text.lower()
    hits = sum(1 for p in DS_PROJECT_SIGNALS if re.search(p, text_lower))
    return min(round((hits / 10) * 20, 1), 20), hits

def quantification_score(resume_text):
    metrics = re.findall(
        r'\b\d+(?:\.\d+)?%|\b\d{4,}\b|\b\d+x\b|\b\d+\s*(times|hours|days|weeks|months|years|users|records|rows|models|projects)',
        resume_text.lower()
    )
    count = len(metrics)
    if count >= 8: return 100, count
    if count >= 5: return 75, count
    if count >= 3: return 50, count
    if count >= 1: return 25, count
    return 0, 0

def detect_sections(text):
    tl = text.lower()
    return {s: bool(re.search(p, tl)) for s, p in SECTION_PATTERNS.items()}

def extract_contact_info(text):
    return {
        "Email":    bool(re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', text)),
        "Phone":    bool(re.search(r'(\+?\d[\d\s\-().]{8,}\d)', text)),
        "LinkedIn": bool(re.search(r'linkedin\.com/in/[\w-]+', text, re.IGNORECASE)),
        "GitHub":   bool(re.search(r'github\.com/[\w-]+', text, re.IGNORECASE)),
    }

def ats_score(resume_text, sections):
    score, reasons, issues = 0, [], []
    for s in ["Experience", "Education", "Skills"]:
        if sections.get(s):
            score += 10; reasons.append(f"'{s}' section present")
        else:
            issues.append(f"Missing '{s}' section — ATS may skip your resume")
    for s in ["Summary", "Projects", "Certifications", "Achievements"]:
        if sections.get(s):
            score += 5; reasons.append(f"'{s}' section found")
    contact = extract_contact_info(resume_text)
    for field, present in contact.items():
        if present:
            score += 4 if field in ("Email","Phone") else 3
            reasons.append(f"{field} detected")
        elif field in ("Email","Phone"):
            issues.append(f"No {field} found — required for recruiters")
    wc = len(resume_text.split())
    if 300 <= wc <= 1200:
        score += 10; reasons.append(f"Good length ({wc} words)")
    elif wc < 300:
        issues.append(f"Too short ({wc} words) — expand your resume")
    else:
        issues.append(f"Possibly too long ({wc} words)")
    special = len(re.findall(r'[^\x00-\x7F]', resume_text))
    if special < 15:
        score += 5; reasons.append("Clean encoding, no special characters")
    else:
        issues.append(f"{special} special characters may cause ATS errors")
    bullets = len(re.findall(r'[•\-\*▪▸◦]', resume_text))
    if bullets >= 5:
        score += 5; reasons.append(f"Good use of bullet points ({bullets})")
    return min(score, 100), reasons, issues

def keyword_density(resume_text, jd_text):
    jd_words = re.findall(r'\b[a-zA-Z]{4,}\b', jd_text.lower())
    stop_words = {"that","this","with","from","have","will","your","their","they",
                  "about","been","were","into","more","also","each","must","should",
                  "would","could","role","team","work","help","strong","good","great",
                  "skills","using","able","based","well","experience","knowledge",
                  "ability","required","preferred","work","join","responsibilities"}
    freq = Counter(w for w in jd_words if w not in stop_words)
    top  = freq.most_common(18)
    rn   = normalize(resume_text)
    return [{"keyword": w, "jd_freq": f,
             "in_resume": bool(re.search(r'\b' + re.escape(w) + r'\b', rn))}
            for w, f in top]

def detect_experience_level(text):
    tl = text.lower()
    matches = re.findall(r'(\d+)\+?\s*years?\s*(of\s*)?(experience|work)', tl)
    if matches:
        yrs = max(int(m[0]) for m in matches)
        level = "Senior (5+ yrs)" if yrs >= 5 else "Mid-level (2–5 yrs)" if yrs >= 2 else "Junior (<2 yrs)"
        return level, yrs
    for sig in ["senior","lead","principal","head of","director","architect"]:
        if sig in tl: return "Senior", "N/A"
    for sig in ["intern","trainee","fresher","entry level","graduate","student"]:
        if sig in tl: return "Entry / Intern", "N/A"
    return "Mid-level", "N/A"

IMPACT_VERBS = [
    "developed","built","designed","implemented","deployed","led","managed",
    "improved","increased","reduced","achieved","created","automated","optimized",
    "analyzed","predicted","trained","evaluated","integrated","architected",
    "engineered","launched","delivered","collaborated","mentored","researched",
    "published","presented","won","awarded","generated","extracted","visualized",
    "cleaned","processed","modeled","forecasted","classified","clustered"
]

def action_verb_score(resume_text):
    tl = resume_text.lower()
    found = [v for v in IMPACT_VERBS if re.search(r'\b' + v + r'\b', tl)]
    return min(len(found) * 8, 100), found

def generate_suggestions(missing_skills, ats_issues, sections, kw_data,
                          tfidf, skill_sc, quant_count, verb_count, ds_hits, bonus_skills):
    sugg = []
    if missing_skills:
        sugg.append({"priority":"High","icon":"🔴","title":"Add Missing JD Skills",
                     "detail":f"These skills appear in the JD but not in your resume: {', '.join(missing_skills[:6])}. Add them naturally in your Skills section or project descriptions."})
    for issue in ats_issues[:3]:
        sugg.append({"priority":"High","icon":"🔴","title":"Fix ATS Issue","detail":issue})
    if quant_count < 3:
        sugg.append({"priority":"High","icon":"🔴","title":"Add Metrics & Numbers",
                     "detail":"Quantify your achievements: '92% accuracy', 'reduced latency by 40%', 'processed 1M+ rows'. Aim for at least 5–8 quantified bullets."})
    if tfidf < 25:
        sugg.append({"priority":"High","icon":"🔴","title":"Mirror the JD Language",
                     "detail":"Your resume language differs significantly from the JD. Use exact phrases from the job description in your Summary and Skills section."})
    for s in ["Summary","Projects","Certifications"]:
        if not sections.get(s):
            detail = {
                "Summary": "A 3–4 line Summary mirroring the JD role title improves ATS ranking and recruiter first impressions.",
                "Projects": "For DS/ML roles, Projects are as critical as Experience. Include 2–3 end-to-end projects with GitHub links and measurable outcomes.",
                "Certifications": "Add credentials from Coursera, Kaggle, deeplearning.ai, or AWS to strengthen your profile."
            }.get(s, f"A '{s}' section improves ATS parsing and recruiter readability.")
            sugg.append({"priority":"Medium","icon":"🟡","title":f"Add {s} Section","detail":detail})
    missing_kw = [k["keyword"] for k in kw_data if not k["in_resume"]][:5]
    if missing_kw:
        sugg.append({"priority":"Medium","icon":"🟡","title":"Improve Keyword Alignment",
                     "detail":f"Naturally incorporate these high-frequency JD terms: {', '.join(missing_kw)}. Place them in your Summary or project descriptions."})
    if verb_count < 6:
        sugg.append({"priority":"Medium","icon":"🟡","title":"Use More Action Verbs",
                     "detail":"Start bullet points with strong verbs: Developed, Deployed, Trained, Evaluated, Optimized, Visualized, Forecasted, Architected."})
    if ds_hits >= 5:
        sugg.append({"priority":"Low","icon":"🟢","title":"Strong Project Evidence Detected",
                     "detail":f"We found {ds_hits} data-science project signals in your resume. Make sure your GitHub profile link is visible at the top."})
    if bonus_skills:
        sugg.append({"priority":"Low","icon":"🟢","title":"Additional Skills Detected",
                     "detail":f"Your resume has skills beyond the JD requirements: {', '.join(bonus_skills[:8])}. Highlight these in an 'Additional Skills' section."})
    if skill_sc >= 70 and quant_count >= 5:
        sugg.append({"priority":"Low","icon":"🟢","title":"Excellent Foundation",
                     "detail":"Strong skill alignment and quantified results. Polish your Summary so the first sentence mirrors the JD role title and seniority."})
    if not sugg:
        sugg.append({"priority":"Low","icon":"🟢","title":"Great Match!",
                     "detail":"Your resume is well-optimized for this role. Ensure your cover letter echoes the JD tone and highlights your top 2–3 projects."})
    return sugg

def compute_final_score(tfidf, skill_sc, ats_sc, quant_sc, ds_bonus):
    base  = (0.30 * tfidf) + (0.38 * skill_sc) + (0.20 * ats_sc) + (0.12 * quant_sc)
    bonus = ds_bonus * 0.4
    return round(min(base + bonus, 100), 1)

# ══════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════
def score_color(s):
    if s >= 75: return "#00e5a0"
    if s >= 50: return "#f5a623"
    return "#ff4d6d"

def score_verdict(s):
    if s >= 82: return ("Excellent Match", "rgba(0,229,160,0.15)", "#00e5a0")
    if s >= 68: return ("Good Match",      "rgba(0,229,160,0.1)",  "#00e5a0")
    if s >= 52: return ("Moderate Match",  "rgba(245,166,35,0.15)","#f5a623")
    if s >= 38: return ("Needs Work",      "rgba(255,77,109,0.12)","#ff4d6d")
    return             ("Low Match",       "rgba(255,77,109,0.15)","#ff4d6d")

def ring_svg(score, color, size=148):
    r = 56; cx = cy = size // 2
    circ = 2 * math.pi * r
    dash = (score / 100) * circ
    uid  = int(score * 10)
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <defs>
        <linearGradient id="rg{uid}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color:{color};stop-opacity:0.45"/>
          <stop offset="100%" style="stop-color:{color};stop-opacity:1"/>
        </linearGradient>
        <filter id="gl{uid}" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="4" result="blur"/>
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
              stroke="rgba(255,255,255,0.05)" stroke-width="11"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
              stroke="url(#rg{uid})" stroke-width="11"
              stroke-linecap="round" filter="url(#gl{uid})"
              stroke-dasharray="{dash:.1f} {circ:.1f}"
              transform="rotate(-90 {cx} {cy})"/>
      <text x="{cx}" y="{cy - 6}" text-anchor="middle" dominant-baseline="central"
            font-family="Syne,sans-serif" font-size="24" font-weight="800" fill="{color}">{score}%</text>
      <text x="{cx}" y="{cy + 16}" text-anchor="middle"
            font-family="Inter,sans-serif" font-size="9" font-weight="500"
            fill="rgba(255,255,255,0.35)" letter-spacing="1.5">MATCH</text>
    </svg>"""

def mini_ring(score, color, size=56):
    r = 22; cx = cy = size // 2
    circ = 2 * math.pi * r
    dash = (score / 100) * circ
    uid  = f"m{int(score*7)}"
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
              stroke="rgba(255,255,255,0.06)" stroke-width="5"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none"
              stroke="{color}" stroke-width="5" stroke-linecap="round"
              stroke-dasharray="{dash:.1f} {circ:.1f}"
              transform="rotate(-90 {cx} {cy})"/>
      <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central"
            font-family="Syne,sans-serif" font-size="10" font-weight="800" fill="{color}">{int(score)}</text>
    </svg>"""

# ══════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
    --bg:        #07090f;
    --surface:   #0d1018;
    --surface2:  #121520;
    --glass:     rgba(255,255,255,0.035);
    --glass2:    rgba(255,255,255,0.065);
    --border:    rgba(255,255,255,0.07);
    --border2:   rgba(255,255,255,0.12);
    --accent:    #4f8fff;
    --accent2:   #a855f7;
    --green:     #00e5a0;
    --amber:     #f5a623;
    --red:       #ff4d6d;
    --text:      #dde4f0;
    --text2:     #a0aabf;
    --muted:     #525d7a;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
    color: var(--text) !important;
}
.stApp { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 3rem 5rem !important; max-width: 1380px !important; }

/* ── GLASSMORPHISM ── */
.input-panel, .metric-card, .card, .cat-card, .stTabs [data-baseweb="tab-list"], .sugg {
    background: rgba(13, 16, 24, 0.45) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* ── BACKGROUND ── */
.bg-orbs {
    position: fixed; inset: 0; pointer-events: none; z-index: -1; overflow: hidden;
    background: #07090f;
}
.bg-orbs::before, .bg-orbs::after, .bg-orbs-3 {
    content: '';
    position: absolute; border-radius: 50%; opacity: 0.35;
    filter: blur(100px);
    animation: floating 14s infinite alternate ease-in-out;
}
.bg-orbs::before {
    width: 700px; height: 700px;
    background: #4f8fff;
    top: -150px; left: -150px;
}
.bg-orbs::after {
    width: 700px; height: 700px;
    background: #a855f7;
    bottom: -150px; right: -150px;
    animation-delay: -6s;
}
.bg-orbs-3 {
    width: 600px; height: 600px;
    background: #00e5a0;
    top: 40%; left: 40%;
    opacity: 0.2;
    animation: floating-center 18s infinite alternate ease-in-out;
}
@keyframes floating {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(150px, -100px) scale(1.4); }
}
@keyframes floating-center {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(-100px, 150px) scale(1.3); }
}

/* ── HERO ── */
.hero-wrap {
    padding: 4rem 1rem 1.5rem;
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    width: 100%;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(79,143,255,0.08);
    border: 1px solid rgba(79,143,255,0.2);
    color: #7aabff;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase;
    padding: 5px 14px; border-radius: 99px; margin-bottom: 1.4rem;
}
.hero-tag span { width:5px; height:5px; border-radius:50%; background:#4f8fff;
                  box-shadow:0 0 8px #4f8fff; display:inline-block; }
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 3.6rem; font-weight: 800; line-height: 1.05;
    background: linear-gradient(140deg, #fff 0%, #c8d9ff 35%, #c4a0ff 65%, #00e5a0 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -2px; margin-bottom: 0.85rem;
}
.hero-desc {
    color: var(--text2); font-size: 0.97rem; line-height: 1.7;
    max-width: 650px; text-align: center;
    margin: 1rem 0 2.5rem 0;
}

/* ── INPUT PANEL ── */
.input-panel {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 24px;
    padding: 2rem 2.5rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.input-panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(79,143,255,0.4) 30%,
                rgba(168,85,247,0.4) 70%, transparent 100%);
}
.input-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--text2); margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 7px;
}
.input-label::before {
    content: ''; width: 5px; height: 5px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 8px var(--accent);
    display: inline-block;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--glass) !important;
    border: 1.5px dashed rgba(79,143,255,0.3) !important;
    border-radius: 14px !important;
    padding: 1.5rem 1rem !important;
    transition: all 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(79,143,255,0.7) !important;
    background: rgba(79,143,255,0.04) !important;
}
[data-testid="stFileUploader"] > div { min-height: 120px !important; }

/* ── TEXTAREA ── */
[data-testid="stTextArea"] textarea {
    background: var(--glass) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    line-height: 1.7 !important;
    padding: 1rem !important;
    resize: none !important;
    min-height: 168px !important;
    transition: border-color 0.25s, box-shadow 0.25s !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: var(--muted) !important; }
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(79,143,255,0.55) !important;
    box-shadow: 0 0 0 3px rgba(79,143,255,0.1) !important;
    outline: none !important;
}

/* Hide all default streamlit labels inside input panel */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"],
[data-testid="stTextArea"] label,
[data-testid="stTextArea"] [data-testid="stWidgetLabel"] {
    display: none !important;
}

/* ── ANALYZE BUTTON ── */
@keyframes pulse-btn {
    0% { box-shadow: 0 0 0 0 rgba(79,143,255,0.7); }
    70% { box-shadow: 0 0 0 12px rgba(79,143,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(79,143,255,0); }
}
.stButton > button {
    animation: pulse-btn 2.5s infinite;
    background: linear-gradient(135deg, #4f8fff 0%, #7c4dff 50%, #a855f7 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 14px !important; padding: 0.85rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 700 !important;
    letter-spacing: 0.3px !important; width: 100% !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(79,143,255,0.3) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── DIVIDER ── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* ── SCORE STRIP ── */
.score-strip {
    display: grid;
    grid-template-columns: 1.1fr 1fr 1fr 1fr 1fr;
    gap: 14px;
    margin-bottom: 1.75rem;
}
.score-main-card {
    background: linear-gradient(145deg, rgba(79,143,255,0.07), rgba(168,85,247,0.07));
    border: 1px solid rgba(79,143,255,0.18);
    border-radius: 20px; padding: 1.75rem 1rem;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; position: relative; overflow: hidden;
    gap: 6px;
}
.score-main-card::before {
    content: ''; position: absolute; top: 0; left: 20%; right: 20%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(79,143,255,0.6),
                rgba(168,85,247,0.6), transparent);
}
.score-main-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--muted);
}
.verdict-badge {
    display: inline-block; padding: 4px 14px;
    border-radius: 99px; font-size: 0.78rem; font-weight: 700;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px; padding: 1.4rem 1.25rem;
    display: flex; flex-direction: column; gap: 4px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--border2); }
.metric-card::after {
    content: ''; position: absolute;
    top: 0; right: 0; width: 60px; height: 60px;
    border-radius: 0 20px 0 60px;
    background: var(--card-accent-bg, rgba(79,143,255,0.04));
}
.mc-label {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: var(--muted); margin-bottom: 2px;
}
.mc-icon { font-size: 1rem; position: absolute; top: 1.1rem; right: 1.1rem; opacity: 0.5; }
.mc-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    line-height: 1; letter-spacing: -1px;
}
.mc-sub { font-size: 0.75rem; color: var(--text2); margin-top: 2px; }
.mc-bar {
    height: 3px; background: rgba(255,255,255,0.06);
    border-radius: 99px; margin-top: 10px; overflow: hidden;
}
.mc-bar-fill { height: 100%; border-radius: 99px; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 5px !important; gap: 3px !important;
    margin-bottom: 0.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
    padding: 9px 18px !important;
    transition: color 0.2s !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text2) !important; }
.stTabs [aria-selected="true"] {
    background: rgba(79,143,255,0.15) !important;
    color: #93c0ff !important;
    box-shadow: 0 0 0 1px rgba(79,143,255,0.25) inset !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.25rem !important; }

/* ── CARDS ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px; padding: 1.5rem;
    height: 100%;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.card:hover, .metric-card:hover, .cat-card:hover {
    transform: translateY(-6px) scale(1.015);
    box-shadow: 0 15px 35px -10px rgba(79,143,255,0.4);
    border-color: rgba(79,143,255,0.4) !important;
    z-index: 10;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: var(--text2); margin-bottom: 1.1rem;
    display: flex; align-items: center; gap: 8px;
}
.card-title i {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 8px var(--accent);
    display: inline-block; flex-shrink: 0;
}

/* ── SKILL PILLS ── */
.pill {
    display: inline-block; border-radius: 99px;
    font-size: 0.72rem; font-weight: 600;
    padding: 3px 10px; margin: 2.5px;
    letter-spacing: 0.2px;
}
.p-green  { background: rgba(0,229,160,0.09);  color: #00e5a0; border: 1px solid rgba(0,229,160,0.22); }
.p-red    { background: rgba(255,77,109,0.09);  color: #ff8099; border: 1px solid rgba(255,77,109,0.22); }
.p-blue   { background: rgba(79,143,255,0.09);  color: #93baff; border: 1px solid rgba(79,143,255,0.22); }
.p-purple { background: rgba(168,85,247,0.09);  color: #c084fc; border: 1px solid rgba(168,85,247,0.22); }
.p-gray   { background: rgba(255,255,255,0.04); color: var(--text2); border: 1px solid var(--border); }

/* ── CATEGORY SKILL CARD ── */
.cat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 1.2rem 1.3rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.cat-card:hover { border-color: var(--border2); }
.cat-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.cat-name { font-family:'Syne',sans-serif; font-weight:700; font-size:0.85rem; color:var(--text); }
.cat-score { font-family:'Syne',sans-serif; font-weight:800; font-size:1rem; }
.cat-bar { height:3px; background:rgba(255,255,255,0.06); border-radius:99px; margin-bottom:10px; overflow:hidden; }
.cat-fill { height:100%; border-radius:99px; }

/* ── KEYWORD TABLE ── */
.kw-grid { display:flex; flex-direction:column; gap:0; }
.kw-row {
    display:grid; grid-template-columns:1fr 2fr 70px;
    align-items:center; gap:1rem;
    padding:0.6rem 0;
    border-bottom:1px solid rgba(255,255,255,0.04);
}
.kw-word { font-size:0.85rem; font-weight:500; color:var(--text); }
.kw-status { font-size:0.75rem; font-weight:700; text-align:right; }

/* ── SUGGESTIONS ── */
.sugg {
    border-radius: 14px; padding: 1rem 1.25rem;
    margin-bottom: 0.7rem; border-left: 2.5px solid;
}
.s-high   { background:rgba(255,77,109,0.06);   border-color:var(--red); }
.s-medium { background:rgba(245,166,35,0.06);   border-color:var(--amber); }
.s-low    { background:rgba(0,229,160,0.06);    border-color:var(--green); }
.sugg-title { font-weight:700; font-size:0.88rem; margin-bottom:4px; color:var(--text); }
.sugg-body  { font-size:0.82rem; color:var(--text2); line-height:1.65; }

/* ── PROFILE TABLE ── */
.prow {
    display:flex; justify-content:space-between; align-items:center;
    padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.04);
    font-size:0.85rem;
}
.prow-label { color:var(--text2); }
.prow-val   { font-weight:600; color:var(--text); }

/* ── ATS SECTION TAGS ── */
.sec-tag {
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 12px; border-radius:10px; margin:4px;
    font-size:0.8rem; font-weight:600; border:1px solid;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--surface); }
::-webkit-scrollbar-thumb { background:rgba(79,143,255,0.25); border-radius:3px; }

/* ── MISC ── */
.stAlert { border-radius:14px !important; }
.stSpinner > div { border-top-color:var(--accent) !important; }
.section-title {
    font-family:'Syne',sans-serif; font-size:0.72rem; font-weight:700;
    letter-spacing:2px; text-transform:uppercase;
    padding:0 0 0.6rem; margin-bottom:0.8rem;
    border-bottom:1px solid var(--border);
}
</style>
<div class="bg-orbs">
    <div class="bg-orbs-3"></div>
</div>
"""

# ══════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════
st.markdown(CSS, unsafe_allow_html=True)

# ── HERO ──
st.markdown("""
<div class="hero-wrap">
    <div class="hero-tag"><span></span> AI Resume Intelligence</div>
    <div class="hero-title">HireSense AI</div>
    <p class="hero-desc">
        Upload your resume and paste a job description to receive a detailed match report —
        ATS scoring, skill gap analysis, keyword radar, and a prioritized action plan.
    </p>
</div>
""", unsafe_allow_html=True)

# ── INPUT PANEL ──
st.markdown('<div class="input-panel">', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="input-label">📄 &nbsp; Resume &nbsp;(PDF)</div>', unsafe_allow_html=True)
    resume_file = st.file_uploader("resume_upload", type=["pdf"], label_visibility="collapsed")

with right_col:
    st.markdown('<div class="input-label">📋 &nbsp; Job Description</div>', unsafe_allow_html=True)
    job_desc = st.text_area(
        "jd_input",
        placeholder="Paste the full job description here…",
        height=168,
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

btn_l, btn_c, btn_r = st.columns([1.5, 2, 1.5])
with btn_c:
    go = st.button("Analyze Resume", width="stretch")

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════
if go:
    st.markdown('<audio src="https://assets.mixkit.co/active_storage/sfx/2568/2568-preview.mp3" autoplay></audio>', unsafe_allow_html=True)
    
    if not resume_file:
        st.error("Please upload a PDF resume.")
    elif not job_desc.strip():
        st.error("Please paste a job description.")
    else:
        lottie_placeholder = st.empty()
        with lottie_placeholder.container():
            st.markdown("<h3 style='text-align: center; color: #4f8fff; font-family: Syne; padding-top:2rem'>Scanning Matrices...</h3>", unsafe_allow_html=True)
            try:
                r = requests.get("https://lottie.host/7e008b62-38d5-4dc2-b131-7e8c07e008ba/nI6V1FjZ6a.json")
                if r.status_code == 200: st_lottie(r.json(), height=250)
            except: pass

        with st.spinner("Compiling Intelligence Report…"):
            st.toast("Scanning Source Repositories...", icon="🔍")
            time.sleep(0.4)
            st.toast("Cross-referencing Matrix Skills...", icon="🧠")
            time.sleep(0.4)
            st.toast("Verifying ATS Parsing Engines...", icon="⚙️")
            time.sleep(0.4)
            
            resume_text  = extract_text(resume_file)
            word_count   = len(resume_text.split())

            tfidf_sc                                       = tfidf_score(resume_text, job_desc)
            skill_sc, cat_results, matched, missing, bonus = skill_analysis(resume_text, job_desc)
            sections                                       = detect_sections(resume_text)
            ats_sc, ats_ok, ats_bad                        = ats_score(resume_text, sections)
            kw_data                                        = keyword_density(resume_text, job_desc)
            exp_level, exp_yrs                             = detect_experience_level(resume_text)
            contact                                        = extract_contact_info(resume_text)
            quant_sc, quant_count                          = quantification_score(resume_text)
            verb_sc,  found_verbs                          = action_verb_score(resume_text)
            ds_bonus, ds_hits                              = ds_project_bonus(resume_text)
            suggestions = generate_suggestions(
                missing, ats_bad, sections, kw_data,
                tfidf_sc, skill_sc, quant_count, len(found_verbs), ds_hits, bonus
            )
            final = compute_final_score(tfidf_sc, skill_sc, ats_sc, quant_sc, ds_bonus)
            verdict, v_bg, v_color = score_verdict(final)
            
            if final >= 80:
                st.balloons()
            
        lottie_placeholder.empty()

        # ── SCORE STRIP ──
        c0, c1, c2, c3, c4 = st.columns([1.1, 1, 1, 1, 1], gap="small")

        with c0:
            gauge_fig = pgo.Figure(pgo.Indicator(
                mode="gauge+number",
                value=final,
                title={'text': "MATCH",'font':{'size':10,'color':"rgba(255,255,255,0.4)"}},
                number={'suffix': "%", 'font':{'size':38, 'color': v_color, 'family': 'Syne'}},
                gauge={
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': v_color, 'thickness': 0.15},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'steps': [{'range': [0, final], 'color': "rgba(255,255,255,0.05)"}],
                }
            ))
            gauge_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=140, paper_bgcolor="rgba(0,0,0,0)")
            
            st.markdown(f'<div class="score-main-card" style="padding:0">', unsafe_allow_html=True)
            st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown(f'<div class="verdict-badge" style="background:{v_bg};color:{v_color};margin-bottom:12px;">{verdict}</div></div>', unsafe_allow_html=True)

        cards = [
            ("Content Fit",   tfidf_sc,        "TF-IDF similarity",          "📝", "rgba(79,143,255,0.05)"),
            ("Skill Match",   round(skill_sc),  f"{len(matched)}/{len(matched)+len(missing)} matched", "🛠", "rgba(0,229,160,0.05)"),
            ("ATS Score",     ats_sc,           "Applicant tracking",         "🤖", "rgba(168,85,247,0.05)"),
            ("Impact Score",  quant_sc,         f"{quant_count} metrics found","📊", "rgba(245,166,35,0.05)"),
        ]
        for col, (label, val, sub, icon, bg) in zip([c1, c2, c3, c4], cards):
            c   = score_color(val)
            pct = min(float(val), 100)
            with col:
                st.markdown(f"""
                <div class="metric-card" style="--card-accent-bg:{bg}">
                    <div class="mc-label">{label}</div>
                    <span class="mc-icon">{icon}</span>
                    <div class="mc-value" style="color:{c}">{val}%</div>
                    <div class="mc-sub">{sub}</div>
                    <div class="mc-bar">
                        <div class="mc-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{c}55,{c})"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── TABS ──
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "  Skills Analysis  ",
            "  Keyword Radar  ",
            "  ATS Report  ",
            "  Action Plan  ",
            "  Resume Profile  ",
            "  📄 PDF Preview  ",
            "  🤖 AI Recruiter  ",
        ])

        # ── TAB 1: SKILLS ──
        with tab1:
            if cat_results:
                st.markdown('<div class="card" style="margin-bottom:1.5rem">', unsafe_allow_html=True)
                st.markdown('<div class="card-title"><i></i> Category Competency Radar</div>', unsafe_allow_html=True)
                categories = list(cat_results.keys())
                scores = [d["score"] for d in cat_results.values()]
                categories.append(categories[0])
                scores.append(scores[0])
                fig = pgo.Figure()
                fig.add_trace(pgo.Scatterpolar(
                    r=scores, theta=categories, fill='toself', name='Score',
                    line_color='#00e5a0', fillcolor='rgba(0, 229, 160, 0.2)'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], color='rgba(255,255,255,0.3)', gridcolor='rgba(255,255,255,0.1)'),
                        angularaxis=dict(color='rgba(255,255,255,0.7)', gridcolor='rgba(255,255,255,0.1)')
                    ),
                    showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                col_a, col_b = st.columns(2, gap="large")
                items = list(cat_results.items())
                half  = math.ceil(len(items) / 2)
                for col, chunk in [(col_a, items[:half]), (col_b, items[half:])]:
                    with col:
                        for cat, data in chunk:
                            c  = score_color(data["score"])
                            mp = "".join(f'<span class="pill p-green">&#10003; {s}</span>' for s in data["matched"])
                            xp = "".join(f'<span class="pill p-red">&#10007; {s}</span>'  for s in data["missing"])
                            st.markdown(f"""
                            <div class="cat-card">
                                <div class="cat-head">
                                    <span class="cat-name">{cat}</span>
                                    <span class="cat-score" style="color:{c}">{data['score']}%</span>
                                </div>
                                <div class="cat-bar">
                                    <div class="cat-fill" style="width:{data['score']}%;
                                         background:linear-gradient(90deg,{c}55,{c})"></div>
                                </div>
                                <div>{mp}{xp}</div>
                            </div>""", unsafe_allow_html=True)
            else:
                st.info("No skills from the taxonomy were found in the job description. Try a more detailed JD.")

            if bonus:
                bonus_pills = "".join(f'<span class="pill p-purple">+ {s}</span>' for s in bonus[:20])
                st.markdown(f"""
                <div class="card" style="margin-top:1rem">
                    <div class="card-title"><i></i> Bonus Skills on Resume (beyond JD requirements)</div>
                    <div>{bonus_pills}</div>
                </div>""", unsafe_allow_html=True)

        # ── TAB 2: KEYWORDS ──
        with tab2:
            covered     = sum(1 for k in kw_data if k["in_resume"])
            not_covered = len(kw_data) - covered
            max_freq    = max((k["jd_freq"] for k in kw_data), default=1)

            st.markdown(f"""
            <div class="card">
                <div class="card-title"><i></i> Job Description Keyword Coverage</div>
                <div style="display:flex;gap:2.5rem;margin-bottom:1.25rem;padding-bottom:1rem;
                            border-bottom:1px solid var(--border)">
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:2rem;
                                    font-weight:800;color:#00e5a0;line-height:1">{covered}</div>
                        <div style="font-size:0.75rem;color:var(--text2);margin-top:3px">Keywords Covered</div>
                    </div>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:2rem;
                                    font-weight:800;color:#ff4d6d;line-height:1">{not_covered}</div>
                        <div style="font-size:0.75rem;color:var(--text2);margin-top:3px">Keywords Missing</div>
                    </div>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-size:2rem;
                                    font-weight:800;color:var(--accent);line-height:1">{round((covered/max(len(kw_data),1))*100)}%</div>
                        <div style="font-size:0.75rem;color:var(--text2);margin-top:3px">Coverage Rate</div>
                    </div>
                </div>
                <div class="kw-grid">
            """, unsafe_allow_html=True)

            for kw in kw_data:
                pct    = round((kw["jd_freq"] / max_freq) * 100)
                bc     = "#00e5a0" if kw["in_resume"] else "#ff4d6d"
                status = f'<span style="color:{bc}">' + ("&#10003; Found" if kw["in_resume"] else "&#10007; Missing") + '</span>'
                st.markdown(f"""
                <div class="kw-row">
                    <div class="kw-word">{kw['keyword']}</div>
                    <div>
                        <div style="background:rgba(255,255,255,0.05);border-radius:99px;height:4px">
                            <div style="width:{pct}%;height:100%;background:{bc}40;border-radius:99px"></div>
                        </div>
                        <span style="font-size:0.7rem;color:var(--muted);margin-top:2px;display:inline-block">
                            {kw['jd_freq']}x in JD</span>
                    </div>
                    <div class="kw-status">{status}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('</div></div>', unsafe_allow_html=True)

        # ── TAB 3: ATS ──
        with tab3:
            a1, a2 = st.columns(2, gap="large")

            with a1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title"><i></i> Passing Checks</div>', unsafe_allow_html=True)
                for r in ats_ok:
                    st.markdown(f'<div class="prow"><span style="color:#00e5a0">&#10003;</span><span style="flex:1;padding-left:8px;font-size:0.84rem;color:var(--text)">{r}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with a2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title"><i style="background:var(--red);box-shadow:0 0 8px var(--red)"></i> Issues Found</div>', unsafe_allow_html=True)
                if ats_bad:
                    for issue in ats_bad:
                        st.markdown(f'<div class="prow"><span style="color:#ff4d6d">&#10007;</span><span style="flex:1;padding-left:8px;font-size:0.84rem;color:var(--text)">{issue}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:#00e5a0;font-size:0.84rem;padding:0.5rem 0">No ATS issues detected. Great job!</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i></i> Resume Sections Detected</div>', unsafe_allow_html=True)
            cols_s = st.columns(4)
            for i, (sec, present) in enumerate(sections.items()):
                color = "#00e5a0" if present else "#ff4d6d"
                bg    = "rgba(0,229,160,0.06)" if present else "rgba(255,77,109,0.06)"
                mark  = "&#10003;" if present else "&#10007;"
                with cols_s[i % 4]:
                    st.markdown(f'<div class="sec-tag" style="background:{bg};color:{color};border-color:{color}33">{mark} {sec}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── TAB 4: ACTION PLAN ──
        with tab4:
            high   = [s for s in suggestions if s["priority"] == "High"]
            medium = [s for s in suggestions if s["priority"] == "Medium"]
            low    = [s for s in suggestions if s["priority"] == "Low"]

            if high:
                st.markdown('<div class="section-title" style="color:#ff4d6d">High Priority</div>', unsafe_allow_html=True)
                for s in high:
                    st.markdown(f'<div class="sugg s-high"><div class="sugg-title">{s["icon"]} &nbsp;{s["title"]}</div><div class="sugg-body">{s["detail"]}</div></div>', unsafe_allow_html=True)
            if medium:
                st.markdown('<div class="section-title" style="color:#f5a623;margin-top:1.5rem">Medium Priority</div>', unsafe_allow_html=True)
                for s in medium:
                    st.markdown(f'<div class="sugg s-medium"><div class="sugg-title">{s["icon"]} &nbsp;{s["title"]}</div><div class="sugg-body">{s["detail"]}</div></div>', unsafe_allow_html=True)
            if low:
                st.markdown('<div class="section-title" style="color:#00e5a0;margin-top:1.5rem">Strengths</div>', unsafe_allow_html=True)
                for s in low:
                    st.markdown(f'<div class="sugg s-low"><div class="sugg-title">{s["icon"]} &nbsp;{s["title"]}</div><div class="sugg-body">{s["detail"]}</div></div>', unsafe_allow_html=True)

        # ── TAB 5: PROFILE ──
        with tab5:
            p1, p2, p3 = st.columns(3, gap="large")

            def prow(label, val):
                return f'<div class="prow"><span class="prow-label">{label}</span><span class="prow-val">{val}</span></div>'

            with p1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title"><i></i> Candidate Profile</div>', unsafe_allow_html=True)
                st.markdown(
                    prow("Experience Level", exp_level) +
                    prow("Years of Exp.", str(exp_yrs)) +
                    prow("Word Count", f"{word_count} words") +
                    prow("Project Signals", f"{ds_hits} detected") +
                    prow("File", resume_file.name),
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with p2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title"><i></i> Contact Information</div>', unsafe_allow_html=True)
                for field, present in contact.items():
                    color  = "#00e5a0" if present else "#ff4d6d"
                    status = "Found" if present else "Missing"
                    mark   = "&#10003;" if present else "&#10007;"
                    st.markdown(f'<div class="prow"><span class="prow-label">{field}</span><span style="color:{color};font-weight:600">{mark} {status}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with p3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title"><i></i> Writing Quality</div>', unsafe_allow_html=True)
                st.markdown(
                    prow("Action Verbs", f"{len(found_verbs)} found") +
                    prow("Quantified Bullets", f"{quant_count} metrics") +
                    prow("Verb Score", f"{verb_sc}%") +
                    prow("Impact Score", f"{quant_sc}%"),
                    unsafe_allow_html=True
                )
                if found_verbs:
                    vp = "".join(f'<span class="pill p-blue">{v}</span>' for v in found_verbs[:12])
                    st.markdown(f'<div style="margin-top:0.8rem">{vp}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ── TAB 6: PDF PREVIEW ──
        with tab6:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i></i> Document Analyzer Render</div>', unsafe_allow_html=True)
            base64_pdf = base64.b64encode(resume_file.getvalue()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="650" type="application/pdf" style="border-radius:12px; border: 1px solid rgba(255,255,255,0.1)"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── TAB 7: AI RECRUITER ──
        with tab7:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title"><i style="background:#a855f7;box-shadow:0 0 8px #a855f7"></i> Deep AI Feedback</div>', unsafe_allow_html=True)
            if st.session_state.get('gemini_key'):
                with st.spinner("AI Recruiter is reading the resume (Gemini 1.5)..."):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=st.session_state.gemini_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        prompt = f"You are a strict, top-tier tech recruiter. Analyze the resume against the job description. Give 3 brutal bullet points on why they might be rejected, 3 strong points on why they might be hired, and 1 final actionable trick to land the interview. Keep it concise, punchy, and formatted with markdown. Resume: {resume_text[:4000]} | Job Desc: {job_desc[:2000]}"
                        response = model.generate_content(prompt)
                        
                        def stream_matrix():
                            for word in response.text.split(" "):
                                yield word + " "
                                time.sleep(0.06)
                                
                        st.write_stream(stream_matrix)
                    except Exception as e:
                        st.error(f"Error connecting to AI Recruiter: {e}")
            else:
                st.info("Unlock AI-powered qualitative feedback! Enter your Google Gemini API Key in the sidebar.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── FOOTER ──
        st.markdown(f"""
        <div style="text-align:center;color:var(--muted);font-size:0.72rem;
                    margin-top:3rem;padding-top:1.5rem;border-top:1px solid var(--border);
                    letter-spacing:0.5px; margin-bottom: 2.5rem;">
            HireSense AI &nbsp;&middot;&nbsp;
            {datetime.now().strftime("%d %b %Y, %H:%M")} &nbsp;&middot;&nbsp;
            {len(matched) + len(missing)} JD skills scanned &nbsp;&middot;&nbsp;
            {len(ALL_SKILLS_FLAT)} skills in taxonomy
        </div>""", unsafe_allow_html=True)
        
        # ── DOWNLOAD REPORT ──
        col_dl_l, col_dl_c, col_dl_r = st.columns([1,1.5,1])
        with col_dl_c:
            report_md = f"# HireSense ATS Intelligence Report\\n\\n**Target Candidate Output:** System Parsed\\n**Final Match Score:** {final}%\\n\\n### Required Actions\\n"
            for s in suggestions: report_md += f"- **{s['title']}**: {s['detail']}\\n"
            st.download_button("📥 Download Analysis Report", data=report_md, file_name="HireSense_Report.md", mime="text/markdown", width="stretch")
