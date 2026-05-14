import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd
import json
import os
from datetime import datetime

# Download NLTK resources
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# Persistent user database file
USER_DB_FILE = "users_db.json"

def load_users_db():
    """Load user database from JSON file."""
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, 'r') as f:
                return json.load(f)
        except:
            return {
                'admin': 'Resume@123',
                'resumeuser': 'securepass123'
            }
    return {
        'admin': 'Resume@123',
        'resumeuser': 'securepass123'
    }

def save_users_db(user_db):
    """Save user database to JSON file."""
    try:
        with open(USER_DB_FILE, 'w') as f:
            json.dump(user_db, f)
    except Exception as e:
        print(f"Error saving user database: {e}")

def signup_with_gmail(gmail_address):
    """Register or login user with Gmail address."""
    if not gmail_address or '@gmail.com' not in gmail_address.lower():
        return False, 'Please enter a valid Gmail address.'
    
    gmail_username = gmail_address.split('@')[0]
    
    if gmail_username in st.session_state.user_db:
        # User already exists, auto-login
        return True, f'Welcome back, {gmail_username}!'
    
    # Create new account with Gmail
    st.session_state.user_db[gmail_username] = f"gmail_{gmail_address}"
    save_users_db(st.session_state.user_db)
    return True, f'Account created with Gmail: {gmail_address}'

# Page Setup
st.set_page_config(page_title="Resume Job Match Scorer", page_icon="📄", layout="wide")

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
if 'job_description' not in st.session_state:
    st.session_state.job_description = None
if 'overall_score' not in st.session_state:
    st.session_state.overall_score = None
if 'section_scores' not in st.session_state:
    st.session_state.section_scores = None
if 'job_keywords' not in st.session_state:
    st.session_state.job_keywords = None
if 'resume_keywords' not in st.session_state:
    st.session_state.resume_keywords = None
if 'section_improvements' not in st.session_state:
    st.session_state.section_improvements = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'login_status' not in st.session_state:
    st.session_state.login_status = ''
if 'user_db' not in st.session_state:
    st.session_state.user_db = load_users_db()
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

st.markdown("""
# 📄 ResumeIQ : AI-Driven Resume Analysis and Skill Gap Detection using Natural Language Processing
Upload your resume (PDF) and paste a job description to see how well they match!  
This tool uses **TF-IDF + Cosine Similarity** to analyze your resume against job requirements.
**Now with INTERVIEW QUESTIONS GENERATOR!** ✨
""")

with st.sidebar:
    st.header("📋 About")
    st.info("""
    This tool helps you:
    - Measure how your resume matches a job description
    - **Identify missing keywords & skills by SECTION**
    - **Get specific improvement suggestions per section**
    - **Section-wise match scores** (Summary, Experience, Skills, Education)
    - **✨ Generate Interview Questions** tailored to the job!
    - Improve your resume based on job requirements
    """)
    st.header("🔧 How It Works")
    st.write("""
    1. Upload your resume (PDF)
    2. Paste the job description
    3. Click **Analyze Match**
    4. Review **scores & recommendations**
    5. **Generate Interview Questions** for preparation!
    """)

# Helper functions
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])

def extract_section_scores(resume_text, job_description):
    """Extract and score different resume sections"""
    resume_lower = resume_text.lower()
    job_processed = remove_stopwords(clean_text(job_description))
    
    sections = {
        'Summary/Objective': ['summary', 'objective', 'profile', 'about'],
        'Experience': ['experience', 'work experience', 'professional experience', 'employment'],
        'Skills': ['skills', 'technical skills', 'core competencies'],
        'Education': ['education', 'academic', 'degree', 'university'],
        'Projects': ['projects', 'project experience']
    }
    
    section_scores = {}
    all_section_text = ""
    
    for section_name, keywords in sections.items():
        # Find section text using keywords
        section_text = ""
        for keyword in keywords:
            pattern = rf'{keyword}.*?(?=\n\n|\Z)'
            matches = re.findall(pattern, resume_lower, re.IGNORECASE | re.DOTALL)
            section_text += " ".join(matches)
        
        if section_text:
            section_processed = remove_stopwords(clean_text(section_text))
            all_section_text += section_processed + " "
            
            # Calculate similarity for this section
            if len(section_processed) > 10 and len(job_processed) > 10:
                vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([section_processed, job_processed])
                score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
                section_scores[section_name] = round(score, 1)
            else:
                section_scores[section_name] = 0.0
        else:
            section_scores[section_name] = 0.0
    
    # Overall resume score
    overall_processed = remove_stopwords(clean_text(resume_text))
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([overall_processed, job_processed])
    overall_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    
    return overall_score, section_scores, sections

def get_section_improvements(job_keywords, section_scores):
    """Generate section-specific improvement suggestions"""
    suggestions = {}
    
    for section, score in section_scores.items():
        job_key_set = set([kw[0] for kw in job_keywords])
        
        if score < 30:
            suggestions[section] = {
                'status': '🔴 Critical',
                'score': score,
                'actions': [
                    f"**Completely rewrite** this section",
                    "Add **3-5 top job keywords**",
                    "Use **exact phrases** from job description"
                ]
            }
        elif score < 60:
            suggestions[section] = {
                'status': '🟡 Needs Work',
                'score': score,
                'actions': [
                    "Add **2-3 missing keywords**",
                    "**Expand** with relevant examples",
                    "Match **job responsibility phrasing**"
                ]
            }
        elif score < 80:
            suggestions[section] = {
                'status': '🟢 Good',
                'score': score,
                'actions': [
                    "Minor tweaks recommended",
                    "Add **1 niche skill** for edge",
                    "**Quantify** achievements"
                ]
            }
        else:
            suggestions[section] = {
                'status': '✅ Excellent',
                'score': score,
                'actions': ["Keep as is - perfect match!"]
            }
    
    return suggestions


def check_credentials(username, password):
    """Validate login credentials for full report access."""
    return st.session_state.user_db.get(username) == password


def add_new_user(username, password):
    """Register a new user account for full report access."""
    if not username or not password:
        return False, 'Username and password are required.'
    if username in st.session_state.user_db:
        return False, 'That username already exists. Please choose another one.'
    if len(password) < 6:
        return False, 'Password should be at least 6 characters long.'
    st.session_state.user_db[username] = password
    save_users_db(st.session_state.user_db)
    return True, 'Account created successfully. Please login with your new credentials.'


def extract_keywords(text, num_keywords=15):
    """Extract important keywords (nouns and adjectives) from text"""
    words = word_tokenize(text)
    words = [w for w in words if len(w) > 2]
    tagged_words = pos_tag(words)
    keywords = [w for w, pos in tagged_words 
                if pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB')]
    word_freq = Counter(keywords)
    return word_freq.most_common(num_keywords)


def extract_skills(text):
    """Extract a normalized set of skills from text using a skill vocabulary."""
    text_lower = text.lower()
    skill_vocab = [
        'python', 'pandas', 'numpy', 'scikit-learn', 'sklearn', 'tensorflow', 'keras',
        'sql', 'sql server', 'postgresql', 'mysql', 'mongodb', 'power bi', 'tableau',
        'excel', 'r', 'matplotlib', 'seaborn', 'opencv', 'computer vision', 'nlp',
        'natural language processing', 'javascript', 'react', 'vue', 'angular',
        'html', 'css', 'django', 'flask', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
        'git', 'api', 'rest api', 'node.js', 'fastapi', 'spark', 'hadoop', 'jira',
        'agile', 'scrum', 'machine learning', 'deep learning', 'data analysis',
        'data visualization', 'business intelligence', 'project management'
    ]
    found = set()
    for skill in skill_vocab:
        if skill in text_lower:
            found.add(skill)
    return found


def compute_ats_score(resume_text, job_description, section_scores, resume_skills, job_skills):
    """Estimate an ATS compatibility score and provide targeted suggestions."""
    resume_lower = resume_text.lower()
    job_lower = job_description.lower()

    headings = ['summary', 'experience', 'skills', 'education', 'projects']
    heading_hits = sum(1 for heading in headings if heading in resume_lower)
    heading_score = (heading_hits / len(headings)) * 25

    bullet_count = resume_text.count('- ') + resume_text.count('* ') + resume_text.count('• ')
    formatting_score = min(bullet_count, 12) / 12 * 25

    keyword_matches = len(resume_skills & job_skills)
    keyword_score = min(keyword_matches, 10) / 10 * 25

    section_match_score = sum(section_scores.values()) / (len(section_scores) * 100) * 25

    score = round(heading_score + formatting_score + keyword_score + section_match_score, 1)

    suggestions = []
    if heading_score < 15:
        suggestions.append('Add clear ATS-friendly headings such as Summary, Experience, Skills, Education, and Projects.')
    if formatting_score < 10:
        suggestions.append('Use simple formatting with bullet lists, consistent fonts, and no tables or images.')
    if keyword_score < 10:
        suggestions.append('Include more exact job keywords and tools from the job description in your resume.')
    if section_match_score < 15:
        suggestions.append('Improve section-level relevance by aligning each section with the job requirements.')
    if 'python' in job_lower and 'python' not in resume_lower:
        suggestions.append('Mention Python explicitly if the job description calls for it.')
    if not suggestions:
        suggestions.append('Your resume looks ATS-friendly. Keep section headings clear and keywords aligned.')

    return score, suggestions


def generate_resume_improvement_suggestions(resume_text, job_description, section_scores, job_keywords, resume_keywords):
    """Create actionable resume improvement suggestions."""
    resume_lower = resume_text.lower()
    resume_words = set(word_tokenize(remove_stopwords(clean_text(resume_text))))
    job_word_set = set(word_tokenize(remove_stopwords(clean_text(job_description))))

    missing_keywords = [kw for kw, _ in job_keywords if kw not in resume_words]
    missing_keywords = missing_keywords[:10]

    action_verbs = {'led', 'managed', 'developed', 'designed', 'built', 'implemented', 'created', 'optimized', 'improved', 'analyzed', 'automated', 'launched', 'collaborated', 'delivered'}
    resume_tokens = set(word_tokenize(clean_text(resume_text)))
    action_count = len(action_verbs & resume_tokens)

    suggestions = []
    if missing_keywords:
        suggestions.append(f"Add missing job keywords such as: {', '.join(missing_keywords)}.")
    if action_count < 5:
        suggestions.append('Use stronger action verbs like developed, led, optimized, automated, and launched.')
    if section_scores.get('Projects', 0) < 60:
        suggestions.append('Add more project impact metrics and quantifiable results in the Projects section.')
    if section_scores.get('Skills', 0) < 70:
        suggestions.append('Expand the Skills section with exact tools, libraries, and technologies you used.')
    if not re.search(r'\d', resume_text):
        suggestions.append('Include more measurable outcomes such as percentages, dollar savings, or time savings.')
    if 'python' in resume_lower and not any(lib in resume_lower for lib in ['pandas', 'numpy', 'opencv', 'tensorflow', 'keras', 'sklearn', 'scikit-learn']):
        suggestions.append('Include specific Python libraries and frameworks used, like pandas, NumPy, OpenCV, or scikit-learn.')
    if not any(h in resume_lower for h in ['summary', 'experience', 'skills', 'education']):
        suggestions.append('Use standard resume section headings to improve readability and ATS parsing.')

    if not suggestions:
        suggestions.append('Your resume is well-structured. Focus on adding more targeted keywords and impact metrics.')

    return suggestions


def build_skill_gap_analysis(resume_text, job_description, job_keywords):
    """Compare candidate skills against job requirements."""
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    matching = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)
    present = sorted(resume_skills)

    skill_gap_df = pd.DataFrame(
        [[skill, '✔' if skill in matching else '', '✔' if skill in missing else ''] for skill in sorted(job_skills | resume_skills)],
        columns=['Skill', 'Present in Resume', 'Required but Missing']
    )

    return matching, missing, present, skill_gap_df


def recommend_jobs(skill_set):
    """Recommend job titles based on extracted skills."""
    role_map = {
        'Data Analyst': {'sql', 'excel', 'tableau', 'power bi', 'data analysis', 'business intelligence'},
        'Machine Learning Intern': {'machine learning', 'python', 'scikit-learn', 'tensorflow', 'keras', 'deep learning', 'data science'},
        'Data Scientist': {'python', 'machine learning', 'numpy', 'pandas', 'scikit-learn', 'tensorflow', 'statistics', 'data analysis'},
        'Business Intelligence Analyst': {'power bi', 'tableau', 'sql', 'excel', 'data visualization', 'business intelligence'},
        'Frontend Developer': {'javascript', 'react', 'html', 'css', 'vue', 'angular'},
        'Backend Developer': {'python', 'django', 'flask', 'node.js', 'api', 'sql'},
        'Full Stack Developer': {'javascript', 'react', 'node.js', 'python', 'django', 'api'},
        'DevOps Engineer': {'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd'},
        'Project Manager': {'agile', 'scrum', 'project management', 'jira', 'communication'}
    }

    scores = []
    for role, required_skills in role_map.items():
        match_count = len(skill_set & required_skills)
        if match_count > 0:
            scores.append((role, match_count))

    if not scores:
        return ['General Analyst', 'Operations Associate', 'Internship / Entry-Level Role']

    scores.sort(key=lambda x: (-x[1], x[0]))
    return [role for role, _ in scores[:4]]

# ============ LLM & RESUME REWRITING FUNCTIONS ============

def generate_interview_questions(resume_text, job_description, job_keywords):
    """Generate interview questions based on resume and job description"""
    
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()
    
    # Extract top keywords and skills
    top_keywords = [kw[0] for kw in job_keywords[:8]]
    
    questions = []
    
    # Category 1: Role-Specific Technical Questions (3-4 questions)
    st.subheader("🎯 Role-Specific Technical Questions")
    
    tech_questions = [
        f"Can you walk us through your experience with {top_keywords[0]}?",
        f"How have you used {top_keywords[1]} in your previous roles?",
        f"Tell us about a project where you applied {top_keywords[2]} to solve a problem.",
        f"What is your experience level with {top_keywords[3]}?",
    ]
    
    questions.extend(tech_questions[:3])
    
    col1, col2 = st.columns(2)
    with col1:
        for i, q in enumerate(tech_questions[:2], 1):
            st.write(f"**Q{i}:** {q}")
            tips = "💡 **Tip:** Mention specific projects, measurable outcomes, and quantify your achievements." if i == 1 else "💡 **Tip:** Use the STAR method (Situation, Task, Action, Result) to structure your answer."
            st.caption(tips)
    
    with col2:
        for i, q in enumerate(tech_questions[2:], 3):
            st.write(f"**Q{i}:** {q}")
            st.caption("💡 **Tip:** Focus on depth of experience and problem-solving approach.")
    
    # Category 2: Experience & Achievement Questions
    st.markdown("---")
    st.subheader("🏆 Experience & Achievement Questions")
    
    exp_questions = [
        "What has been your biggest professional achievement in your career?",
        "Can you describe a time when you faced a technical challenge and how you overcame it?",
        "Tell us about a project you're most proud of. What was your role and impact?",
        "How do you stay updated with the latest trends in " + top_keywords[0] + "?",
    ]
    
    questions.extend(exp_questions)
    
    col1, col2 = st.columns(2)
    with col1:
        for i, q in enumerate(exp_questions[:2], 1):
            st.write(f"**Q{len(tech_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Use concrete examples with numbers and percentages.")
    
    with col2:
        for i, q in enumerate(exp_questions[2:], 3):
            st.write(f"**Q{len(tech_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Research the company beforehand and align your answer with their values.")
    
    # Category 3: Problem-Solving & Approach Questions
    st.markdown("---")
    st.subheader("🧠 Problem-Solving Questions")
    
    problem_questions = [
        f"How would you approach building a solution using {top_keywords[2]}?",
        "Describe your process for debugging a complex issue.",
        "How do you prioritize tasks when working on multiple projects?",
        "Give an example of how you've contributed to improving team processes.",
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        for i, q in enumerate(problem_questions[:2], 1):
            st.write(f"**Q{len(tech_questions)+len(exp_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Explain your thought process step-by-step.")
    
    with col2:
        for i, q in enumerate(problem_questions[2:], 3):
            st.write(f"**Q{len(tech_questions)+len(exp_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Show teamwork, leadership, and initiative.")
    
    # Category 4: Culture Fit & Motivation Questions
    st.markdown("---")
    st.subheader("💼 Culture Fit & Motivation Questions")
    
    culture_questions = [
        "Why are you interested in this role and our company?",
        "How do you handle disagreements with team members?",
        "Where do you see yourself in 3-5 years?",
        "What motivates you in your work?",
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        for i, q in enumerate(culture_questions[:2], 1):
            st.write(f"**Q{len(tech_questions)+len(exp_questions)+len(problem_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Research company values and mission. Show genuine interest.")
    
    with col2:
        for i, q in enumerate(culture_questions[2:], 3):
            st.write(f"**Q{len(tech_questions)+len(exp_questions)+len(problem_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Align your goals with company growth. Be authentic and positive.")
    
    return tech_questions + exp_questions + problem_questions + culture_questions

def generate_preparation_tips(job_description, job_keywords):
    """Generate interview preparation tips based on job requirements"""
    
    st.markdown("---")
    st.subheader("📚 Interview Preparation Checklist")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Before the Interview:
        - ✅ Research the company thoroughly
        - ✅ Review the job description
        - ✅ Prepare specific examples (STAR method)
        - ✅ Practice common technical questions
        - ✅ Review your resume - be ready to discuss each point
        - ✅ Prepare thoughtful questions for interviewers
        """)
    
    with col2:
        st.markdown(f"""
        ### Key Skills to Highlight:
        - **{job_keywords[0][0].title()}** - Your expertise level
        - **{job_keywords[1][0].title()}** - Practical experience
        - **{job_keywords[2][0].title()}** - Project examples
        - **Communication** - Explain complex concepts clearly
        - **Problem-solving** - Think out loud during technical questions
        - **Teamwork** - Show collaboration examples
        """)
    
    st.markdown("---")
    st.subheader("💬 Questions to Ask Your Interviewer")
    
    questions_to_ask = [
        "What does success look like in this role?",
        "What are the biggest challenges the team is currently facing?",
        "How does this role contribute to the company's goals?",
        "What is the team structure and who would I be working closely with?",
        "What's the typical career progression for this role?",
        "What do you enjoy most about working here?"
    ]
    
    for i, q in enumerate(questions_to_ask, 1):
        st.write(f"**Q{i}:** {q}")

def generate_interview_questions(resume_text, job_description, job_keywords):
    """Generate interview questions based on resume and job description"""
    
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()
    
    # Extract top keywords and skills
    top_keywords = [kw[0] for kw in job_keywords[:8]]
    
    questions = []
    
    # Category 1: Role-Specific Technical Questions (3-4 questions)
    st.subheader("🎯 Role-Specific Technical Questions")
    
    tech_questions = [
        f"Can you walk us through your experience with {top_keywords[0]}?",
        f"How have you used {top_keywords[1]} in your previous roles?",
        f"Tell us about a project where you applied {top_keywords[2]} to solve a problem.",
        f"What is your experience level with {top_keywords[3]}?",
    ]
    
    questions.extend(tech_questions[:3])
    
    col1, col2 = st.columns(2)
    with col1:
        for i, q in enumerate(tech_questions[:2], 1):
            st.write(f"**Q{i}:** {q}")
            tips = "💡 **Tip:** Mention specific projects, measurable outcomes, and quantify your achievements." if i == 1 else "💡 **Tip:** Use the STAR method (Situation, Task, Action, Result) to structure your answer."
            st.caption(tips)
    
    with col2:
        for i, q in enumerate(tech_questions[2:], 3):
            st.write(f"**Q{i}:** {q}")
            st.caption("💡 **Tip:** Focus on depth of experience and problem-solving approach.")
    
    # Category 2: Experience & Achievement Questions
    st.markdown("---")
    st.subheader("🏆 Experience & Achievement Questions")
    
    exp_questions = [
        "What has been your biggest professional achievement in your career?",
        "Can you describe a time when you faced a technical challenge and how you overcame it?",
        "Tell us about a project you're most proud of. What was your role and impact?",
        "How do you stay updated with the latest trends in " + top_keywords[0] + "?",
    ]
    
    questions.extend(exp_questions)
    
    col1, col2 = st.columns(2)
    with col1:
        for i, q in enumerate(exp_questions[:2], 1):
            st.write(f"**Q{len(tech_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Use concrete examples with numbers and percentages.")
    
    with col2:
        for i, q in enumerate(exp_questions[2:], 3):
            st.write(f"**Q{len(tech_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Research the company beforehand and align your answer with their values.")
    
    # Category 3: Problem-Solving & Approach Questions
    st.markdown("---")
    st.subheader("🧠 Problem-Solving Questions")
    
    problem_questions = [
        f"How would you approach building a solution using {top_keywords[2]}?",
        "Describe your process for debugging a complex issue.",
        "How do you prioritize tasks when working on multiple projects?",
        "Give an example of how you've contributed to improving team processes.",
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        for i, q in enumerate(problem_questions[:2], 1):
            st.write(f"**Q{len(tech_questions)+len(exp_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Explain your thought process step-by-step.")
    
    with col2:
        for i, q in enumerate(problem_questions[2:], 3):
            st.write(f"**Q{len(tech_questions)+len(exp_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Show teamwork, leadership, and initiative.")
    
    # Category 4: Culture Fit & Motivation Questions
    st.markdown("---")
    st.subheader("💼 Culture Fit & Motivation Questions")
    
    culture_questions = [
        "Why are you interested in this role and our company?",
        "How do you handle disagreements with team members?",
        "Where do you see yourself in 3-5 years?",
        "What motivates you in your work?",
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        for i, q in enumerate(culture_questions[:2], 1):
            st.write(f"**Q{len(tech_questions)+len(exp_questions)+len(problem_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Research company values and mission. Show genuine interest.")
    
    with col2:
        for i, q in enumerate(culture_questions[2:], 3):
            st.write(f"**Q{len(tech_questions)+len(exp_questions)+len(problem_questions)+i}:** {q}")
            st.caption("💡 **Tip:** Align your goals with company growth. Be authentic and positive.")
    
    return tech_questions + exp_questions + problem_questions + culture_questions

def create_section_gauge_chart(section_scores):
    """Create a simple section score bar chart."""
    chart_data = pd.DataFrame(
        {'Match Score': list(section_scores.values())},
        index=list(section_scores.keys())
    )
    st.bar_chart(chart_data)
    st.markdown("**Section-wise match scores** — higher is better. Use this to spot weak areas.")

# Main app
def main():
    st.subheader("📤 Upload Your Documents")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your **resume (PDF)**", type=['pdf'])
    
    with col2:
        st.markdown("### 📝 Job Description")
        job_description = st.text_area("", height=250, 
                                     placeholder="Paste the full job description here...")
    
    if st.button("🚀 **Analyze Resume Match**", type="primary"):
        if not uploaded_file:
            st.warning("❌ **Please upload your resume**")
            return
        if not job_description.strip():
            st.warning("❌ **Please paste the job description**")
            return
        
        with st.spinner("🔍 Analyzing your resume sections & job match..."):
            # Extract text and analyze sections
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("❌ Could not extract text from PDF. Please try another file.")
                return 
            
            # Store in session state
            st.session_state.resume_text = resume_text
            st.session_state.job_description = job_description
            
            # Get overall and section scores
            overall_score, section_scores, sections = extract_section_scores(
                resume_text, job_description
            )
            
            # Extract keywords
            job_processed = remove_stopwords(clean_text(job_description))
            job_keywords = extract_keywords(job_processed)
            resume_processed = remove_stopwords(clean_text(resume_text))
            resume_keywords = extract_keywords(resume_processed)
            
            # Get section improvements
            section_improvements = get_section_improvements(job_keywords, section_scores)
            
            # Store results in session state
            st.session_state.overall_score = overall_score
            st.session_state.section_scores = section_scores
            st.session_state.job_keywords = job_keywords
            st.session_state.resume_keywords = resume_keywords
            st.session_state.section_improvements = section_improvements
            st.session_state.analysis_complete = True
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete:
        # === RESULTS SECTION ===
        st.markdown("---")
        st.subheader("📊 **Match Analysis Results**")
        
        # Overall Score & Section Radar Chart
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("🎯 **Overall Match Score**", f"{st.session_state.overall_score:.1f}%")
        
        with col2:
            create_section_gauge_chart(st.session_state.section_scores)
        
        # Overall feedback
        if st.session_state.overall_score < 40:
            st.error("🔴 **Low Overall Match** - Significant tailoring needed across sections!")
        elif st.session_state.overall_score < 70:
            st.info("🟡 **Good Overall Match** - Focus on weak sections")
        else:
            st.success("🟢 **Excellent Overall Match!** - Strong across most sections")

        st.markdown("---")
        st.info("🔒 Login or sign up to unlock the full detailed report: section analysis, ATS scoring, skills gap, recommendations, and interview questions.")
        with st.expander("🔐 Login or Sign Up for Full Report"):
            login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

            with login_tab:
                st.markdown("### 📧 Login with Gmail")
                gmail_login = st.text_input("Enter your Gmail address", placeholder="your.email@gmail.com", key="gmail_login")
                if st.button("📨 Continue with Gmail", key="gmail_login_btn"):
                    success, message = signup_with_gmail(gmail_login)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.current_user = gmail_login.split('@')[0]
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                
                st.markdown("---")
                st.markdown("### 🔐 Or Login with Username")
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                if st.button("Login", key="login_btn"):
                    if check_credentials(username, password):
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        st.success("✅ Login successful. Full report unlocked.")
                    else:
                        st.error("❌ Invalid credentials. Please try again.")

            with signup_tab:
                st.markdown("### 📧 Sign Up with Gmail")
                new_gmail = st.text_input("Enter your Gmail address", placeholder="your.email@gmail.com", key="signup_gmail")
                if st.button("📨 Sign Up with Gmail", key="signup_gmail_btn"):
                    success, message = signup_with_gmail(new_gmail)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.current_user = new_gmail.split('@')[0]
                        st.success(f"✅ {message} You can now view the full report.")
                    else:
                        st.error(f"❌ {message}")
                
                st.markdown("---")
                st.markdown("### 🔐 Or Sign Up with Username")
                new_user = st.text_input("Choose a username", key="signup_user")
                new_pass = st.text_input("Choose a password", type="password", key="signup_pass")
                confirm_pass = st.text_input("Confirm password", type="password", key="signup_confirm")
                if st.button("Sign Up", key="signup_btn"):
                    if new_pass != confirm_pass:
                        st.error("❌ Passwords do not match.")
                    else:
                        success, message = add_new_user(new_user, new_pass)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)

        if not st.session_state.authenticated:
            st.markdown("### 🔒 Full report locked")
            st.write("You can view the overall score and chart above. Create an account or login to see the rest of the detailed analysis.")
            return

        # === SECTION-WISE ANALYSIS ===
        st.markdown("---")
        st.subheader("📋 **Section-wise Analysis**")
        
        section_df_data = []
        for section, score in st.session_state.section_scores.items():
            status = "🔴 Critical" if score < 30 else "🟡 Needs Work" if score < 60 else "🟢 Good" if score < 80 else "✅ Excellent"
            section_df_data.append([section, f"{score:.1f}%", status])
        
        section_df = pd.DataFrame(section_df_data, columns=['Section', 'Match Score', 'Status'])
        st.dataframe(section_df, use_container_width=True)

        # === DETAILED SECTION RECOMMENDATIONS ===
        st.markdown("---")
        st.subheader("🔧 **Section-specific Improvement Plan**")
        
        for section, info in st.session_state.section_improvements.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric(f"📂 {section}", f"{info['score']:.1f}%")
            with col2:
                st.markdown(f"**{info['status']}**")
                for action in info['actions']:
                    st.markdown(f"• {action}")

        # === KEYWORD ANALYSIS ===
        st.markdown("---")
        st.subheader("🔑 **Keyword Analysis**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📋 **Top Job Keywords**")
            job_df = pd.DataFrame(st.session_state.job_keywords[:10], columns=['Keyword', 'Frequency'])
            st.dataframe(job_df, use_container_width=True)
        
        with col2:
            st.markdown("### ✅ **Your Resume Keywords**")
            resume_df = pd.DataFrame(st.session_state.resume_keywords[:10], columns=['Keyword', 'Frequency'])
            st.dataframe(resume_df, use_container_width=True)

        # === ATS COMPATIBILITY CHECKER ===
        st.markdown("---")
        st.subheader("🔥 **ATS Compatibility Checker**")

        resume_skill_set = extract_skills(st.session_state.resume_text)
        job_skill_set = extract_skills(st.session_state.job_description)
        ats_score, ats_suggestions = compute_ats_score(
            st.session_state.resume_text,
            st.session_state.job_description,
            st.session_state.section_scores,
            resume_skill_set,
            job_skill_set
        )

        st.metric("🧾 ATS Score", f"{ats_score}/100")
        st.markdown("**ATS Improvement Suggestions:**")
        for suggestion in ats_suggestions:
            st.markdown(f"- {suggestion}")

        # === SKILL GAP ANALYSIS ===
        st.markdown("---")
        st.subheader("📈 **Skill Gap Analysis**")

        matching, missing, present, skill_gap_df = build_skill_gap_analysis(
            st.session_state.resume_text,
            st.session_state.job_description,
            st.session_state.job_keywords
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ Matching Skills")
            if matching:
                for skill in matching:
                    st.markdown(f"- {skill}")
            else:
                st.write("No exact matches detected yet.")
        with col2:
            st.markdown("### ❌ Missing Skills")
            if missing:
                for skill in missing[:12]:
                    st.markdown(f"- {skill}")
            else:
                st.write("Great job — no missing skills detected from key terms.")

        st.markdown("### 📊 Skill Gap Table")
        st.dataframe(skill_gap_df, use_container_width=True)

        # === RESUME IMPROVEMENT SUGGESTIONS ===
        st.markdown("---")
        st.subheader("✨ **Resume Improvement Suggestions**")

        improvement_suggestions = generate_resume_improvement_suggestions(
            st.session_state.resume_text,
            st.session_state.job_description,
            st.session_state.section_scores,
            st.session_state.job_keywords,
            st.session_state.resume_keywords
        )
        for suggestion in improvement_suggestions:
            st.markdown(f"- {suggestion}")

        # === JOB RECOMMENDATION SYSTEM ===
        st.markdown("---")
        st.subheader("🤖 **Job Recommendation System**")
        recommended_roles = recommend_jobs(resume_skill_set)
        st.markdown("### Recommended Roles")
        for role in recommended_roles:
            st.markdown(f"- {role}")

        if missing:
            st.markdown("### Recommended Skills to Learn")
            for skill in missing[:8]:
                st.markdown(f"- {skill}")

        # === INTERVIEW QUESTIONS GENERATOR SECTION ===
        st.markdown("---")
        st.subheader("✨ **Interview Questions Generator**")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info("🎯 Get AI-generated interview questions tailored to this job!")
        
        if st.button("🚀 **Generate Interview Questions**", type="primary", key="interview_btn"):
            with st.spinner("📝 Generating interview questions..."):
                st.success("✅ Interview questions generated!")
                st.markdown("---")
                
                # Generate and display questions
                generate_interview_questions(
                    st.session_state.resume_text,
                    st.session_state.job_description,
                    st.session_state.job_keywords
                )
                
                # Generate preparation tips
                generate_preparation_tips(
                    st.session_state.job_description,
                    st.session_state.job_keywords
                )
        
        # === PRO TIPS ===
        with st.expander("📚 **Pro Tips for Interview Success**"):
            st.markdown("""
            **Before the Interview:**
            1. Research the company thoroughly
            2. Prepare specific examples from your resume
            3. Practice with STAR method (Situation, Task, Action, Result)
            4. Review the job description and identify key requirements
            5. Prepare thoughtful questions for the interviewer
            
            **During the Interview:**
            - Speak clearly and maintain confident body language
            - Listen carefully and answer questions completely
            - Use concrete examples and quantifiable achievements
            - Show enthusiasm for the role and company
            - Ask insightful questions about the role and team
            """)

if __name__ == "__main__":
    main()








