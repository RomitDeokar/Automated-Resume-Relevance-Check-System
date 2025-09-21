# Enhanced Automated Resume Relevance Check System
# Complete implementation for Innomatics Research Labs Hackathon
# Version 2.0 - Enhanced with Beautiful UI, Bug Fixes, and New Features

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import PyPDF2
import docx2txt
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import io
import json
from typing import Dict, List, Tuple, Optional
import os
from dataclasses import dataclass
import hashlib
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64

def enhanced_home_page(components):
    """Enhanced home page with comprehensive dashboard"""
    st.markdown("## Welcome to Innomatics Resume System")
    st.write("System is ready for use!")
    
    # Basic metrics
    jobs = components['db'].get_job_descriptions()
    st.metric("Active Jobs", len(jobs))
# Enhanced imports for better functionality
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("⚠️ Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

# Enhanced Gemini AI Integration
class GeminiAI:
    """Enhanced Google Gemini AI integration for semantic analysis"""
    
    def __init__(self):
        if GEMINI_AVAILABLE:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.enabled = True
            else:
                self.enabled = False
        else:
            self.enabled = False
    
    def analyze_resume_fit(self, resume_text: str, job_description: str) -> Dict:
        """Enhanced resume fit analysis using Gemini AI"""
        if not self.enabled:
            return {
                'semantic_score': 50.0,
                'ai_suggestions': ["🤖 Set up Gemini API key for AI-powered suggestions"],
                'skill_analysis': {},
                'improvement_areas': [],
                'strengths': []
            }
        
        prompt = f"""
        Analyze the following resume against the job description and provide a comprehensive assessment for Innomatics Research Labs placement system.
        
        Job Description:
        {job_description}
        
        Resume:
        {resume_text}
        
        Please provide a detailed analysis in the following format:
        
        SCORE: [0-100 numeric score based on how well the resume matches the job requirements]
        
        SUGGESTIONS:
        - [specific improvement suggestion 1]
        - [specific improvement suggestion 2]
        - [specific improvement suggestion 3]
        
        STRENGTHS:
        - [candidate strength 1]
        - [candidate strength 2]
        - [candidate strength 3]
        
        IMPROVEMENT_AREAS:
        - [area that needs improvement 1]
        - [area that needs improvement 2]
        
        ANALYSIS:
        [Detailed paragraph explaining the overall fit, skill gaps, and recommendation]
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_gemini_response(response.text)
        except Exception as e:
            st.warning(f"⚠️ Gemini AI temporarily unavailable: {str(e)[:100]}...")
            return {
                'semantic_score': 50.0,
                'ai_suggestions': ["🔄 AI analysis temporarily unavailable - using rule-based scoring"],
                'skill_analysis': {},
                'improvement_areas': [],
                'strengths': []
            }
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Enhanced parsing of Gemini AI response"""
        lines = response_text.split('\n')
        
        score = 50.0
        suggestions = []
        strengths = []
        improvement_areas = []
        analysis = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('SCORE:'):
                try:
                    score_text = line.split(':')[1].strip()
                    # Extract numeric value from text
                    score_match = re.search(r'\d+', score_text)
                    if score_match:
                        score = float(score_match.group())
                except:
                    score = 50.0
            elif line.startswith('SUGGESTIONS:'):
                current_section = 'suggestions'
            elif line.startswith('STRENGTHS:'):
                current_section = 'strengths'
            elif line.startswith('IMPROVEMENT_AREAS:'):
                current_section = 'improvement_areas'
            elif line.startswith('ANALYSIS:'):
                current_section = 'analysis'
            elif line.startswith('- ') and current_section == 'suggestions':
                suggestions.append(line[2:])
            elif line.startswith('- ') and current_section == 'strengths':
                strengths.append(line[2:])
            elif line.startswith('- ') and current_section == 'improvement_areas':
                improvement_areas.append(line[2:])
            elif current_section == 'analysis' and line:
                analysis += line + " "
        
        return {
            'semantic_score': min(100, max(0, score)),
            'ai_suggestions': suggestions[:3],
            'skill_analysis': {'summary': analysis.strip()},
            'improvement_areas': improvement_areas[:3],
            'strengths': strengths[:3]
        }

# Enhanced Free Embedding Generator
class FreeEmbeddingGenerator:
    """Enhanced free embedding generation using sentence-transformers"""
    
    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.enabled = True
                st.success("✅ Advanced semantic matching enabled")
            except Exception as e:
                st.warning(f"⚠️ Advanced embedding model not available: {e}. Using TF-IDF fallback.")
                self.enabled = False
        else:
            self.enabled = False
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts with error handling"""
        if not self.enabled:
            return np.array([])
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings
        except Exception as e:
            st.warning(f"⚠️ Embedding generation error: {e}")
            return np.array([])

# Enhanced Resume Analysis Data Structure
@dataclass
class EnhancedResumeAnalysis:
    relevance_score: float
    verdict: str
    missing_skills: List[str]
    matching_skills: List[str]
    suggestions: List[str]
    experience_match: bool
    education_match: bool
    strengths: List[str]
    improvement_areas: List[str]
    ai_feedback: str
    location: str
    contact_info: Dict
    projects_count: int
    certifications_count: int

# Enhanced Resume Parser with better extraction
class EnhancedResumeParser:
    """Enhanced resume parser with better text extraction and information parsing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Enhanced PDF text extraction with fallback methods"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + "\n"
                except Exception as e:
                    st.warning(f"⚠️ Could not extract text from page {page_num + 1}: {e}")
                    continue
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            return text.strip()
        except Exception as e:
            st.error(f"❌ Error reading PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(docx_file) -> str:
        """Enhanced DOCX text extraction with better error handling"""
        try:
            text = docx2txt.process(docx_file)
            if not text.strip():
                raise ValueError("No text could be extracted from DOCX")
            return text.strip()
        except Exception as e:
            st.error(f"❌ Error reading DOCX: {e}")
            return ""
    
    def extract_resume_info(self, text: str) -> Dict:
        """Enhanced information extraction from resume text"""
        info = {
            'skills': self._extract_skills(text),
            'experience_years': self._extract_experience_years(text),
            'education': self._extract_education(text),
            'certifications': self._extract_certifications(text),
            'projects': self._extract_projects(text),
            'location': self._extract_location(text),
            'contact_info': self._extract_contact_info(text),
            'languages': self._extract_languages(text),
            'achievements': self._extract_achievements(text)
        }
        return info
    
    def _extract_skills(self, text: str) -> List[str]:
        """Enhanced technical skills extraction with broader coverage"""
        # Expanded skills pattern for better coverage
        skills_patterns = [
            # Programming Languages
            r'\b(?:python|java|javascript|typescript|c\+\+|c#|php|ruby|go|rust|kotlin|swift|scala|r|matlab|perl)\b',
            # Web Technologies
            r'\b(?:react|angular|vue|node\.?js|express|django|flask|fastapi|spring|laravel|rails|asp\.net)\b',
            # Databases
            r'\b(?:sql|mysql|postgresql|mongodb|redis|cassandra|elasticsearch|sqlite|oracle|dynamodb)\b',
            # Cloud & DevOps
            r'\b(?:aws|azure|gcp|google cloud|docker|kubernetes|jenkins|gitlab|github|terraform|ansible)\b',
            # Data Science & AI
            r'\b(?:machine learning|deep learning|ai|artificial intelligence|data science|pandas|numpy|tensorflow|pytorch|scikit-learn|opencv|nlp|computer vision)\b',
            # Frontend
            r'\b(?:html|css|sass|scss|bootstrap|tailwind|jquery|webpack|babel|typescript)\b',
            # Tools & Others
            r'\b(?:git|svn|jira|confluence|slack|figma|photoshop|illustrator|linux|windows|macos|agile|scrum|devops|ci/cd)\b'
        ]
        
        skills = []
        text_lower = text.lower()
        
        for pattern in skills_patterns:
            matches = re.findall(pattern, text_lower)
            skills.extend(matches)
        
        # Clean and deduplicate
        skills = [skill.replace('.', '') for skill in skills]
        return list(set(skills))
    
    def _extract_experience_years(self, text: str) -> int:
        """Enhanced experience extraction with multiple patterns"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'experience\s*[:of]*\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*yrs?\s*(?:experience|exp)',
            r'(\d+)\+?\s*year\s*(?:experience|exp)',
            r'over\s*(\d+)\s*years?',
            r'more than\s*(\d+)\s*years?'
        ]
        
        max_experience = 0
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                for match in matches:
                    try:
                        exp_years = int(match)
                        max_experience = max(max_experience, exp_years)
                    except ValueError:
                        continue
        
        return max_experience
    
    def _extract_education(self, text: str) -> List[str]:
        """Enhanced education extraction"""
        edu_patterns = [
            r'\b(?:bachelor|master|phd|doctorate|diploma|degree|graduate|undergraduate)\b',
            r'\b(?:b\.?tech|m\.?tech|b\.?e|m\.?e|bsc|msc|ba|ma|mba|mca|bca)\b',
            r'\b(?:engineering|computer science|information technology|electronics|mechanical|civil)\b'
        ]
        
        education = []
        text_lower = text.lower()
        
        for pattern in edu_patterns:
            matches = re.findall(pattern, text_lower)
            education.extend(matches)
        
        return list(set(education))
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Enhanced certification extraction"""
        cert_patterns = [
            r'\b(?:aws|azure|gcp|google cloud|microsoft|oracle|cisco|comptia)\b.*?(?:certified|certification)',
            r'(?:certified|certification).*?\b(?:aws|azure|gcp|google cloud|microsoft|oracle|cisco)\b',
            r'\b(?:cissp|ceh|oscp|cisa|cism|pmp|itil|prince2)\b'
        ]
        
        certs = []
        text_lower = text.lower()
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text_lower)
            certs.extend(matches)
        
        return list(set(certs))
    
    def _extract_projects(self, text: str) -> List[str]:
        """Enhanced project extraction with better parsing"""
        project_indicators = [
            'project', 'projects', 'work', 'portfolio', 'github', 'git',
            'developed', 'built', 'created', 'implemented', 'designed'
        ]
        
        projects = []
        lines = text.split('\n')
        in_project_section = False
        
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Check if this line indicates a project section
            if any(indicator in line_lower for indicator in project_indicators[:5]):
                in_project_section = True
                continue
            
            # If we're in a project section and line has meaningful content
            if in_project_section and line_stripped:
                if (len(line_stripped.split()) > 5 and 
                    any(indicator in line_lower for indicator in project_indicators[5:])):
                    projects.append(line_stripped[:100])  # Limit length
                    if len(projects) >= 10:  # Limit to 10 projects
                        break
        
        return projects
    
    def _extract_location(self, text: str) -> str:
        """Extract location information"""
        # Indian cities pattern for Innomatics Research Labs
        cities = [
            'hyderabad', 'bangalore', 'pune', 'delhi', 'mumbai', 'chennai', 
            'kolkata', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur',
            'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri', 'patna',
            'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad'
        ]
        
        text_lower = text.lower()
        for city in cities:
            if city in text_lower:
                return city.title()
        
        return "Not specified"
    
    def _extract_contact_info(self, text: str) -> Dict:
        """Extract contact information"""
        contact_info = {
            'email': '',
            'phone': '',
            'linkedin': '',
            'github': ''
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone pattern (Indian numbers)
        phone_pattern = r'(?:\+91|0)?[6-9]\d{9}'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phone'] = phones[0]
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text.lower())
        if linkedin:
            contact_info['linkedin'] = linkedin[0]
        
        # GitHub pattern
        github_pattern = r'github\.com/[\w-]+'
        github = re.findall(github_pattern, text.lower())
        if github:
            contact_info['github'] = github[0]
        
        return contact_info
    
    def _extract_languages(self, text: str) -> List[str]:
        """Extract programming and spoken languages"""
        prog_languages = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
            'go', 'rust', 'kotlin', 'swift', 'scala', 'r', 'matlab'
        ]
        
        found_languages = []
        text_lower = text.lower()
        
        for lang in prog_languages:
            if lang in text_lower:
                found_languages.append(lang)
        
        return found_languages
    
    def _extract_achievements(self, text: str) -> List[str]:
        """Extract achievements and awards"""
        achievement_indicators = [
            'award', 'achievement', 'recognition', 'honor', 'winner', 'champion',
            'first place', 'second place', 'third place', 'medal', 'certificate'
        ]
        
        achievements = []
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in achievement_indicators):
                if len(line.strip()) > 10:  # Meaningful achievement
                    achievements.append(line.strip()[:100])  # Limit length
        
        return achievements[:5]  # Limit to 5 achievements

# Enhanced Job Description Parser
class EnhancedJobDescriptionParser:
    """Enhanced job description parser with better requirement extraction"""
    
    def parse_jd(self, jd_text: str) -> Dict:
        """Enhanced JD parsing with comprehensive information extraction"""
        return {
            'required_skills': self._extract_required_skills(jd_text),
            'preferred_skills': self._extract_preferred_skills(jd_text),
            'experience_required': self._extract_experience_requirement(jd_text),
            'education_required': self._extract_education_requirement(jd_text),
            'role_type': self._extract_role_type(jd_text),
            'location': self._extract_location(jd_text),
            'salary_range': self._extract_salary_range(jd_text),
            'company_size': self._extract_company_size(jd_text),
            'responsibilities': self._extract_responsibilities(jd_text)
        }
    
    def _extract_required_skills(self, text: str) -> List[str]:
        """Enhanced required skills extraction"""
        # Look for required skills sections with better patterns
        required_sections = re.findall(
            r'(?:required|must have|essential|mandatory).*?(?:skills?|technologies?|qualifications?)[:\s]*(.*?)(?:\n\s*\n|\n(?:[A-Z][a-z]+:|$))', 
            text, re.IGNORECASE | re.DOTALL
        )
        
        skills = []
        for section in required_sections:
            # Extract technical terms with better pattern
            tech_skills = re.findall(r'\b[A-Za-z][A-Za-z0-9+#\.\-]*\b', section)
            skills.extend([skill.lower() for skill in tech_skills if len(skill) > 2 and skill.lower() not in ['the', 'and', 'for', 'with', 'you', 'are', 'have', 'will']])
        
        # Common technical skills extraction
        common_skills_pattern = r'\b(?:python|java|javascript|react|node|angular|vue|sql|mongodb|postgresql|mysql|aws|azure|gcp|docker|kubernetes|git|machine learning|deep learning|ai|data science|pandas|numpy|tensorflow|pytorch|scikit-learn|html|css|bootstrap|spring|django|flask|fastapi|rest|api|microservices|agile|scrum|devops|ci/cd|jenkins|linux|windows|typescript|php|ruby|go|rust|kotlin|swift)\b'
        
        common_skills = re.findall(common_skills_pattern, text.lower())
        skills.extend(common_skills)
        
        return list(set(skills))
    
    def _extract_preferred_skills(self, text: str) -> List[str]:
        """Enhanced preferred skills extraction"""
        preferred_sections = re.findall(
            r'(?:preferred|good to have|nice to have|plus|bonus|additional).*?(?:skills?|experience)[:\s]*(.*?)(?:\n\s*\n|\n(?:[A-Z][a-z]+:|$))', 
            text, re.IGNORECASE | re.DOTALL
        )
        
        skills = []
        for section in preferred_sections:
            tech_skills = re.findall(r'\b[A-Za-z][A-Za-z0-9+#\.\-]*\b', section)
            skills.extend([skill.lower() for skill in tech_skills if len(skill) > 2])
        
        return list(set(skills))
    
    def _extract_experience_requirement(self, text: str) -> int:
        """Enhanced experience requirement extraction"""
        patterns = [
            r'(\d+)\+?\s*(?:to\s*\d+\s*)?years?\s*(?:of\s*)?experience',
            r'minimum\s*(?:of\s*)?(\d+)\s*years?',
            r'(\d+)\+?\s*yrs?\s*(?:of\s*)?experience',
            r'experience\s*[:of]*\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*year\s*(?:of\s*)?experience'
        ]
        
        max_experience = 0
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                for match in matches:
                    try:
                        exp_years = int(match)
                        max_experience = max(max_experience, exp_years)
                    except ValueError:
                        continue
        
        return max_experience
    
    def _extract_education_requirement(self, text: str) -> List[str]:
        """Enhanced education requirement extraction"""
        edu_patterns = [
            r'\b(?:bachelor|master|phd|doctorate|degree|graduate|undergraduate)\b',
            r'\b(?:b\.?tech|m\.?tech|b\.?e|m\.?e|bsc|msc|ba|ma|mba|mca|bca)\b',
            r'\b(?:engineering|computer science|information technology|electronics|mechanical|civil)\b'
        ]
        
        education = []
        text_lower = text.lower()
        
        for pattern in edu_patterns:
            matches = re.findall(pattern, text_lower)
            education.extend(matches)
        
        return list(set(education))
    
    def _extract_role_type(self, text: str) -> str:
        """Enhanced role type extraction"""
        text_lower = text.lower()
        
        role_indicators = {
            'frontend': ['frontend', 'front-end', 'ui', 'user interface', 'react', 'angular', 'vue', 'html', 'css', 'javascript'],
            'backend': ['backend', 'back-end', 'server', 'api', 'database', 'microservices', 'rest', 'node', 'python', 'java'],
            'fullstack': ['fullstack', 'full-stack', 'full stack'],
            'data_science': ['data science', 'data scientist', 'machine learning', 'ai', 'analytics', 'data analyst', 'ml engineer'],
            'devops': ['devops', 'infrastructure', 'cloud', 'aws', 'azure', 'kubernetes', 'docker', 'ci/cd'],
            'mobile': ['mobile', 'android', 'ios', 'react native', 'flutter', 'kotlin', 'swift'],
            'qa': ['qa', 'quality assurance', 'testing', 'automation testing', 'selenium', 'test engineer']
        }
        
        for role_type, indicators in role_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return role_type
        
        return 'general'
    
    def _extract_location(self, text: str) -> str:
        """Extract job location"""
        # Indian cities for Innomatics Research Labs
        cities = ['hyderabad', 'bangalore', 'pune', 'delhi', 'mumbai', 'chennai', 'kolkata', 'gurgaon', 'noida']
        
        text_lower = text.lower()
        for city in cities:
            if city in text_lower:
                return city.title()
        
        # Check for remote work
        if any(term in text_lower for term in ['remote', 'work from home', 'wfh']):
            return 'Remote'
        
        return 'Not specified'
    
    def _extract_salary_range(self, text: str) -> str:
        """Extract salary range information"""
        salary_patterns = [
            r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|lac|l|lakhs|lacs|ls)?\s*(?:to|[-–])\s*₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:lakh|lac|l|lakhs|lacs|ls)?',
            r'(\d+(?:\.\d+)?)\s*(?:lakh|lac|l|lakhs|lacs|ls)\s*(?:to|[-–])\s*(\d+(?:\.\d+)?)\s*(?:lakh|lac|l|lakhs|lacs|ls)',
            r'₹\s*(\d+(?:,\d+)*)',
            r'salary.*?₹\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        text_lower = text.lower()
        for pattern in salary_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                return f"₹{matches[0]} (estimated)"
        
        return "Not specified"
    
    def _extract_company_size(self, text: str) -> str:
        """Extract company size information"""
        size_patterns = [
            r'(\d+)\+?\s*(?:employees|people|team members)',
            r'team of (\d+)\+?',
            r'(\d+)\+?\s*person team'
        ]
        
        for pattern in size_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                size = int(matches[0])
                if size < 50:
                    return "Startup (< 50)"
                elif size < 200:
                    return "Small (50-200)"
                elif size < 1000:
                    return "Medium (200-1000)"
                else:
                    return "Large (1000+)"
        
        return "Not specified"
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract key responsibilities"""
        responsibility_patterns = [
            r'(?:responsibilities|duties|role)[:\s]*(.*?)(?:\n\s*\n|requirements|qualifications)',
            r'(?:you will|the candidate will)[:\s]*(.*?)(?:\n\s*\n|requirements|qualifications)'
        ]
        
        responsibilities = []
        for pattern in responsibility_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by bullet points or numbered lists
                resp_list = re.split(r'[•\-\*\d+\.]\s*', match)
                for resp in resp_list:
                    resp = resp.strip()
                    if len(resp) > 20:  # Meaningful responsibility
                        responsibilities.append(resp[:100])  # Limit length
        
        return responsibilities[:5]  # Limit to 5 responsibilities

# Enhanced Relevance Scorer with improved algorithms
class EnhancedRelevanceScorer:
    """Enhanced relevance scoring with multiple algorithms and better accuracy"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=2000, 
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.embedding_generator = FreeEmbeddingGenerator()
        self.gemini_ai = GeminiAI()
    
    def calculate_relevance(self, resume_info: Dict, jd_info: Dict, resume_text: str, jd_text: str) -> EnhancedResumeAnalysis:
        """Enhanced comprehensive relevance calculation"""
        
        # 1. Hard Match Score (30% weight) - Exact skill matches
        hard_score = self._calculate_hard_match(resume_info, jd_info)
        
        # 2. Semantic Match Score (30% weight) - Content similarity
        semantic_score = self._calculate_semantic_match(resume_text, jd_text)
        
        # 3. Experience Match Score (20% weight) - Experience alignment
        exp_score = self._calculate_experience_match(resume_info, jd_info)
        
        # 4. Education Match Score (10% weight) - Education alignment
        edu_score = self._calculate_education_match(resume_info, jd_info)
        
        # 5. AI Enhancement Score (10% weight) - Gemini AI insights
        ai_analysis = self.gemini_ai.analyze_resume_fit(resume_text, jd_text)
        ai_score = ai_analysis.get('semantic_score', 50.0)
        
        # Combined weighted score
        final_score = (
            hard_score * 0.3 + 
            semantic_score * 0.3 + 
            exp_score * 0.2 + 
            edu_score * 0.1 + 
            ai_score * 0.1
        )
        final_score = min(100, max(0, final_score))
        
        # Enhanced verdict with more nuanced thresholds
        verdict = self._get_enhanced_verdict(final_score, hard_score, exp_score)
        
        # Find missing and matching skills
        missing_skills = self._find_missing_skills(resume_info['skills'], jd_info['required_skills'])
        matching_skills = self._find_matching_skills(resume_info['skills'], jd_info['required_skills'])
        
        # Enhanced suggestions
        suggestions = self._generate_enhanced_suggestions(resume_info, jd_info, missing_skills, ai_analysis)
        
        return EnhancedResumeAnalysis(
            relevance_score=final_score,
            verdict=verdict,
            missing_skills=missing_skills,
            matching_skills=matching_skills,
            suggestions=suggestions,
            experience_match=resume_info['experience_years'] >= jd_info['experience_required'],
            education_match=bool(set(resume_info['education']).intersection(set(jd_info['education_required']))),
            strengths=ai_analysis.get('strengths', []),
            improvement_areas=ai_analysis.get('improvement_areas', []),
            ai_feedback=ai_analysis.get('skill_analysis', {}).get('summary', ''),
            location=resume_info.get('location', 'Not specified'),
            contact_info=resume_info.get('contact_info', {}),
            projects_count=len(resume_info.get('projects', [])),
            certifications_count=len(resume_info.get('certifications', []))
        )
    
    def _calculate_hard_match(self, resume_info: Dict, jd_info: Dict) -> float:
        """Enhanced hard match calculation with fuzzy matching"""
        required_skills = set(skill.lower() for skill in jd_info['required_skills'])
        resume_skills = set(skill.lower() for skill in resume_info['skills'])
        
        if not required_skills:
            return 70.0  # Neutral-positive score if no specific requirements
        
        # Exact matches
        exact_matches = len(required_skills.intersection(resume_skills))
        
        # Fuzzy matches (similar skills)
        fuzzy_matches = 0
        for req_skill in required_skills:
            if req_skill not in resume_skills:
                for res_skill in resume_skills:
                    if self._skills_similar(req_skill, res_skill):
                        fuzzy_matches += 0.5  # Half credit for fuzzy match
                        break
        
        total_matches = exact_matches + fuzzy_matches
        total_required = len(required_skills)
        
        score = (total_matches / total_required) * 100
        return min(100, score)
    
    def _skills_similar(self, skill1: str, skill2: str) -> bool:
        """Check if two skills are similar"""
        similar_skills = {
            'javascript': ['js', 'node', 'nodejs'],
            'python': ['py', 'django', 'flask'],
            'java': ['spring', 'springboot'],
            'react': ['reactjs', 'react.js'],
            'angular': ['angularjs', 'angular.js'],
            'machine learning': ['ml', 'ai', 'artificial intelligence'],
            'deep learning': ['dl', 'neural networks'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb'],
            'cloud': ['aws', 'azure', 'gcp', 'google cloud']
        }
        
        for base_skill, variants in similar_skills.items():
            if skill1 in [base_skill] + variants and skill2 in [base_skill] + variants:
                return True
        
        # Simple substring matching
        if len(skill1) > 3 and len(skill2) > 3:
            if skill1 in skill2 or skill2 in skill1:
                return True
        
        return False
    
    def _calculate_semantic_match(self, resume_text: str, jd_text: str) -> float:
        """Enhanced semantic matching using multiple methods"""
        try:
            # Method 1: TF-IDF + Cosine Similarity
            texts = [resume_text, jd_text]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Method 2: Embedding-based similarity (if available)
            embedding_similarity = 0.5  # Default
            if self.embedding_generator.enabled:
                embeddings = self.embedding_generator.get_embeddings(texts)
                if len(embeddings) == 2:
                    embedding_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            # Combine similarities with weights
            combined_similarity = (tfidf_similarity * 0.6) + (embedding_similarity * 0.4)
            return combined_similarity * 100
            
        except Exception as e:
            st.warning(f"⚠️ Semantic matching error: {e}")
            return 50.0  # Neutral score if calculation fails
    
    def _calculate_experience_match(self, resume_info: Dict, jd_info: Dict) -> float:
        """Enhanced experience matching with more nuanced scoring"""
        resume_exp = resume_info['experience_years']
        required_exp = jd_info['experience_required']
        
        if required_exp == 0:
            return 100.0  # No experience requirement
        
        if resume_exp >= required_exp:
            # Graduated bonus for excess experience
            if resume_exp <= required_exp * 1.5:
                return 100.0  # Perfect match
            elif resume_exp <= required_exp * 2:
                return 95.0   # Slight over-qualification
            else:
                return 85.0   # Significant over-qualification (might be expensive)
        else:
            # Graduated penalty for insufficient experience
            ratio = resume_exp / required_exp
            if ratio >= 0.8:
                return 85.0   # Close to requirement
            elif ratio >= 0.6:
                return 70.0   # Somewhat below requirement
            elif ratio >= 0.4:
                return 50.0   # Significantly below requirement
            else:
                return 30.0   # Far below requirement
    
    def _calculate_education_match(self, resume_info: Dict, jd_info: Dict) -> float:
        """Enhanced education matching"""
        resume_edu = set(edu.lower() for edu in resume_info['education'])
        required_edu = set(edu.lower() for edu in jd_info['education_required'])
        
        if not required_edu:
            return 100.0  # No specific education requirement
        
        # Check for exact matches
        if resume_edu.intersection(required_edu):
            return 100.0
        
        # Check for equivalent qualifications
        education_hierarchy = {
            'phd': 4, 'doctorate': 4,
            'master': 3, 'mtech': 3, 'mca': 3, 'mba': 3, 'me': 3, 'msc': 3, 'ma': 3,
            'bachelor': 2, 'btech': 2, 'be': 2, 'bca': 2, 'bsc': 2, 'ba': 2,
            'diploma': 1
        }
        
        resume_level = max([education_hierarchy.get(edu, 0) for edu in resume_edu] or [0])
        required_level = max([education_hierarchy.get(edu, 0) for edu in required_edu] or [0])
        
        if resume_level >= required_level:
            return 90.0  # Higher or equal qualification
        elif resume_level == required_level - 1:
            return 70.0  # One level below
        else:
            return 40.0  # Significantly below requirement
    
    def _get_enhanced_verdict(self, final_score: float, hard_score: float, exp_score: float) -> str:
        """Enhanced verdict calculation considering multiple factors"""
        # Base verdict from score
        if final_score >= 80:
            base_verdict = "High"
        elif final_score >= 60:
            base_verdict = "Medium"
        else:
            base_verdict = "Low"
        
        # Adjust based on critical factors
        if hard_score < 30:  # Very few skill matches
            if base_verdict == "High":
                base_verdict = "Medium"
            elif base_verdict == "Medium":
                base_verdict = "Low"
        
        if exp_score < 40:  # Significantly under-experienced
            if base_verdict == "High":
                base_verdict = "Medium"
        
        return base_verdict
    
    def _find_missing_skills(self, resume_skills: List[str], required_skills: List[str]) -> List[str]:
        """Enhanced missing skills detection with prioritization"""
        resume_skills_lower = set(skill.lower() for skill in resume_skills)
        required_skills_lower = set(skill.lower() for skill in required_skills)
        
        missing = required_skills_lower - resume_skills_lower
        
        # Remove fuzzy matches from missing skills
        truly_missing = []
        for missing_skill in missing:
            is_fuzzy_match = False
            for resume_skill in resume_skills_lower:
                if self._skills_similar(missing_skill, resume_skill):
                    is_fuzzy_match = True
                    break
            if not is_fuzzy_match:
                truly_missing.append(missing_skill)
        
        return truly_missing[:10]  # Limit to top 10
    
    def _find_matching_skills(self, resume_skills: List[str], required_skills: List[str]) -> List[str]:
        """Enhanced matching skills detection including fuzzy matches"""
        resume_skills_lower = set(skill.lower() for skill in resume_skills)
        required_skills_lower = set(skill.lower() for skill in required_skills)
        
        # Exact matches
        exact_matches = list(resume_skills_lower.intersection(required_skills_lower))
        
        # Fuzzy matches
        fuzzy_matches = []
        for req_skill in required_skills_lower:
            if req_skill not in exact_matches:
                for res_skill in resume_skills_lower:
                    if self._skills_similar(req_skill, res_skill):
                        fuzzy_matches.append(f"{res_skill} (similar to {req_skill})")
                        break
        
        return exact_matches + fuzzy_matches
    
    def _generate_enhanced_suggestions(self, resume_info: Dict, jd_info: Dict, missing_skills: List[str], ai_analysis: Dict) -> List[str]:
        """Enhanced suggestion generation with AI insights"""
        suggestions = []
        
        # AI-powered suggestions first
        ai_suggestions = ai_analysis.get('ai_suggestions', [])
        suggestions.extend(ai_suggestions)
        
        # Skill gap suggestions
        if missing_skills:
            critical_skills = missing_skills[:3]  # Top 3 missing skills
            suggestions.append(f"🎯 Priority: Learn {', '.join(critical_skills)} to match job requirements")
        
        # Experience suggestions
        resume_exp = resume_info['experience_years']
        required_exp = jd_info['experience_required']
        if resume_exp < required_exp:
            gap = required_exp - resume_exp
            if gap <= 2:
                suggestions.append(f"💼 Gain {gap} more years of relevant experience through internships or projects")
            else:
                suggestions.append(f"💼 Consider gaining {gap} more years of experience or apply for junior positions")
        
        # Education suggestions
        if not set(resume_info['education']).intersection(set(jd_info['education_required'])) and jd_info['education_required']:
            suggestions.append(f"🎓 Consider pursuing: {', '.join(jd_info['education_required'][:2])}")
        
        # Project suggestions
        project_count = len(resume_info.get('projects', []))
        if project_count < 3:
            suggestions.append(f"🚀 Add more relevant projects (currently {project_count}, aim for 3-5)")
        
        # Certification suggestions
        cert_count = len(resume_info.get('certifications', []))
        if cert_count == 0 and jd_info['role_type'] in ['data_science', 'devops', 'cloud']:
            suggestions.append(f"📜 Consider getting industry certifications for {jd_info['role_type']} role")
        
        # Location suggestions
        if jd_info.get('location') != 'Remote' and resume_info.get('location') != jd_info.get('location'):
            suggestions.append(f"📍 Consider relocating to {jd_info.get('location')} or highlight remote work capability")
        
        return suggestions[:7]  # Limit to 7 suggestions

# Enhanced Database Manager with better schema and operations
class EnhancedDatabaseManager:
    """Enhanced database manager with improved schema and operations"""
    
    def __init__(self, db_path: str = "enhanced_resume_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced job descriptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS job_descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                company TEXT,
                content TEXT NOT NULL,
                location TEXT,
                salary_range TEXT,
                experience_required INTEGER DEFAULT 0,
                role_type TEXT,
                posted_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Enhanced resume evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resume_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER,
                candidate_name TEXT,
                candidate_email TEXT,
                candidate_phone TEXT,
                resume_file_name TEXT,
                relevance_score REAL,
                verdict TEXT,
                missing_skills TEXT,
                matching_skills TEXT,
                suggestions TEXT,
                strengths TEXT,
                improvement_areas TEXT,
                ai_feedback TEXT,
                location TEXT,
                projects_count INTEGER DEFAULT 0,
                certifications_count INTEGER DEFAULT 0,
                experience_years INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES job_descriptions (id)
            )
        """)
        
        # Analytics table for tracking system usage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Student feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id INTEGER,
                feedback_rating INTEGER,
                feedback_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (evaluation_id) REFERENCES resume_evaluations (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_job_description(self, title: str, company: str, content: str, location: str = "", salary_range: str = "", experience_required: int = 0, role_type: str = "") -> int:
        """Enhanced job description saving with additional fields"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO job_descriptions 
            (title, company, content, location, salary_range, experience_required, role_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (title, company, content, location, salary_range, experience_required, role_type))
        
        job_id = cursor.lastrowid
        
        # Log analytics
        cursor.execute("""
            INSERT INTO analytics_log (action_type, details)
            VALUES (?, ?)
        """, ("job_posted", f"Job: {title} at {company}"))
        
        conn.commit()
        conn.close()
        
        return job_id
    
    def save_evaluation(self, job_id: int, candidate_name: str, file_name: str, analysis: EnhancedResumeAnalysis):
        """Enhanced evaluation saving with comprehensive data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO resume_evaluations 
            (job_id, candidate_name, candidate_email, candidate_phone, resume_file_name, 
             relevance_score, verdict, missing_skills, matching_skills, suggestions,
             strengths, improvement_areas, ai_feedback, location, projects_count,
             certifications_count, experience_years)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id, candidate_name,
            analysis.contact_info.get('email', ''),
            analysis.contact_info.get('phone', ''),
            file_name, analysis.relevance_score, analysis.verdict,
            json.dumps(analysis.missing_skills),
            json.dumps(analysis.matching_skills),
            json.dumps(analysis.suggestions),
            json.dumps(analysis.strengths),
            json.dumps(analysis.improvement_areas),
            analysis.ai_feedback,
            analysis.location,
            analysis.projects_count,
            analysis.certifications_count,
            0  # experience_years - to be extracted
        ))
        
        # Log analytics
        cursor.execute("""
            INSERT INTO analytics_log (action_type, details)
            VALUES (?, ?)
        """, ("resume_evaluated", f"Candidate: {candidate_name}, Score: {analysis.relevance_score}"))
        
        conn.commit()
        conn.close()
    
    def get_job_descriptions(self, active_only: bool = True) -> List[Dict]:
        """Enhanced job description retrieval with filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT id, title, company, location, salary_range, experience_required, 
                   role_type, created_at, is_active
            FROM job_descriptions 
        """
        if active_only:
            query += "WHERE is_active = TRUE "
        query += "ORDER BY created_at DESC"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            "id": row[0], "title": row[1], "company": row[2], "location": row[3],
            "salary_range": row[4], "experience_required": row[5], "role_type": row[6],
            "created_at": row[7], "is_active": row[8]
        } for row in rows]
    
    def get_evaluations_for_job(self, job_id: int) -> List[Dict]:
        """Enhanced evaluation retrieval with comprehensive data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT candidate_name, candidate_email, candidate_phone, resume_file_name, 
                   relevance_score, verdict, missing_skills, matching_skills, suggestions,
                   strengths, improvement_areas, ai_feedback, location, projects_count,
                   certifications_count, experience_years, created_at
            FROM resume_evaluations 
            WHERE job_id = ? 
            ORDER BY relevance_score DESC
        """, (job_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        evaluations = []
        for row in rows:
            evaluations.append({
                "candidate_name": row[0],
                "candidate_email": row[1],
                "candidate_phone": row[2],
                "file_name": row[3],
                "relevance_score": row[4],
                "verdict": row[5],
                "missing_skills": json.loads(row[6]) if row[6] else [],
                "matching_skills": json.loads(row[7]) if row[7] else [],
                "suggestions": json.loads(row[8]) if row[8] else [],
                "strengths": json.loads(row[9]) if row[9] else [],
                "improvement_areas": json.loads(row[10]) if row[10] else [],
                "ai_feedback": row[11],
                "location": row[12],
                "projects_count": row[13],
                "certifications_count": row[14],
                "experience_years": row[15],
                "created_at": row[16]
            })
        
        return evaluations
    
    def get_analytics_data(self, days: int = 30) -> Dict:
        """Get system analytics data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Daily activity
        cursor.execute("""
            SELECT DATE(timestamp) as date, action_type, COUNT(*) as count
            FROM analytics_log 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp), action_type
            ORDER BY date
        """, (start_date.strftime('%Y-%m-%d'),))
        
        daily_activity = cursor.fetchall()
        
        # Total statistics
        cursor.execute("SELECT COUNT(*) FROM job_descriptions WHERE is_active = TRUE")
        total_jobs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM resume_evaluations")
        total_evaluations = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT AVG(relevance_score) FROM resume_evaluations 
            WHERE created_at >= ?
        """, (start_date.strftime('%Y-%m-%d'),))
        avg_score = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'daily_activity': daily_activity,
            'total_jobs': total_jobs,
            'total_evaluations': total_evaluations,
            'average_score': round(avg_score, 2)
        }
    
    def delete_job_description(self, job_id: int) -> bool:
        """Enhanced job deletion with soft delete option"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Soft delete - mark as inactive
            cursor.execute("UPDATE job_descriptions SET is_active = FALSE WHERE id = ?", (job_id,))
            
            # Log analytics
            cursor.execute("""
                INSERT INTO analytics_log (action_type, details)
                VALUES (?, ?)
            """, ("job_deleted", f"Job ID: {job_id}"))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            conn.rollback()
            conn.close()
            st.error(f"❌ Error deleting job: {e}")
            return False

# Initialize enhanced components
@st.cache_resource
def init_enhanced_components():
    """Initialize all enhanced system components"""
    components = {
        'resume_parser': EnhancedResumeParser(),
        'jd_parser': EnhancedJobDescriptionParser(),
        'scorer': EnhancedRelevanceScorer(),
        'db': EnhancedDatabaseManager(),
        'gemini_ai': GeminiAI()
    }
    
    # Test components
    try:
        jobs_count = len(components['db'].get_job_descriptions())
        st.sidebar.success(f"✅ System initialized - {jobs_count} active jobs")
    except Exception as e:
        st.sidebar.error(f"❌ System initialization error: {e}")
    
    return components

# Enhanced utility functions
def create_score_gauge(score: float) -> go.Figure:
    """Create a beautiful gauge chart for relevance score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Relevance Score"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_skills_comparison_chart(matching_skills: List[str], missing_skills: List[str]) -> go.Figure:
    """Create a comparison chart for skills"""
    categories = ['Matching Skills', 'Missing Skills']
    values = [len(matching_skills), len(missing_skills)]
    colors = ['#00CC96', '#FF6692']
    
    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Skills Analysis",
        yaxis_title="Count",
        height=300
    )
    
    return fig

def format_contact_info(contact_info: Dict) -> str:
    """Format contact information for display"""
    formatted = []
    if contact_info.get('email'):
        formatted.append(f"📧 {contact_info['email']}")
    if contact_info.get('phone'):
        formatted.append(f"📱 {contact_info['phone']}")
    if contact_info.get('linkedin'):
        formatted.append(f"💼 {contact_info['linkedin']}")
    if contact_info.get('github'):
        formatted.append(f"🐙 {contact_info['github']}")
    
    return " • ".join(formatted) if formatted else "Contact information not available"

# Enhanced main application
def enhanced_main():
    """Enhanced main application with beautiful UI and improved functionality"""
    
    # Page configuration
    st.set_page_config(
        page_title="🚀 Innomatics Resume Relevance System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://innomatics.in',
            'Report a bug': "mailto:support@innomatics.in",
            'About': "# Innomatics Research Labs\n## AI-Powered Resume Evaluation System"
        }
    )
    
    # Enhanced CSS with beautiful styling and animations
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Custom CSS Variables */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #00d4aa;
        --warning-color: #f093fb;
        --error-color: #ff6b6b;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --card-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        --card-hover-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        --border-radius: 20px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Global Styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--background-gradient);
        min-height: 100vh;
    }
    
    /* Main container with glass morphism effect */
    .main .block-container {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
        margin: 1rem;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: slideInUp 0.6s ease-out;
    }
    
    /* Header styling with gradient text */
    h1 {
    color: white !important;
    font-weight: 800;
    text-align: center;
    font-size: 3rem !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 0 4px 8px rgba(0,0,0,0.5);
    animation: fadeInDown 0.8s ease-out;
    }    
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        border-bottom: 3px solid transparent;
        border-image: var(--background-gradient) 1;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem !important;
        animation: slideInLeft 0.6s ease-out;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 500;
        margin-bottom: 1rem !important;
    }
    
    /* Enhanced sidebar with gradient background */
    .css-1d391kg {
        background: var(--background-gradient);
        border-radius: 0 var(--border-radius) var(--border-radius) 0;
        backdrop-filter: blur(10px);
    }
    
    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stMetric label,
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    .css-1d391kg .stMarkdown {
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Enhanced metric cards with hover effects */
    [data-testid="metric-container"] {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: var(--card-shadow);
        text-align: center;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    [data-testid="metric-container"]:hover::before {
        left: 100%;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: var(--card-hover-shadow);
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 2.5rem;
        font-weight: 800;
        background: var(--background-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Enhanced buttons with gradient and hover effects */
    .stButton > button {
        background: var(--background-gradient);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: var(--transition);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transition: all 0.6s;
        transform: translate(-50%, -50%);
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Enhanced form controls */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        background: #f8fafc;
        transition: var(--transition);
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        transform: translateY(-2px);
    }
    
    /* Enhanced file uploader with drag-drop styling */
    .stFileUploader {
        background: linear-gradient(45deg, #f8fafc, #e2e8f0);
        border: 3px dashed #cbd5e0;
        border-radius: var(--border-radius);
        padding: 3rem 2rem;
        text-align: center;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .stFileUploader::before {
        content: '📁';
        font-size: 3rem;
        display: block;
        margin-bottom: 1rem;
        opacity: 0.7;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-color);
        background: linear-gradient(45deg, #f0f4ff, #e6f3ff);
        transform: scale(1.02);
        box-shadow: var(--card-shadow);
    }
    
    /* Enhanced cards with glassmorphism */
    .stExpander {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px) !important;
        margin: 1rem 0 !important;
        transition: var(--transition) !important;
    }
    
    .stExpander:hover {
        transform: translateY(-2px);
        box-shadow: var(--card-hover-shadow);
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)) !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        border: none !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(248, 250, 252, 0.8) !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.5rem !important;
        border: none !important;
    }
    
    /* Enhanced progress bar */
    .stProgress > div > div > div > div {
        background: var(--background-gradient) !important;
        border-radius: 10px !important;
        height: 12px !important;
        position: relative;
        overflow: hidden;
    }
    
    .stProgress > div > div > div > div::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        animation: shimmer 2s infinite;
    }
    
    /* Enhanced info/success/warning/error boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 12px !important;
        border: none !important;
        padding: 1rem 1.5rem !important;
        font-weight: 500 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(147, 197, 253, 0.2)) !important;
        border-left: 4px solid #3b82f6 !important;
        color: #1e40af !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(110, 231, 183, 0.2)) !important;
        border-left: 4px solid #10b981 !important;
        color: #047857 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(251, 191, 36, 0.2)) !important;
        border-left: 4px solid #f59e0b !important;
        color: #92400e !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(252, 165, 165, 0.2)) !important;
        border-left: 4px solid #ef4444 !important;
        color: #dc2626 !important;
    }
    
    /* Enhanced DataFrame styling */
    .stDataFrame {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: var(--card-shadow) !important;
        border: none !important;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(248, 250, 252, 0.8);
        padding: 0.5rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: var(--transition);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.9);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--background-gradient) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced slider */
    .stSlider > div > div > div > div {
        background: var(--background-gradient) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }
    
    /* Animations */
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes shimmer {
        0% {
            left: -100%;
        }
        100% {
            left: 100%;
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(241, 245, 249, 0.8);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--background-gradient);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: center;
        margin: 0.25rem;
    }
    
    .status-high {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        color: #155724;
        border: 2px solid #28a745;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.3);
    }
    
    .status-medium {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        color: #856404;
        border: 2px solid #ffc107;
        box-shadow: 0 2px 8px rgba(255, 193, 7, 0.3);
    }
    
    .status-low {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        color: #721c24;
        border: 2px solid #dc3545;
        box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
    }
    
    /* Loading animation */
    .stSpinner > div {
        border-color: var(--primary-color) !important;
    }
    
    /* Enhanced tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
        text-align: center;
        border-radius: 8px;
        padding: 0.5rem;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
            margin: 0.5rem;
        }
        
        h1 {
            font-size: 2rem !important;
        }
        
        [data-testid="metric-container"] {
            padding: 1rem;
        }
        
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
        }
    }
    
    /* Print styles */
    @media print {
        .stApp {
            background: white !important;
        }
        
        .main .block-container {
            box-shadow: none !important;
            border: 1px solid #ccc !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with enhanced styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🚀 Innomatics Resume Relevance System</h1>
        <p style="font-size: 1.2rem; color: #64748b; margin-top: 0;">
            <strong>AI-Powered Resume Evaluation Platform</strong><br>
            <span style="font-size: 1rem;">Hyderabad • Bangalore • Pune • Delhi NCR</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize components
    components = init_enhanced_components()
    
    # Enhanced sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        st.markdown("---")
        
        page = st.selectbox(
            "Choose a page:",
            [
                "🏠 Home Dashboard",
                "📝 Upload Job Description", 
                "📋 Evaluate Resumes", 
                "📊 Placement Dashboard",
                "📈 Advanced Analytics",
                "📤 Export & Reports",
                "🛠️ Job Management",
                "🤖 AI Assistant",
                "👥 Student Feedback",
                "⚙️ System Settings"
            ]
        )
        
        # Enhanced system status
        st.markdown("---")
        st.markdown("## 📊 System Status")
        
        try:
            analytics = components['db'].get_analytics_data(7)  # Last 7 days
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Jobs", analytics['total_jobs'], delta=None)
            with col2:
                st.metric("Total Evaluations", analytics['total_evaluations'])
            
            if analytics['average_score'] > 0:
                st.metric("Avg Score (7d)", f"{analytics['average_score']:.1f}%", 
                         delta=f"{analytics['average_score']-70:.1f}%")
            
            # Status indicator
            if analytics['total_evaluations'] > 0:
                st.success("✅ System Active")
            else:
                st.info("🔄 Ready for Use")
                
        except Exception as e:
            st.error(f"❌ System Error: {e}")
        
        # Quick actions
        st.markdown("---")
        st.markdown("## ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Data", key="refresh_data"):
            st.cache_resource.clear()
            st.rerun()
        
        if st.button("📊 View Analytics", key="quick_analytics"):
            st.session_state['page'] = "📈 Advanced Analytics"
            st.rerun()
        
        # AI Status
        st.markdown("---")
        st.markdown("## 🤖 AI Status")
        
        if components['gemini_ai'].enabled:
            st.success("✅ Gemini AI Active")
        else:
            st.warning("⚠️ AI Features Limited")
            with st.expander("Enable AI Features"):
                st.markdown("""
                **Get free Gemini API key:**
                1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
                2. Create free API key
                3. Set environment variable: `GEMINI_API_KEY=your_key`
                """)
    
    # Route to appropriate page
    from enhanced_pages import (
    enhanced_upload_job_page,
    enhanced_evaluate_resumes_page, 
    student_feedback_page,
    enhanced_settings_page
    )
    if page == "🏠 Home Dashboard":
        from enhanced_pages import enhanced_home_page
        enhanced_home_page(components)
        
    elif page == "📝 Upload Job Description":
        from enhanced_pages import enhanced_upload_job_page
        enhanced_upload_job_page(components)
    elif page == "📋 Evaluate Resumes":
        from enhanced_pages import enhanced_evaluate_resumes_page
        enhanced_evaluate_resumes_page(components)
    elif page == "📊 Placement Dashboard":
        from enhanced_pages import enhanced_dashboard_page
        enhanced_dashboard_page(components)
    elif page == "📈 Advanced Analytics":
        from enhanced_pages import enhanced_analytics_page
        enhanced_analytics_page(components)
    elif page == "📤 Export & Reports":
        from enhanced_pages import enhanced_export_page
        enhanced_export_page(components)
    elif page == "👥 Student Feedback":
        from enhanced_pages import student_feedback_page
        student_feedback_page(components)
    elif page == "⚙️ System Settings":
        from enhanced_pages import enhanced_settings_page
        enhanced_settings_page(components)
    elif page == "🛠️ Job Management":
        from enhanced_pages import job_management_page
        job_management_page(components)
    elif page == "🤖 AI Assistant":
        from enhanced_pages import ai_chatbot_page
        ai_chatbot_page(components)    
# Enhanced page implementations will continue in the next part...

# Enhanced Home Page
def enhanced_home_page(components):
    """Enhanced home page with comprehensive dashboard"""
    
    st.markdown("## 🏠 Welcome to Innomatics Resume System")
    
    # Get comprehensive analytics
    analytics = components['db'].get_analytics_data(30)  # Last 30 days
    jobs = components['db'].get_job_descriptions()
    
    # Calculate additional metrics
    total_jobs = len(jobs)
    recent_evaluations = []
    high_candidates = 0
    
    for job in jobs[-5:]:  # Last 5 jobs
        evaluations = components['db'].get_evaluations_for_job(job['id'])
        recent_evaluations.extend(evaluations[-10:])  # Last 10 evaluations per job
        high_candidates += len([e for e in evaluations if e['verdict'] == 'High'])
    
    # Enhanced metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📋 Active Jobs", 
            total_jobs,
            delta=f"+{len([j for j in jobs if (datetime.now() - datetime.strptime(j['created_at'][:10], '%Y-%m-%d')).days <= 7])}" if jobs else None,
            help="Total number of active job postings"
        )
    
    with col2:
        st.metric(
            "📊 Total Evaluations", 
            analytics['total_evaluations'],
            delta=f"+{len([e for e in recent_evaluations if (datetime.now() - datetime.strptime(e['created_at'][:10], '%Y-%m-%d')).days <= 7])}" if recent_evaluations else None,
            help="Total resumes evaluated across all jobs"
        )
    
    with col3:
        st.metric(
            "🌟 High Suitability", 
            high_candidates,
            delta=f"{(high_candidates/len(recent_evaluations)*100):.1f}% of total" if recent_evaluations else "0%",
            help="Candidates with high suitability rating"
        )
    
    with col4:
        avg_score = analytics['average_score'] if analytics['average_score'] > 0 else (
            sum(e['relevance_score'] for e in recent_evaluations) / len(recent_evaluations) if recent_evaluations else 0
        )
        st.metric(
            "📈 Average Score", 
            f"{avg_score:.1f}%",
            delta=f"{avg_score-70:.1f}% from target" if avg_score > 0 else None,
            help="Average relevance score across all evaluations"
        )
    
    # Visual analytics section
    if recent_evaluations:
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Score Distribution")
            
            # Create score distribution chart
            scores = [e['relevance_score'] for e in recent_evaluations]
            bins = [0, 25, 50, 75, 100]
            labels = ['Poor (0-25)', 'Fair (26-50)', 'Good (51-75)', 'Excellent (76-100)']
            
            score_counts = []
            for i in range(len(bins)-1):
                count = len([s for s in scores if bins[i] < s <= bins[i+1]])
                score_counts.append(count)
            
            fig = px.bar(
                x=labels, 
                y=score_counts, 
                title="Resume Score Distribution",
                color=score_counts,
                color_continuous_scale="viridis"
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Verdict Summary")
            
            # Verdict distribution
            verdict_counts = {}
            for eval in recent_evaluations:
                verdict = eval['verdict']
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            
            if verdict_counts:
                fig = px.pie(
                    values=list(verdict_counts.values()),
                    names=list(verdict_counts.keys()),
                    title="Candidate Suitability",
                    color_discrete_map={
                        'High': '#00CC96',
                        'Medium': '#FFA15A',
                        'Low': '#FF6692'
                    }
                )
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
    
    # Quick start guide with enhanced styling
    st.markdown("---")
    st.markdown("## 🚀 Quick Start Guide")
    
    tab1, tab2, tab3 = st.tabs(["📝 For HR Team", "🎯 System Features", "📈 Best Practices"])
    
    with tab1:
        st.markdown("""
        ### Step-by-Step Process
        
        1. **📝 Upload Job Description**
           - Navigate to "Upload Job Description"
           - Paste complete job requirements
           - AI extracts key skills and requirements automatically
        
        2. **📋 Evaluate Resumes**
           - Go to "Evaluate Resumes"
           - Select the relevant job posting
           - Upload multiple resume files (PDF/DOCX supported)
           - Get instant AI-powered relevance scores
        
        3. **📊 Review Results**
           - Check "Placement Dashboard" for comprehensive results
           - Filter candidates by score, location, or skills
           - Export shortlists for hiring managers
           - View detailed candidate analysis
        
        4. **📈 Track Performance**
           - Monitor system analytics and trends
           - Identify skill gaps across candidate pool
           - Generate hiring reports and insights
        """)
    
    with tab2:
        st.markdown("""
        ### 🎯 Advanced AI Features
        
        - **🤖 AI-Powered Analysis**: Advanced NLP and machine learning
        - **📄 Multi-format Support**: PDF and DOCX resume parsing
        - **⚡ Batch Processing**: Handle hundreds of resumes simultaneously
        - **🔍 Semantic Matching**: Beyond keyword matching using AI
        - **📊 Detailed Scoring**: 
          - Hard Match (30%): Exact skill matches
          - Semantic Match (30%): Content similarity
          - Experience Match (20%): Experience alignment
          - Education Match (10%): Qualification matching
          - AI Enhancement (10%): Advanced insights
        - **💡 Actionable Insights**: Specific improvement suggestions
        - **📤 Export Capabilities**: CSV, reports, and shortlists
        - **🌍 Location Filtering**: Multi-city support (Hyderabad, Bangalore, Pune, Delhi NCR)
        """)
    
    with tab3:
        st.markdown("""
        ### 📈 Best Practices for Optimal Results
        
        **Job Descriptions:**
        - ✅ Include specific technology names and versions
        - ✅ Clearly separate must-have vs nice-to-have skills
        - ✅ Mention required years of experience
        - ✅ Specify education requirements
        - ✅ Use industry-standard terminology
        
        **Resume Evaluation:**
        - ✅ Ensure files are text-searchable (not scanned images)
        - ✅ Process resumes in batches for efficiency
        - ✅ Review both scores and detailed analysis
        - ✅ Use missing skills analysis for candidate development
        - ✅ Consider location preferences and remote options
        
        **Score Interpretation:**
        - 🟢 **High (75-100)**: Strong match, recommend for interview
        - 🟡 **Medium (50-74)**: Potential fit, consider for alternative roles
        - 🔴 **Low (0-49)**: Skills gap too large, recommend training
        """)
    
    # Recent activity section
    if recent_evaluations:
        st.markdown("---")
        st.markdown("## 📈 Recent Activity")
        
        # Create recent activity dataframe
        recent_df = pd.DataFrame([{
            'Date': eval['created_at'][:10],
            'Candidate': eval['candidate_name'][:25] + '...' if len(eval['candidate_name']) > 25 else eval['candidate_name'],
            'Score': f"{eval['relevance_score']:.1f}%",
            'Verdict': eval['verdict'],
            'Location': eval.get('location', 'N/A')
        } for eval in recent_evaluations[-15:]])
        
        # Apply styling to the dataframe
        def color_verdict(val):
            colors = {
                'High': 'background-color: #d4edda; color: #155724; font-weight: bold;',
                'Medium': 'background-color: #fff3cd; color: #856404; font-weight: bold;',
                'Low': 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
            }
            return colors.get(val, '')
        
        styled_df = recent_df.style.applymap(color_verdict, subset=['Verdict'])
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Call-to-action section
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Upload New Job", type="primary", use_container_width=True):
            st.switch_page("📝 Upload Job Description")
    
    with col2:
        if st.button("📋 Evaluate Resumes", type="secondary", use_container_width=True):
            st.switch_page("📋 Evaluate Resumes")
    
    with col3:
        if st.button("📊 View Dashboard", type="secondary", use_container_width=True):
            st.switch_page("📊 Placement Dashboard")

# The implementation continues with other enhanced pages...
# Due to length constraints, I'll provide the key enhanced pages

if __name__ == "__main__":
    enhanced_main()