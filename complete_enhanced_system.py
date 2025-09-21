#!/usr/bin/env python3
"""
Enhanced Automated Resume Relevance Check System
Complete implementation for Innomatics Research Labs Hackathon
Version 2.0 - Beautiful UI, Bug Fixes, and Enhanced Features

This is the complete system combining all enhanced features and pages.
Run this file to start the application.
"""

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main enhanced system
from enhanced_resume_system import enhanced_main

# Import enhanced pages
from enhanced_pages import (
    enhanced_upload_job_page,
    enhanced_evaluate_resumes_page,
    student_feedback_page,
    enhanced_settings_page
)

def main():
    """Main entry point for the enhanced system"""
    try:
        enhanced_main()
    except Exception as e:
        st.error(f"🚨 System Error: {e}")
        st.info("💡 Please refresh the page or contact support if the issue persists.")
        
        # Show error details in expander for debugging
        with st.expander("🔍 Error Details (For Developers)"):
            st.code(f"""
Error Type: {type(e).__name__}
Error Message: {str(e)}
            """)

if __name__ == "__main__":
    main()