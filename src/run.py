#!/usr/bin/env python3
"""
Quick start script for GDPR Compliance Assistant
"""
import subprocess
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_missing_packages():
    """Install missing packages"""
    packages_to_install = []
    
    # Check for required packages
    try:
        import streamlit
    except ImportError:
        packages_to_install.append("streamlit")
    
    try:
        import pandas
    except ImportError:
        packages_to_install.append("pandas")
    
    try:
        import faiss
    except ImportError:
        packages_to_install.append("faiss-cpu")
    
    try:
        import sklearn
    except ImportError:
        packages_to_install.append("scikit-learn")
    
    try:
        import plotly
    except ImportError:
        packages_to_install.append("plotly")
    
    try:
        import aixplain
    except ImportError:
        packages_to_install.append("aixplain")
    
    if packages_to_install:
        logger.info(f"📦 Installing missing packages: {', '.join(packages_to_install)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)
            logger.info("✅ All packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install packages: {e}")
            return False
    else:
        logger.info("✅ All required packages are already installed")
        return True

def check_setup():
    """Check if the application is properly set up"""
    issues = []
    
    # Check if data files exist
    if not Path("gdpr_text.csv").exists():
        issues.append("Missing gdpr_text.csv")
    if not Path("gdpr_violations.csv").exists():
        issues.append("Missing gdpr_violations.csv")
    
    # Check if embeddings exist
    if not Path("embeddings").exists():
        issues.append("Embeddings not generated")
    
    return issues

def run_setup():
    """Run the setup script"""
    logger.info("🔧 Running setup...")
    try:
        subprocess.check_call([sys.executable, "setup.py"])
        return True
    except subprocess.CalledProcessError:
        return False

def start_streamlit():
    """Start the Streamlit application"""
    logger.info("🚀 Starting GDPR Compliance Assistant...")
    logger.info("📱 The application will open in your default browser")
    logger.info("🌐 URL: http://localhost:8501")
    logger.info("⏹️ Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        logger.info("\n👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting application: {e}")

def main():
    """Main function"""
    logger.info("🔒 GDPR Compliance Assistant - Quick Start")
    logger.info("="*50)
    
    # First, install missing packages
    if not install_missing_packages():
        logger.error("❌ Failed to install required packages")
        sys.exit(1)
    
    # Check setup status
    issues = check_setup()
    
    if issues:
        logger.warning("⚠️ Setup issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        
        # Auto-generate embeddings if missing
        if "Embeddings not generated" in issues:
            logger.info("🔄 Generating embeddings...")
            try:
                subprocess.check_call([sys.executable, "data_processor.py"])
                logger.info("✅ Embeddings generated successfully!")
            except subprocess.CalledProcessError:
                logger.error("❌ Failed to generate embeddings")
                sys.exit(1)
    else:
        logger.info("✅ Application is ready to start!")
    
    # Start the application
    start_streamlit()

if __name__ == "__main__":
    from config import AIXPLAIN_API_KEY
    os.environ['AIXPLAIN_API_KEY'] = AIXPLAIN_API_KEY
    os.environ['TEAM_API_KEY'] = AIXPLAIN_API_KEY
    main()
    
