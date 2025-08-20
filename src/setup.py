#!/usr/bin/env python3
"""
Setup script for GDPR Compliance Assistant
Automates the initialization process for the application
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"✅ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")

def install_requirements():
    """Install required packages"""
    try:
        logger.info("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✅ All packages installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install packages: {e}")
        sys.exit(1)

def check_data_files():
    """Check if required data files exist"""
    required_files = ["gdpr_text.csv", "gdpr_violations.csv"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"⚠️ Missing data files: {', '.join(missing_files)}")
        logger.info("Please ensure the following files are in the project directory:")
        for file in missing_files:
            logger.info(f"  - {file}")
        return False
    else:
        logger.info("✅ All required data files found")
        return True

def create_environment_file():
    """Create .env file with API keys"""
    env_content = """# aiXplain API Configuration
AIXPLAIN_API_KEY=869fad7c13c88be607658f5967ad5586210f53a436081ef092e21132cf1b063d

# Model IDs from aiXplain Platform
TEXT_EMBEDDING_MODEL_ID=673248d66eb563b2b00f75d1
GPT4_MINI_MODEL_ID=67fd9ddfef0365783d06e2ef
SCRAPE_WEBSITE_TOOL_ID=66f423426eb563fa213a3531

# Optional: OpenAI API Key (if using OpenAI models)
# OPENAI_API_KEY=your_openai_api_key_here
"""
    
    if not Path(".env").exists():
        with open(".env", "w") as f:
            f.write(env_content)
        logger.info("✅ Created .env file with API keys")
    else:
        logger.info("✅ .env file already exists")

def generate_embeddings():
    """Generate embeddings for the knowledge base"""
    try:
        logger.info("🔄 Generating embeddings for knowledge base...")
        from data_processor import GDPRDataProcessor
        
        processor = GDPRDataProcessor()
        processor.process_and_save_all()
        
        logger.info("✅ Embeddings generated successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to generate embeddings: {e}")
        logger.info("You can generate embeddings later by running: python data_processor.py")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Setting up GDPR Compliance Assistant...")
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Create environment file
    create_environment_file()
    
    # Check data files
    data_files_exist = check_data_files()
    
    # Generate embeddings if data files exist
    if data_files_exist:
        embeddings_generated = generate_embeddings()
    else:
        embeddings_generated = False
        logger.warning("⚠️ Skipping embeddings generation due to missing data files")
    
    # Setup summary
    logger.info("\n" + "="*50)
    logger.info("🎉 Setup completed!")
    logger.info("="*50)
    
    logger.info("\n📋 Setup Summary:")
    logger.info("✅ Python version checked")
    logger.info("✅ Packages installed")
    logger.info("✅ Environment file created")
    logger.info(f"{'✅' if data_files_exist else '⚠️'} Data files {'found' if data_files_exist else 'missing'}")
    logger.info(f"{'✅' if embeddings_generated else '⚠️'} Embeddings {'generated' if embeddings_generated else 'skipped'}")
    
    logger.info("\n🚀 Next Steps:")
    if not data_files_exist:
        logger.info("1. Add the required data files (gdpr_text.csv, gdpr_violations.csv)")
        logger.info("2. Run: python data_processor.py")
        logger.info("3. Start the app: streamlit run app.py")
    else:
        logger.info("1. Start the application: streamlit run app.py")
        logger.info("2. Open your browser to: http://localhost:8501")
        logger.info("3. Initialize the agent using the sidebar")
    
    logger.info("\n📚 Documentation:")
    logger.info("- Full documentation: README.md")
    logger.info("- Troubleshooting: Check README.md for common issues")
    logger.info("- Support: Create an issue on the repository")

if __name__ == "__main__":
    main()
