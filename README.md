# 🔒 GDPR Compliance Assistant

A comprehensive AI-powered platform for GDPR compliance analysis, privacy policy evaluation, and regulatory guidance. Built for the aiXplain Platform Certification using advanced RAG (Retrieval-Augmented Generation) technology.

## ✨ Features

### 🤖 AI-Powered Assistant
- **Intelligent GDPR Q&A**: Get expert answers to GDPR compliance questions
- **Real-time Chat Interface**: Interactive conversation with specialized GDPR knowledge
- **Context-Aware Responses**: Answers based on comprehensive GDPR documentation and case studies

### 🌐 Website Analysis
- **Privacy Policy Scraping**: Automatically extract and analyze privacy policies from websites
- **GDPR Compliance Assessment**: Detailed evaluation of privacy policies against GDPR requirements
- **Violation Risk Analysis**: Identify potential compliance gaps and risks

### 📊 Advanced Analytics
- **Violation Case Studies**: Comprehensive database of real GDPR violation cases
- **Fine Analysis**: Statistical analysis of GDPR penalties across countries and violation types
- **Interactive Visualizations**: Charts and graphs showing compliance trends and patterns

### 📚 Knowledge Base
- **Comprehensive GDPR Database**: Complete collection of GDPR articles and regulations
- **Semantic Search**: Find relevant information using natural language queries
- **Vector-Based Retrieval**: Advanced similarity search using FAISS and embeddings

### 🚀 RAG System
- **Embeddings Generation**: Convert GDPR documents into vector representations
- **FAISS Integration**: High-performance similarity search and retrieval
- **Upload Functionality**: Custom embeddings upload for enhanced performance

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  aiXplain Agent  │────│  RAG System     │
│                 │    │                  │    │                 │
│ • Chat Interface│    │ • Web Scraper    │    │ • FAISS Index   │
│ • Analytics     │    │ • GPT-4 Mini     │    │ • Embeddings    │
│ • File Upload   │    │ • Model Tools    │    │ • Similarity    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │   Knowledge Base    │
                    │                     │
                    │ • GDPR Articles     │
                    │ • Violation Cases   │
                    │ • Legal Precedents  │
                    └─────────────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- aiXplain Platform Account
- Required API keys (see configuration section)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd gdpr-compliance-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root with your API keys:

```env
AIXPLAIN_API_KEY=869fad7c13c88be607658f5967ad5586210f53a436081ef092e21132cf1b063d
TEXT_EMBEDDING_MODEL_ID=673248d66eb563b2b00f75d1
GPT4_MINI_MODEL_ID=67fd9ddfef0365783d06e2ef
SCRAPE_WEBSITE_TOOL_ID=66f423426eb563fa213a3531
```

### 3. Data Preparation

Ensure your GDPR data files are in the project directory:
- `gdpr_text.csv` - Complete GDPR articles and regulations
- `gdpr_violations.csv` - Historical GDPR violation cases

### 4. Initialize Embeddings

```bash
# Generate embeddings for the knowledge base
python data_processor.py
```

### 5. Launch Application

```bash
# Start the Streamlit application
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📱 Usage Guide

### 🤖 AI Assistant Tab
1. **Initialize Agent**: Click "Initialize Agent" in the sidebar
2. **Ask Questions**: Use the chat interface to ask GDPR-related questions
3. **Get Expert Answers**: Receive detailed responses with article references

Example questions:
- "What are the data subject rights under GDPR?"
- "How should we handle a data breach notification?"
- "What is the lawful basis for processing employee data?"

### 🌐 Website Analysis Tab
1. **Enter URL**: Input the privacy policy URL you want to analyze
2. **Click Analyze**: The agent will scrape and analyze the content
3. **Review Results**: Get detailed compliance assessment and recommendations

### 📊 Analytics Tab
- View comprehensive statistics of GDPR violations
- Explore interactive charts and visualizations
- Analyze trends by country, violation type, and fine amounts

### 📚 Knowledge Base Tab
- Search through the complete GDPR database
- Find relevant articles and violation cases
- Use natural language queries for semantic search

### 📤 Upload Embeddings
1. **Prepare Embeddings**: Generate custom embeddings using your data
2. **Upload Files**: Use the sidebar to upload `.npy` or `.pkl` files
3. **Enhance Performance**: Improve search accuracy with custom embeddings

## 🔧 Technical Details

### aiXplain Agent Configuration

The GDPR Compliance Agent is configured with:

```python
agent = AgentFactory.create(
    name="GDPR Compliance Assistant",
    description="Expert GDPR Compliance Assistant specialized in European data protection regulations...",
    instructions="You are an expert GDPR Compliance Assistant with deep knowledge...",
    tools=[scrape_tool, analysis_tool],
    llm=GPT4_MINI_MODEL_ID
)
```

### Tools Integration

1. **Web Scraping Tool** (`66f423426eb563fa213a3531`)
   - Extracts content from privacy policy pages
   - Handles various website formats and structures

2. **GPT-4 Mini Analysis Tool** (`67fd9ddfef0365783d06e2ef`)
   - Performs detailed legal analysis
   - Generates compliance reports and recommendations

3. **Text Embedding Model** (`673248d66eb563b2b00f75d1`)
   - Creates vector representations of documents
   - Enables semantic search capabilities

### RAG System Components

1. **Data Processing**
   - Loads GDPR text and violation data
   - Generates embeddings using SentenceTransformers
   - Creates comprehensive metadata for each document

2. **Vector Database**
   - FAISS index for high-performance similarity search
   - Supports both flat and IVF indexing strategies
   - Optimized for different dataset sizes

3. **Retrieval System**
   - Semantic search using cosine similarity
   - Contextual ranking of results
   - Integration with agent responses

## 📊 Data Sources

### GDPR Text Dataset (`gdpr_text.csv`)
- **Columns**: chapter, chapter_title, article, article_title, sub_article, gdpr_text, href
- **Content**: Complete GDPR articles with hierarchical structure
- **Usage**: Legal reference and compliance guidance

### GDPR Violations Dataset (`gdpr_violations.csv`)
- **Columns**: id, picture, name, price, authority, date, controller, article_violated, type, source, summary
- **Content**: Real-world GDPR violation cases with penalties
- **Usage**: Risk assessment and precedent analysis

## 🛡️ Security & Privacy

- **API Key Management**: Secure storage of sensitive credentials
- **Data Processing**: Local processing of uploaded documents
- **Privacy Protection**: No personal data stored or transmitted
- **Compliance**: Built with GDPR principles in mind

## 🔍 Advanced Features

### Custom Embeddings
- Upload pre-computed embeddings for specific domains
- Support for different embedding models and dimensions
- Integration with existing vector databases

### Intelligent Filtering
- Automatic detection of non-GDPR related queries
- Graceful handling of out-of-scope questions
- Professional boundary enforcement

### Performance Optimization
- Caching of agent initialization
- Efficient vector search algorithms
- Optimized data loading and processing

## 🚨 Troubleshooting

### Common Issues

1. **Agent Initialization Failed**
   - Check API keys in configuration
   - Verify internet connection
   - Ensure aiXplain account is active

2. **Embeddings Not Loading**
   - Run `python data_processor.py` to generate embeddings
   - Check data files are present and readable
   - Verify sufficient disk space

3. **Search Results Empty**
   - Initialize embeddings first
   - Try different search terms
   - Check if data was loaded correctly

### Performance Issues

1. **Slow Response Times**
   - Use smaller batch sizes for embedding generation
   - Consider upgrading to IVF index for large datasets
   - Monitor API rate limits

2. **Memory Issues**
   - Reduce embedding dimensions if needed
   - Process data in smaller chunks
   - Use efficient data types

## 📈 Future Enhancements

- [ ] Multi-language support for international GDPR applications
- [ ] Integration with additional legal databases and sources
- [ ] Advanced visualization dashboards for compliance tracking
- [ ] Automated compliance scoring and risk assessment
- [ ] Export functionality for compliance reports
- [ ] Integration with document management systems

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **aiXplain Platform** for providing the AI agent framework
- **Streamlit** for the excellent web application framework
- **GDPR.eu** and regulatory authorities for comprehensive GDPR data
- **FAISS** team for high-performance vector search capabilities

## 📞 Support

For questions, issues, or feature requests:
- Create an issue in the repository
- Contact the development team
- Refer to the aiXplain platform documentation

---

**Built for aiXplain Platform Certification** 🚀

*Empowering organizations with AI-driven GDPR compliance solutions*
