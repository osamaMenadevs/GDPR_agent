import os
import logging
import pandas as pd
from typing import Dict, List, Any
from aixplain.factories import AgentFactory, ModelFactory
from aixplain.modules.agent.tool.model_tool import ModelTool
from data_processor import GDPRDataProcessor
from config import (
    AIXPLAIN_API_KEY, 
    TEXT_EMBEDDING_MODEL_ID, 
    GPT4_MINI_MODEL_ID, 
    SCRAPE_WEBSITE_TOOL_ID,
    AGENT_DESCRIPTION,
    AGENT_INSTRUCTIONS
)

# Set aiXplain API key
os.environ['AIXPLAIN_API_KEY'] = AIXPLAIN_API_KEY
os.environ['TEAM_API_KEY'] = AIXPLAIN_API_KEY  # aiXplain also expects TEAM_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GDPRComplianceAgent:
    """
    GDPR Compliance Agent using aiXplain framework with RAG capabilities
    """
    
    def __init__(self):
        """Initialize the GDPR Compliance Agent"""
        self.agent = None
        self.data_processor = GDPRDataProcessor()
        self.setup_agent()
        
    def setup_agent(self):
        """
        Create and configure the aiXplain agent with web scraping and analysis capabilities
        """
        try:
            logger.info("Setting up GDPR Compliance Agent...")
            
            # Create web scraping tool
            scrape_tool = AgentFactory.create_model_tool(
                model=SCRAPE_WEBSITE_TOOL_ID,
                description="""
                Web scraping tool for analyzing website privacy policies and terms of service.
                The input query must be in the format: 'text': 'https://website.com/privacy-policy'
                This tool will extract and return the complete text content from privacy policy pages,
                terms of service, and other compliance-related web pages for GDPR analysis.
                """
            )
            
            # Create GPT-4 Mini analysis tool
            analysis_tool = AgentFactory.create_model_tool(
                model=GPT4_MINI_MODEL_ID,
                description="""
                Advanced language model for detailed GDPR compliance analysis and interpretation.
                Use this tool for complex legal analysis, policy interpretation, and generating
                comprehensive compliance reports. Input should be specific questions or requests
                for detailed GDPR analysis.
                """
            )
            
            # Create the main GDPR compliance agent
            self.agent = AgentFactory.create(
                name="GDPR Compliance Assistant",
                description=AGENT_DESCRIPTION,
                instructions=AGENT_INSTRUCTIONS,
                tools=[scrape_tool, analysis_tool],
                llm=GPT4_MINI_MODEL_ID  # Use GPT-4 Mini as the main LLM
            )
            
            logger.info(f"GDPR Compliance Agent created successfully!")
            logger.info(f"Agent ID: {self.agent.id}")
            logger.info(f"Agent Name: {self.agent.name}")
            
        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            raise
    
    def load_knowledge_base(self):
        """
        Load or create the GDPR knowledge base with embeddings
        """
        try:
            logger.info("Loading GDPR knowledge base...")
            
            # Try to load existing embeddings
            if os.path.exists("embeddings"):
                logger.info("Loading existing embeddings...")
                self.data_processor.load_embeddings_and_index()
            else:
                logger.info("Creating new embeddings...")
                self.data_processor.process_and_save_all()
                
            logger.info("GDPR knowledge base loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            raise
    
    def search_knowledge_base(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search the GDPR knowledge base for relevant information
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            results = self.data_processor.search_similar(query, k=k)
            logger.info(f"Found {len(results)} relevant documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []
    
    def analyze_website_privacy_policy(self, url: str) -> Dict[str, Any]:
        """
        Analyze a website's privacy policy for GDPR compliance
        
        Args:
            url: URL of the privacy policy page
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Analyzing privacy policy at: {url}")
            
            # Use agent to scrape and analyze the website
            scrape_query = f"Please analyze the GDPR compliance of the privacy policy at this URL: {url}"
            
            response = self.agent.run(scrape_query)
            
            # Extract relevant information from GDPR knowledge base
            kb_results = self.search_knowledge_base(f"privacy policy data protection compliance", k=3)
            
            return {
                'url': url,
                'agent_analysis': response,
                'relevant_gdpr_articles': kb_results,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing website privacy policy: {str(e)}")
            return {'error': str(e)}
    
    def answer_gdpr_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a GDPR-related question using the knowledge base and agent
        
        Args:
            question: User's question about GDPR
            
        Returns:
            Dictionary containing the answer and supporting information
        """
        try:
            logger.info(f"Answering GDPR question: {question}")
            
            # Check if question is GDPR-related (basic keyword filtering)
            gdpr_keywords = [
                'gdpr', 'data protection', 'privacy', 'personal data', 'data subject',
                'controller', 'processor', 'consent', 'lawful basis', 'dpo',
                'data breach', 'privacy policy', 'right to be forgotten', 'portability',
                'rectification', 'access', 'dpia', 'transfer', 'fine', 'penalty',
                'compliance', 'regulation', 'article', 'recital'
            ]
            
            question_lower = question.lower()
            is_gdpr_related = any(keyword in question_lower for keyword in gdpr_keywords)
            
            if not is_gdpr_related:
                return {
                    'answer': "I don't have information about this question. Please ask a question that is related to GDPR, data protection, privacy policies, or regulatory compliance domains.",
                    'relevant_documents': [],
                    'agent_response': None
                }
            
            # Search knowledge base for relevant information
            kb_results = self.search_knowledge_base(question, k=5)
            
            # Create enhanced context for the agent
            context = f"Question: {question}\n\nRelevant GDPR Information:\n"
            for i, result in enumerate(kb_results[:3], 1):
                context += f"{i}. {result['text'][:500]}...\n\n"
            
            # Get agent response
            agent_response = self.agent.run(context)
            
            return {
                'question': question,
                'answer': agent_response,
                'relevant_documents': kb_results,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error answering GDPR question: {str(e)}")
            return {'error': str(e)}
    
    def get_violation_cases(self, article: str = None, country: str = None, violation_type: str = None) -> List[Dict]:
        """
        Get GDPR violation cases based on filters
        
        Args:
            article: GDPR article number (e.g., "Art. 6 GDPR")
            country: Country name
            violation_type: Type of violation
            
        Returns:
            List of matching violation cases
        """
        try:
            # Build search query based on filters
            query_parts = []
            if article:
                query_parts.append(f"article {article}")
            if country:
                query_parts.append(f"country {country}")
            if violation_type:
                query_parts.append(f"violation type {violation_type}")
            
            query = " ".join(query_parts) if query_parts else "GDPR violation cases"
            
            # Search for violation cases
            results = self.search_knowledge_base(query, k=10)
            
            # Filter for violation cases only
            violation_cases = [
                result for result in results 
                if result['metadata']['type'] == 'violation_case'
            ]
            
            logger.info(f"Found {len(violation_cases)} violation cases")
            return violation_cases
            
        except Exception as e:
            logger.error(f"Error getting violation cases: {str(e)}")
            return []
    
    def get_agent_info(self) -> Dict[str, str]:
        """
        Get information about the agent
        
        Returns:
            Dictionary with agent information
        """
        if self.agent:
            return {
                'id': self.agent.id,
                'name': self.agent.name,
                'description': AGENT_DESCRIPTION[:200] + "...",
                'status': 'active'
            }
        else:
            return {'status': 'not_initialized'}


# Initialize the global agent instance
gdpr_agent = None

def get_gdpr_agent() -> GDPRComplianceAgent:
    """
    Get or create the global GDPR agent instance
    
    Returns:
        GDPRComplianceAgent instance
    """
    global gdpr_agent
    if gdpr_agent is None:
        gdpr_agent = GDPRComplianceAgent()
        gdpr_agent.load_knowledge_base()
    return gdpr_agent


if __name__ == "__main__":
    # Test the agent
    import pandas as pd
    
    agent = get_gdpr_agent()
    
    # Test question answering
    test_question = "What are the data subject rights under GDPR?"
    result = agent.answer_gdpr_question(test_question)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    
    # Test violation cases
    violations = agent.get_violation_cases(country="Spain")
    print(f"\nFound {len(violations)} violation cases from Spain")
