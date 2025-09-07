#!/usr/bin/env python3
"""
Standalone AI Ad Prompt Generation Test Script

This script directly uses Gemini API and RAG functionality without requiring
a FastAPI server. Perfect for testing prompt generation capabilities.

Usage:
    python standalone_prompt_test.py

Requirements:
    pip install google-generativeai chromadb sentence-transformers PyPDF2

Environment Variables:
    GEMINI_API_KEY=your_gemini_api_key_here
"""

import os
import json
import csv
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
GEMINI_API_KEY="IzaSyCrvNvQ4ODuLtp67qf_QeUaR-wFWUEqkYg"
# Third-party imports (install with pip)
try:
    import google.generativeai as genai
    import chromadb
    from sentence_transformers import SentenceTransformer
    import PyPDF2
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Please install required packages:")
    print("pip install google-generativeai chromadb sentence-transformers PyPDF2")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = "./test_outputs"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
VECTOR_DB_PATH = "./chroma_db"

# Test prompts for advertisement generation
TEST_PROMPTS = [
    {
        "id": 1,
        "category": "Technology",
        "input": "Create a modern advertisement for a new smartphone targeting tech-savvy millennials, emphasizing camera quality and sleek design"
    },
    {
        "id": 2,
        "category": "Food & Beverage", 
        "input": "Design an advertisement for an organic coffee brand targeting environmentally conscious consumers, highlighting sustainability and premium quality"
    },
    {
        "id": 3,
        "category": "Fashion",
        "input": "Create a luxury fashion advertisement for designer handbags targeting affluent women, emphasizing craftsmanship and exclusivity"
    },
    {
        "id": 4,
        "category": "Automotive",
        "input": "Generate an advertisement for an electric vehicle targeting eco-friendly families, focusing on safety and environmental benefits"
    },
    {
        "id": 5,
        "category": "Health & Wellness",
        "input": "Create an advertisement for a fitness app targeting busy professionals, emphasizing convenience and quick results"
    },
    {
        "id": 6,
        "category": "Travel",
        "input": "Design a travel advertisement for a luxury resort targeting couples, highlighting romance and exotic destinations"
    },
    {
        "id": 7,
        "category": "Financial Services",
        "input": "Create an advertisement for a cryptocurrency trading platform targeting young investors, emphasizing security and ease of use"
    },
    {
        "id": 8,
        "category": "Education",
        "input": "Generate an advertisement for an online learning platform targeting working professionals, focusing on career advancement and flexibility"
    },
    {
        "id": 9,
        "category": "Home & Garden",
        "input": "Create an advertisement for smart home devices targeting tech enthusiasts, emphasizing automation and energy efficiency"
    },
    {
        "id": 10,
        "category": "Entertainment",
        "input": "Design an advertisement for a streaming service targeting families, highlighting diverse content and affordable pricing"
    }
]


class StandaloneRAG:
    """Standalone RAG implementation using ChromaDB and SentenceTransformers."""
    
    def __init__(self, db_path: str = VECTOR_DB_PATH):
        self.db_path = db_path
        self.client = None
        self.collection = None
        self.encoder = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the vector database and encoder."""
        try:
            # Initialize sentence transformer
            logger.info("Loading sentence transformer...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
            logger.info("Initializing ChromaDB...")
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Try to get existing collection or create new one
            try:
                self.collection = self.client.get_collection("ad_knowledge")
                logger.info("Loaded existing vector store")
            except:
                logger.info("Creating new vector store...")
                self.collection = self.client.create_collection("ad_knowledge")
                self._load_sample_data()
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            # Create fallback sample data
            self._create_fallback_data()
    
    def _load_sample_data(self):
        """Load sample advertising knowledge into the vector store."""
        sample_docs = [
            {
                "id": "tech_1",
                "text": "Modern smartphone advertisements should emphasize camera quality, sleek design, and innovative features. Target tech-savvy millennials with vibrant visuals and contemporary aesthetics.",
                "category": "technology"
            },
            {
                "id": "food_1", 
                "text": "Organic coffee brand advertisements should highlight sustainability, premium quality, and environmental consciousness. Use earth tones and natural imagery to appeal to eco-conscious consumers.",
                "category": "food_beverage"
            },
            {
                "id": "fashion_1",
                "text": "Luxury fashion advertisements for designer handbags should emphasize craftsmanship, exclusivity, and prestige. Use elegant typography and sophisticated color palettes.",
                "category": "fashion"
            },
            {
                "id": "auto_1",
                "text": "Electric vehicle advertisements should focus on environmental benefits, safety features, and family appeal. Clean, modern designs with green elements work best.",
                "category": "automotive"
            },
            {
                "id": "wellness_1",
                "text": "Fitness app advertisements targeting professionals should emphasize convenience, time efficiency, and quick results. Use dynamic imagery and energetic colors.",
                "category": "health_wellness"
            },
            {
                "id": "travel_1",
                "text": "Luxury resort advertisements for couples should highlight romance, exotic destinations, and premium experiences. Use warm, inviting colors and scenic imagery.",
                "category": "travel"
            },
            {
                "id": "finance_1",
                "text": "Cryptocurrency platform advertisements should emphasize security, user-friendliness, and modern technology. Use clean, professional designs with tech-forward aesthetics.",
                "category": "financial"
            },
            {
                "id": "education_1",
                "text": "Online learning platform advertisements should focus on career advancement, flexibility, and professional growth. Use inspiring imagery and professional color schemes.",
                "category": "education"
            },
            {
                "id": "smart_home_1",
                "text": "Smart home device advertisements should emphasize automation, energy efficiency, and technological innovation. Use modern, clean designs with tech aesthetics.",
                "category": "home_garden"
            },
            {
                "id": "streaming_1",
                "text": "Family streaming service advertisements should highlight diverse content, affordable pricing, and family entertainment. Use bright, inclusive imagery and family-friendly colors.",
                "category": "entertainment"
            }
        ]
        
        # Add documents to collection
        for doc in sample_docs:
            self.collection.add(
                documents=[doc["text"]],
                ids=[doc["id"]],
                metadatas=[{"category": doc["category"]}]
            )
        
        logger.info(f"Added {len(sample_docs)} sample documents to vector store")
    
    def _create_fallback_data(self):
        """Create fallback knowledge base if ChromaDB fails."""
        self.fallback_knowledge = {
            "technology": "Modern tech ads should emphasize innovation, sleek design, and cutting-edge features with contemporary aesthetics.",
            "food_beverage": "Food and beverage ads should highlight quality, sustainability, and premium ingredients with natural imagery.",
            "fashion": "Fashion ads should emphasize style, craftsmanship, and exclusivity with elegant and sophisticated designs.",
            "automotive": "Auto ads should focus on safety, performance, and environmental benefits with clean, modern visuals.",
            "health_wellness": "Wellness ads should emphasize convenience, effectiveness, and lifestyle benefits with energetic designs.",
            "travel": "Travel ads should highlight experiences, destinations, and luxury with warm, inviting imagery.",
            "financial": "Financial service ads should emphasize security, trust, and ease of use with professional designs.",
            "education": "Education ads should focus on growth, opportunity, and flexibility with inspiring visuals.",
            "home_garden": "Home and garden ads should emphasize comfort, efficiency, and modern living with clean designs.",
            "entertainment": "Entertainment ads should highlight fun, variety, and value with bright, engaging visuals."
        }
        logger.warning("Using fallback knowledge base")
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for relevant documents."""
        try:
            if self.collection:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k
                )
                return results['documents'][0] if results['documents'] else []
            else:
                # Fallback: return generic advice based on query keywords
                fallback_docs = []
                query_lower = query.lower()
                for category, knowledge in self.fallback_knowledge.items():
                    if any(word in query_lower for word in category.split('_')):
                        fallback_docs.append(knowledge)
                return fallback_docs[:k] if fallback_docs else [list(self.fallback_knowledge.values())[0]]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return ["Focus on clear messaging, target audience appeal, and professional visual design."]


class StandaloneGeminiClient:
    """Standalone Gemini API client for prompt generation."""
    
    def __init__(self):
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Gemini API client."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def generate_ad_prompt(self, user_request: str, rag_context: List[str]) -> str:
        """Generate advertisement prompt using Gemini."""
        try:
            # Prepare context from RAG
            context = "\n".join(rag_context) if rag_context else "No specific context available."
            
            # Create the prompt for Gemini
            system_prompt = f"""
You are an expert advertisement copywriter and visual designer. Based on the user's request and relevant context, create a detailed image generation prompt for an advertisement.

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER REQUEST:
{user_request}

Create a detailed prompt for image generation that includes:
1. Visual composition and layout
2. Color scheme and aesthetic style  
3. Typography suggestions
4. Target audience considerations
5. Key visual elements and messaging
6. Specific artistic style or photography direction

The prompt should be detailed enough for an AI image generator to create a professional advertisement.

Format your response as a single, comprehensive image generation prompt (not a list or explanation).
"""

            response = self.model.generate_content(system_prompt)
            
            if response.text:
                logger.info("✅ Gemini generated advertisement prompt successfully")
                return response.text.strip()
            else:
                raise Exception("Gemini returned empty response")
                
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            # Fallback prompt generation
            return self._create_fallback_prompt(user_request, rag_context)
    
    def _create_fallback_prompt(self, user_request: str, rag_context: List[str]) -> str:
        """Create a basic fallback prompt if Gemini fails."""
        context_summary = rag_context[0] if rag_context else "professional advertisement design"
        return f"Create a professional advertisement based on: {user_request}. Style: {context_summary}. High quality, modern design, commercial photography style."


class StandalonePromptTester:
    """Standalone prompt generation tester."""
    
    def __init__(self):
        self.rag = None
        self.gemini = None
        self._initialize()
    
    def _initialize(self):
        """Initialize RAG and Gemini components."""
        try:
            logger.info("Initializing RAG system...")
            self.rag = StandaloneRAG()
            
            logger.info("Initializing Gemini API...")
            self.gemini = StandaloneGeminiClient()
            
            logger.info("✅ All systems initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def test_single_prompt(self, user_request: str) -> Dict[str, Any]:
        """Test a single prompt generation."""
        start_time = time.time()
        
        try:
            # Step 1: RAG search
            logger.info(f"🔍 Searching knowledge base for: {user_request[:50]}...")
            rag_results = self.rag.search(user_request, k=3)
            
            # Step 2: Generate prompt with Gemini
            logger.info("🧠 Generating prompt with Gemini...")
            ad_prompt = self.gemini.generate_ad_prompt(user_request, rag_results)
            
            end_time = time.time()
            
            return {
                "success": True,
                "ad_prompt": ad_prompt,
                "rag_context": rag_results,
                "error_message": "",
                "processing_time_seconds": round(end_time - start_time, 2)
            }
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"❌ Test failed: {e}")
            
            return {
                "success": False,
                "ad_prompt": "",
                "rag_context": [],
                "error_message": str(e),
                "processing_time_seconds": round(end_time - start_time, 2)
            }
    
    def run_full_test_suite(self) -> List[Dict[str, Any]]:
        """Run the complete test suite."""
        logger.info("🚀 Starting standalone prompt generation test suite...")
        
        results = []
        
        for i, test_case in enumerate(TEST_PROMPTS, 1):
            logger.info(f"\n=== Test {i}/10: {test_case['category']} ===")
            
            result = self.test_single_prompt(test_case['input'])
            
            # Add metadata
            result.update({
                'test_id': test_case['id'],
                'category': test_case['category'],
                'original_input': test_case['input'],
                'timestamp': datetime.now().isoformat()
            })
            
            results.append(result)
            
            # Log result
            if result['success']:
                logger.info(f"✅ Success: {result['ad_prompt'][:60]}...")
            else:
                logger.info(f"❌ Failed: {result['error_message'][:60]}...")
            
            # Wait between requests to be polite to APIs
            if i < len(TEST_PROMPTS):
                time.sleep(1)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Save test results to files."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_paths = {}
        
        # Main results file
        json_path = f"{OUTPUT_DIR}/standalone_results_{TIMESTAMP}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        file_paths['json'] = json_path
        
        # Input-Output pairs
        pairs_path = f"{OUTPUT_DIR}/prompt_pairs_{TIMESTAMP}.txt"
        with open(pairs_path, 'w', encoding='utf-8') as f:
            f.write("Standalone AI Ad Generation - Input/Output Prompt Pairs\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            successful_tests = [r for r in results if r.get('success') and r.get('ad_prompt')]
            f.write(f"Successful Tests: {len(successful_tests)}/{len(results)}\n\n")
            
            if successful_tests:
                f.write("SUCCESSFUL PROMPT GENERATIONS:\n")
                f.write("=" * 60 + "\n\n")
                
                for i, result in enumerate(successful_tests, 1):
                    f.write(f"PAIR {i}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Category: {result['category']}\n\n")
                    f.write(f"INPUT:\n{result['original_input']}\n\n")
                    f.write(f"GENERATED AD PROMPT:\n{result['ad_prompt']}\n\n")
                    f.write(f"RAG CONTEXT:\n")
                    for j, context in enumerate(result.get('rag_context', []), 1):
                        f.write(f"  {j}. {context}\n")
                    f.write(f"\nProcessing Time: {result['processing_time_seconds']}s\n")
                    f.write("=" * 60 + "\n\n")
        
        file_paths['pairs'] = pairs_path
        
        # Summary report
        summary_path = f"{OUTPUT_DIR}/test_summary_{TIMESTAMP}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            successful = len([r for r in results if r.get('success')])
            total = len(results)
            success_rate = (successful / total) * 100 if total > 0 else 0
            avg_time = sum(r.get('processing_time_seconds', 0) for r in results) / total if total > 0 else 0
            
            f.write(f"Standalone AI Ad Generation Test Summary\n")
            f.write(f"======================================\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"RESULTS:\n")
            f.write(f"- Total Tests: {total}\n")
            f.write(f"- Successful: {successful}\n")
            f.write(f"- Failed: {total - successful}\n")
            f.write(f"- Success Rate: {success_rate:.1f}%\n")
            f.write(f"- Average Time: {avg_time:.2f} seconds\n\n")
            
            # Category breakdown
            categories = {}
            for result in results:
                cat = result.get('category', 'Unknown')
                if cat not in categories:
                    categories[cat] = {'total': 0, 'successful': 0}
                categories[cat]['total'] += 1
                if result.get('success'):
                    categories[cat]['successful'] += 1
            
            f.write("CATEGORY BREAKDOWN:\n")
            for cat, stats in categories.items():
                rate = (stats['successful'] / stats['total']) * 100 if stats['total'] > 0 else 0
                f.write(f"- {cat}: {stats['successful']}/{stats['total']} ({rate:.1f}%)\n")
        
        file_paths['summary'] = summary_path
        
        return file_paths


def main():
    """Main execution function."""
    print("🚀 Standalone AI Ad Prompt Generation Test")
    print("=" * 50)
    print("This test runs independently without requiring a FastAPI server.")
    print("It directly uses Gemini API and RAG functionality.\n")
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY environment variable not set!")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        print("\nGet your API key from: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # Initialize tester
        tester = StandalonePromptTester()
        
        # Run tests
        results = tester.run_full_test_suite()
        
        if not results:
            print("❌ No tests were executed.")
            return
        
        # Calculate statistics
        successful = sum(1 for r in results if r.get('success'))
        total = len(results)
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"Total Tests: {total}")
        print(f"Successful: {successful}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if successful > 0:
            avg_time = sum(r['processing_time_seconds'] for r in results if r.get('success')) / successful
            print(f"Average Processing Time: {avg_time:.2f} seconds")
            
            # Show sample
            sample = next((r for r in results if r.get('success')), None)
            if sample:
                print(f"\n🎯 SAMPLE GENERATION:")
                print(f"Category: {sample['category']}")
                print(f"Input: {sample['original_input'][:80]}...")
                print(f"Generated: {sample['ad_prompt'][:100]}...")
        
        # Save results
        try:
            file_paths = tester.save_results(results)
            print(f"\n💾 RESULTS SAVED TO:")
            for file_type, path in file_paths.items():
                print(f"  {file_type.upper()}: {path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        print(f"\n🎉 STANDALONE TEST COMPLETE!")
        print(f"📁 Check the '{OUTPUT_DIR}' folder for detailed results.")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure GEMINI_API_KEY is set correctly")
        print("2. Check internet connection")
        print("3. Install required packages: pip install google-generativeai chromadb sentence-transformers PyPDF2")


if __name__ == "__main__":
    main()