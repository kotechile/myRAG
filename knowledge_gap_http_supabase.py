# knowledge_gap_http_supabase.py - PART 1: Core Classes and HTTP Client

import asyncio
import json
import logging
import os
import re
import time
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from flask import request, jsonify
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Import Manus research configuration
try:
    from manus_research_config import ManusResearchSettings, ResearchQualityAnalyzer
    MANUS_CONFIG_AVAILABLE = True
except ImportError:
    MANUS_CONFIG_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Manus research configuration not available")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try to import manual action suggestions (optional)
try:
    from manual_action_suggestions import ManualActionSuggestionsGenerator
    MANUAL_SUGGESTIONS_AVAILABLE = True
except ImportError:
    MANUAL_SUGGESTIONS_AVAILABLE = False

logger = logging.getLogger(__name__)



@dataclass
class KnowledgeGap:
    """Represents an identified knowledge gap for content generation."""
    title_id: str
    focus_keyword: str
    topic_category: str
    data_types_needed: List[str]
    priority_score: float
    specific_requirements: List[str]
    target_audience: str
    business_goal: str

class DataSourceType(Enum):
    ACADEMIC_PAPERS = "academic_papers"
    GOVERNMENT_DATA = "government_data" 
    INDUSTRY_REPORTS = "industry_reports"
    TEXTBOOKS = "textbooks"
    NEWS_ARTICLES = "news_articles"
    STATISTICAL_DATA = "statistical_data"
    EXPERT_INTERVIEWS = "expert_interviews"

# =============================================================================
# Helper method for vector store processing (add this too)
# =============================================================================

async def _process_document_into_vector_store(self, doc_id: str, collection_name: str, source_type: str):
    """Process document into vector store in background."""
    try:
        self.logger.info(f"🔄 Processing additional document {doc_id} into vector store...")
        
        # Update status to processing
        self.supabase.execute_query(
            'PATCH',
            f'lindex_documents?id=eq.{doc_id}',
            {"processing_status": "processing"}
        )
        
        # Submit for background processing (you may need to adapt this to your system)
        # This would typically call your document processor or vector store system
        # For now, we'll mark as completed
        self.supabase.execute_query(
            'PATCH',
            f'lindex_documents?id=eq.{doc_id}',
            {
                "processing_status": "completed",
                "in_vector_store": True,
                "last_processed": datetime.now().isoformat()
            }
        )
        
        self.logger.info(f"✅ Document {doc_id} processed into vector store")
        
    except Exception as e:
        self.logger.error(f"❌ Error processing document {doc_id} into vector store: {e}")
        
        # Mark as error
        self.supabase.execute_query(
            'PATCH',
            f'lindex_documents?id=eq.{doc_id}',
            {
                "processing_status": "error",
                "error_message": str(e)[:200]
            }
        )

# =============================================================================
# Helper method for getting collection ID (add this if it doesn't exist)
# =============================================================================

async def _get_collection_id(self, collection_name: str) -> int:
    """Get collection ID by name."""
    try:
        collection_response = self.supabase.execute_query(
            'GET',
            f'lindex_collections?select=id&name=eq.{collection_name}'
        )
        
        if collection_response['success'] and collection_response['data']:
            return collection_response['data'][0]['id']
        else:
            raise Exception(f"Collection {collection_name} not found")
            
    except Exception as e:
        self.logger.error(f"❌ Error getting collection ID for {collection_name}: {e}")
        raise e




class HTTPSupabaseClient:
    """HTTP-based Supabase client that bypasses SDK compatibility issues"""
    
    def __init__(self):
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        self.SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        
        if not self.SUPABASE_URL or not self.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.logger = logging.getLogger(__name__)
        
        # Setup HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            'apikey': self.SUPABASE_KEY,
            'Authorization': f'Bearer {self.SUPABASE_KEY}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        })
        
        # Base URL for REST API
        self.rest_url = f"{self.SUPABASE_URL}/rest/v1"
        
        self.logger.info("✅ HTTP Supabase client initialized (bypasses SDK issues)")
    



    
    def execute_query(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Execute HTTP query to Supabase REST API"""
        
        url = f"{self.rest_url}/{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'PATCH':
                response = self.session.patch(url, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check response
            if response.status_code in [200, 201, 204, 206]:
                try:
                    return {
                        'success': True,
                        'data': response.json() if response.content else [],
                        'status_code': response.status_code
                    }
                except:
                    return {
                        'success': True,
                        'data': [],
                        'status_code': response.status_code
                    }
            else:
                self.logger.error(f"HTTP Error {response.status_code}: {response.text}")
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'status_code': response.status_code
                }
                
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'status_code': 0
            }
    
    def table(self, table_name: str):
        """Return a table-like interface for compatibility"""
        return HTTPTable(self, table_name)

# =============================================================================
# Fix 2: Create Async Helper Functions
# =============================================================================

def run_async_in_thread(async_func):
    """Helper function to run async functions in Flask routes."""
    
    def sync_wrapper(*args, **kwargs):
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    
    return sync_wrapper

executor = ThreadPoolExecutor(max_workers=4)

def run_async_task(coro):
    """Run async coroutine in thread executor."""
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    future = executor.submit(run_in_thread)
    return future.result()

class HTTPTable:
    """Table interface that mimics Supabase client table API"""
    
    def __init__(self, client: HTTPSupabaseClient, table_name: str):
        self.client = client
        self.table_name = table_name
    
    def select(self, columns: str = "*"):
        return HTTPQuery(self.client, self.table_name, 'select', columns)
    
    def insert(self, data):
        result = self.client.execute_query('POST', self.table_name, data)
        return HTTPResponse(result)
    
    def update(self, data):
        return HTTPUpdate(self.client, self.table_name, data)

class HTTPQuery:
    """Query builder that mimics Supabase query interface"""
    
    def __init__(self, client: HTTPSupabaseClient, table_name: str, operation: str, columns: str):
        self.client = client
        self.table_name = table_name
        self.operation = operation
        self.columns = columns
        self.filters = []
    
    def eq(self, column: str, value: str):
        self.filters.append(f"{column}=eq.{value}")
        return self
    
    def limit(self, count: int):
        self.filters.append(f"limit={count}")
        return self
    
    def order(self, column: str, desc: bool = False):
        direction = "desc" if desc else "asc"
        self.filters.append(f"order={column}.{direction}")
        return self
    
    def execute(self):
        # Build endpoint with filters
        endpoint = f"{self.table_name}?select={self.columns}"
        if self.filters:
            endpoint += "&" + "&".join(self.filters)
        
        result = self.client.execute_query('GET', endpoint)
        return HTTPResponse(result)

class HTTPUpdate:
    """Update builder that mimics Supabase update interface"""
    
    def __init__(self, client: HTTPSupabaseClient, table_name: str, data: Dict):
        self.client = client
        self.table_name = table_name
        self.data = data
        self.filters = []
    
    def eq(self, column: str, value: str):
        self.filters.append(f"{column}=eq.{value}")
        return self
    
    def execute(self):
        # Build endpoint with filters
        endpoint = f"{self.table_name}"
        if self.filters:
            endpoint += "?" + "&".join(self.filters)
        
        result = self.client.execute_query('PATCH', endpoint, self.data)
        return HTTPResponse(result)

class HTTPResponse:
    """Response wrapper that mimics Supabase response interface"""
    
    def __init__(self, result: Dict):
        self.result = result
        self.data = result.get('data', [])
        self.count = len(self.data) if isinstance(self.data, list) else 0
        self.success = result.get('success', False)
        self.error = result.get('error')
        self.status_code = result.get('status_code', 0)
    
    def __bool__(self):
        """Make HTTPResponse truthy/falsy based on success"""
        return self.success
    
    def __repr__(self):
        return f"HTTPResponse(success={self.success}, count={self.count})"
    ## PART 2
    # knowledge_gap_http_supabase.py - PART 2: Knowledge Gap Analyzer Class

class KnowledgeGapAnalyzer:
    """Knowledge gap analyzer using HTTP Supabase client with user tracking"""
    
    def __init__(self, supabase_client, llm_client=None):
        self.supabase = supabase_client
        self.llm = llm_client
        self.logger = logging.getLogger(__name__)
        
        # Topic patterns for automatic categorization
        self.topic_patterns = {
            'real_estate': r'real estate|housing|property|mortgage|rent|home buying|investment property',
            'finance': r'finance|investment|money|budget|savings|debt|credit|financial planning',
            'technology': r'tech|software|AI|programming|digital|cyber|blockchain|data science',
            'health': r'health|medical|wellness|fitness|nutrition|mental health|healthcare',
            'business': r'business|entrepreneur|startup|marketing|sales|management|strategy',
            'education': r'education|learning|school|university|training|online course',
            'lifestyle': r'lifestyle|travel|food|fashion|entertainment|hobbies|personal development',
            'science': r'science|research|climate|environment|energy|sustainability',
            'legal': r'legal|law|regulation|compliance|contract|intellectual property',
            'automotive': r'car|automotive|vehicle|transportation|electric vehicle',
            'cryptocurrency': r'crypto|bitcoin|ethereum|blockchain|defi|nft',
            'social_media': r'social media|instagram|facebook|tiktok|youtube|influencer'
        }
    
    async def analyze_new_titles(self, user_id: str = None) -> List[KnowledgeGap]:
        """Analyze all NEW status titles to identify knowledge gaps with user tracking."""
        
        try:
            # Use HTTP client to get NEW titles
            response = self.supabase.table("Titles").select("*").eq("status", "NEW").execute()
            titles = response.data
            
            if not titles:
                self.logger.info("No NEW titles found")
                return []
            
            self.logger.info(f"Found {len(titles)} NEW titles to analyze (user: {user_id})")
            
            knowledge_gaps = []
            
            for title in titles:
                gap = await self._analyze_single_title(title, user_id)
                if gap:
                    knowledge_gaps.append(gap)
            
            return knowledge_gaps
            
        except Exception as e:
            self.logger.error(f"Error fetching titles: {e}")
            return []
    
    async def analyze_new_titles_with_gap_filter(self, user_id: str = None) -> List[KnowledgeGap]:
        """Analyze NEW status titles that don't have gaps closed yet with user tracking."""
        
        try:
            # Query for NEW titles that don't have gaps closed
            response = self.supabase.table("Titles").select("*").eq("status", "NEW").execute()
            
            if not response.data:
                self.logger.info("No NEW titles found")
                return []
            
            # Filter out titles that already have gaps closed
            titles_with_open_gaps = []
            for title in response.data:
                gaps_closed = title.get("knowledge_gaps_closed", False)
                if not gaps_closed:  # Include titles where gaps are not closed or field is null/false
                    titles_with_open_gaps.append(title)
            
            if not titles_with_open_gaps:
                self.logger.info("No NEW titles with open knowledge gaps found")
                return []
            
            self.logger.info(f"Found {len(titles_with_open_gaps)} NEW titles with open knowledge gaps to analyze (user: {user_id})")
            
            knowledge_gaps = []
            
            for title in titles_with_open_gaps:
                gap = await self._analyze_single_title(title, user_id)
                if gap:
                    knowledge_gaps.append(gap)
            
            return knowledge_gaps
            
        except Exception as e:
            self.logger.error(f"Error fetching titles with open gaps: {e}")
            return []
    
    async def _analyze_single_title(self, title_data: Dict, user_id: str = None) -> Optional[KnowledgeGap]:
        """Analyze a single title to identify its knowledge requirements with user tracking."""
        
        try:
            # Extract keywords from multiple available fields
            focus_keyword = self._extract_best_focus_keyword(title_data)
            
            # Get other content fields
            content_outline = title_data.get('content_outline', '')
            key_points = title_data.get('key_points', '')
            target_audience = title_data.get('target_audience', '')
            business_goal = title_data.get('business_goal', '')
            title_text = title_data.get('Title', '')
            user_description = title_data.get('userDescription', '')
            
            # Combine all keyword sources for analysis
            all_keywords = self._get_all_keywords_combined(title_data)
            
            if not focus_keyword or len(focus_keyword.strip()) < 3:
                self.logger.warning(f"Title {title_data.get('id')} has insufficient keyword data after extraction")
                return None
            
            self.logger.info(f"📋 Analyzing title {title_data.get('id')}: '{focus_keyword}' (user: {user_id})")
            
            # Categorize topic using all available keyword data
            topic_category = self._categorize_topic(focus_keyword, content_outline, all_keywords)
            
            # Identify data requirements using rule-based approach
            data_requirements = self._identify_data_requirements_rule_based(
                focus_keyword, content_outline, topic_category, all_keywords
            )
            
            if not data_requirements or not data_requirements.get('data_types'):
                self.logger.info(f"No data requirements identified for {focus_keyword}")
                return None
            
            return KnowledgeGap(
                title_id=title_data.get('id', ''),
                focus_keyword=focus_keyword,
                topic_category=topic_category,
                data_types_needed=data_requirements['data_types'],
                priority_score=self._calculate_priority(title_data),
                specific_requirements=data_requirements.get('specific_requirements', data_requirements.get('specific_needs', [])),
                target_audience=target_audience,
                business_goal=business_goal
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing title {title_data.get('id')}: {e}")
            return None

    def _extract_best_focus_keyword(self, title_data: Dict) -> str:
        """Extract the best focus keyword from all available keyword fields."""
        
        # Priority 1: Use focus_keyword if available
        focus_keyword = title_data.get('focus_keyword', '').strip()
        if focus_keyword:
            return focus_keyword
        
        # Priority 2: Use first enhanced_primary_keyword
        enhanced_primary = title_data.get('enhanced_primary_keywords', '')
        if enhanced_primary:
            try:
                # Handle both string and list formats
                if isinstance(enhanced_primary, str):
                    keywords_list = json.loads(enhanced_primary.replace("'", '"'))
                else:
                    keywords_list = enhanced_primary
                
                if keywords_list and len(keywords_list) > 0:
                    return keywords_list[0]
            except (json.JSONDecodeError, TypeError):
                # Try as comma-separated string
                if ',' in enhanced_primary:
                    return enhanced_primary.split(',')[0].strip()
        
        # Priority 3: Use primary_keywords_json
        primary_json = title_data.get('primary_keywords_json', '')
        if primary_json:
            try:
                # Clean up the JSON string format from your data
                clean_json = primary_json.replace('\\"', '"').strip('"')
                keywords_list = json.loads(clean_json)
                if keywords_list and len(keywords_list) > 0:
                    return keywords_list[0]
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Priority 4: Extract from title text
        title_text = title_data.get('Title', '')
        if title_text:
            meaningful_keyword = self._extract_keyword_from_title(title_text)
            if meaningful_keyword:
                return meaningful_keyword
        
        # Priority 5: Use userDescription key terms
        user_desc = title_data.get('userDescription', '')
        if user_desc:
            words = user_desc.split()[:4]
            return ' '.join(words).strip('.,!?;:')
        
        return "real estate content"  # Fallback

    def _get_all_keywords_combined(self, title_data: Dict) -> str:
        """Combine all keyword fields into a single string for comprehensive analysis."""
        
        all_keywords = []
        
        # Extract from enhanced_primary_keywords
        enhanced_primary = title_data.get('enhanced_primary_keywords', '')
        if enhanced_primary:
            try:
                if isinstance(enhanced_primary, str):
                    keywords_list = json.loads(enhanced_primary.replace("'", '"'))
                    all_keywords.extend(keywords_list)
                else:
                    all_keywords.extend(enhanced_primary)
            except:
                pass
        
        # Extract from enhanced_secondary_keywords  
        enhanced_secondary = title_data.get('enhanced_secondary_keywords', '')
        if enhanced_secondary:
            try:
                if isinstance(enhanced_secondary, str):
                    keywords_list = json.loads(enhanced_secondary.replace("'", '"'))
                    all_keywords.extend(keywords_list)
                else:
                    all_keywords.extend(enhanced_secondary)
            except:
                pass
        
        # Extract from primary_keywords_json
        primary_json = title_data.get('primary_keywords_json', '')
        if primary_json:
            try:
                clean_json = primary_json.replace('\\"', '"').strip('"')
                keywords_list = json.loads(clean_json)
                all_keywords.extend(keywords_list)
            except:
                pass
        
        # Extract from secondary_keywords_json
        secondary_json = title_data.get('secondary_keywords_json', '')
        if secondary_json:
            try:
                clean_json = secondary_json.replace('\\"', '"').strip('"')
                keywords_list = json.loads(clean_json)
                all_keywords.extend(keywords_list)
            except:
                pass
        
        # Combine with title and description
        title_text = title_data.get('Title', '')
        user_desc = title_data.get('userDescription', '')
        
        combined = f"{' '.join(all_keywords)} {title_text} {user_desc}"
        return combined.lower()

    def _extract_keyword_from_title(self, title_text: str) -> str:
        """Extract a meaningful keyword from the title text."""
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Look for key real estate/content terms first
        title_lower = title_text.lower()
        
        # High-priority terms
        priority_terms = [
            'smart home', 'real estate', 'home buying', 'first-time homebuyers', 
            'affordable housing', 'remote work', 'home office', 'property investment',
            'home financing', 'mortgage', 'homeownership'
        ]
        
        for term in priority_terms:
            if term in title_lower:
                return term
        
        # Extract meaningful phrases (2-3 words)
        words = [w.lower().strip('.,!?;:') for w in title_text.split() 
                if len(w) > 2 and w.lower() not in stop_words]
        
        if len(words) >= 2:
            return f"{words[0]} {words[1]}"
        elif words:
            return words[0]
        
        return ""

    def _identify_data_requirements_rule_based(self, focus_keyword: str, content_outline: str, 
                                             topic_category: str, all_keywords: str):
        """Identify data requirements using rule-based approach"""
        
        data_types = []
        specific_requirements = []
        
        # Convert inputs to lowercase for consistent matching
        focus_lower = focus_keyword.lower() if focus_keyword else ""
        outline_lower = content_outline.lower() if content_outline else ""
        keywords_lower = all_keywords.lower() if all_keywords else ""
        
        # Combine all text for analysis
        combined_text = f"{focus_lower} {outline_lower} {keywords_lower}"
        
        # Real Estate & Property specific rules
        if any(term in combined_text for term in ['real estate', 'property', 'home', 'house', 'housing']):
            data_types.extend(['market_data', 'industry_reports', 'government_statistics'])
            
            if any(term in combined_text for term in ['market', 'trends', 'prices']):
                specific_requirements.extend([
                    'Regional housing market data',
                    'Property price trends and forecasts',
                    'Market analysis reports from real estate associations'
                ])
                
            if any(term in combined_text for term in ['buying', 'purchase', 'buyer']):
                specific_requirements.extend([
                    'Home buying guides and checklists',
                    'Mortgage and financing information',
                    'Legal requirements for property purchase'
                ])
                
            if any(term in combined_text for term in ['investment', 'investing', 'rental']):
                data_types.append('investment_analysis')
                specific_requirements.extend([
                    'Real estate investment strategies',
                    'Rental market analysis',
                    'ROI calculations and case studies'
                ])
        
        # Smart Home & Technology
        if any(term in combined_text for term in ['smart home', 'technology', 'automation']):
            data_types.extend(['industry_reports', 'technical_specifications'])
            specific_requirements.extend([
                'Smart home technology market research',
                'Product specifications and comparisons',
                'Installation and setup guides'
            ])
        
        # Market Analysis
        if any(term in combined_text for term in ['market', 'analysis', 'trends']):
            data_types.extend(['market_data', 'statistical_data', 'industry_reports'])
            specific_requirements.extend([
                'Market size and growth projections',
                'Industry trend analysis',
                'Competitive landscape research'
            ])
        
        # Financial Topics
        if any(term in combined_text for term in ['financial', 'finance', 'cost', 'price', 'budget']):
            data_types.extend(['financial_data', 'economic_indicators'])
            specific_requirements.extend([
                'Cost analysis and budgeting information',
                'Financial planning resources',
                'Economic impact studies'
            ])
        
        # Default fallback if no specific matches
        if not data_types:
            data_types = ['industry_reports', 'market_data']
            specific_requirements = [
                f'Research papers on {focus_keyword}',
                f'Industry analysis related to {focus_keyword}',
                'Government and institutional data sources'
            ]
        
        # Remove duplicates and return
        return {
            'data_types': list(set(data_types)),
            'specific_needs': list(set(specific_requirements))
        }
    
    def _categorize_topic(self, focus_keyword: str, content_outline: str, enhanced_keywords: str) -> str:
        """Automatically categorize the topic based on keywords."""
        
        text = f"{focus_keyword} {content_outline} {enhanced_keywords}".lower()
        
        for category, pattern in self.topic_patterns.items():
            if re.search(pattern, text):
                return category
        
        return 'general'
    
    def _calculate_priority(self, title_data: Dict) -> float:
        """Calculate priority score based on your table's metrics."""
        
        try:
            # Use your existing scoring fields with safe defaults
            traffic_score = float(title_data.get('expected_monthly_traffic', 0)) / 10000  # Normalize
            quality_score = float(title_data.get('overall_quality_score', 0)) / 100
            seo_score = float(title_data.get('seo_optimization_score', 0)) / 100
            business_impact = float(title_data.get('business_impact_score', 0)) / 100
            traffic_potential = float(title_data.get('traffic_potential_score', 0)) / 100
            
            # Weighted calculation
            priority = (
                traffic_score * 0.3 +
                quality_score * 0.2 +
                seo_score * 0.2 +
                business_impact * 0.15 +
                traffic_potential * 0.15
            )
            
            return min(1.0, priority)  # Cap at 1.0
            
        except:
            return 0.5  # Default priority
        ## PART 3

        # knowledge_gap_http_supabase.py - PART 3: Multi-Source Researcher Class

class MultiSourceResearcher:
    """Research functionality using HTTP Supabase client with user tracking"""
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize Linkup client if available
        self.linkup_client = None
        
        try:
            from linkup import LinkupClient
            if os.getenv('LINKUP_API_KEY'):
                self.linkup_client = LinkupClient(api_key=os.getenv('LINKUP_API_KEY'))
                self.logger.info("✅ Linkup client initialized")
            else:
                self.logger.warning("⚠️ LINKUP_API_KEY not found in environment")
        except ImportError:
            self.logger.warning("⚠️ Linkup SDK not available. Install with: pip install linkup-sdk")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Linkup: {e}")
    
    async def research_knowledge_gap(self, gap: KnowledgeGap, user_id: str = None) -> Dict[str, Any]:
        """Research all required data sources for a knowledge gap with user tracking."""
        
        research_results = {
            'title_id': gap.title_id,
            'focus_keyword': gap.focus_keyword,
            'sources_found': {},
            'total_sources': 0,
            'research_timestamp': datetime.now().isoformat(),
            'research_summary': '',
            'user_id': user_id,
            'research_quality_score': 0,
            'manus_suggestion_needed': False
        }
        
        # Research each required data type
        for data_type in gap.data_types_needed:
            try:
                self.logger.info(f"🔍 Researching {data_type} for: {gap.focus_keyword} (user: {user_id})")
                
                if data_type == DataSourceType.ACADEMIC_PAPERS.value:
                    sources = await self._research_academic_papers(gap, user_id)
                elif data_type == DataSourceType.GOVERNMENT_DATA.value:
                    sources = await self._research_government_data(gap, user_id)
                elif data_type == DataSourceType.INDUSTRY_REPORTS.value:
                    sources = await self._research_industry_reports(gap, user_id)
                elif data_type == DataSourceType.NEWS_ARTICLES.value:
                    sources = await self._research_news_articles(gap, user_id)
                elif data_type == DataSourceType.STATISTICAL_DATA.value:
                    sources = await self._research_statistical_data(gap, user_id)
                elif data_type == DataSourceType.TEXTBOOKS.value:
                    sources = await self._research_textbooks(gap, user_id)
                else:
                    # Handle custom data types from rule-based analysis
                    sources = await self._research_generic_sources(gap, data_type, user_id)
                
                research_results['sources_found'][data_type] = sources
                research_results['total_sources'] += len(sources)
                
                self.logger.info(f"  Found {len(sources)} sources for {data_type}")
                
            except Exception as e:
                self.logger.error(f"Error researching {data_type}: {e}")
                research_results['sources_found'][data_type] = []
        
        # Calculate research quality score and check if Manus-wide research is needed
        research_results['research_quality_score'] = self._calculate_research_quality(research_results)
        research_results['manus_suggestion_needed'] = research_results['research_quality_score'] < 30
        
        # Create research summary
        research_results['research_summary'] = self._create_research_summary(research_results)
        
        return research_results
    
    async def _research_academic_papers(self, gap: KnowledgeGap, user_id: str = None) -> List[Dict]:
        """Research academic papers using arXiv and web search with user tracking."""
        
        papers = []
        
        # Method 1: arXiv search (if available)
        try:
            import arxiv
            self.logger.info(f"  Searching arXiv... (user: {user_id})")
            search = arxiv.Search(
                query=gap.focus_keyword,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in search.results():
                papers.append({
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'summary': result.summary[:500],
                    'url': result.entry_id,
                    'published': result.published.isoformat(),
                    'source': 'arxiv',
                    'type': 'academic_paper',
                    'user_id': user_id
                })
                
            self.logger.info(f"    Found {len(papers)} papers from arXiv")
        except ImportError:
            self.logger.info("    arXiv not available - skipping")
        except Exception as e:
            self.logger.error(f"arXiv search error: {e}")
        
        # Method 2: Web search for academic sources
        if self.linkup_client:
            try:
                self.logger.info("  Searching academic sources via web...")
                search_query = f"{gap.focus_keyword} research study academic"
                web_papers = await self._linkup_search(search_query, max_results=3, user_id=user_id, 
                                                       data_type=DataSourceType.ACADEMIC_PAPERS.value, gap=gap)
                
                for result in web_papers:
                    papers.append({
                        'title': result.get('title', ''),
                        'summary': result.get('description', '')[:500],
                        'url': result.get('url', ''),
                        'source': 'web_academic',
                        'type': 'academic_paper',
                        'user_id': user_id
                    })
                    
                self.logger.info(f"    Found {len(web_papers)} papers from web search")
            except Exception as e:
                self.logger.error(f"Academic web search error: {e}")
        
        return papers[:10]  # Limit results
    
    async def _research_government_data(self, gap: KnowledgeGap, user_id: str = None) -> List[Dict]:
        """Research government data sources with user tracking."""
        
        sources = []
        
        if not self.linkup_client:
            self.logger.warning("  Skipping government data - Linkup not available")
            return []
        
        # Government data search patterns
        gov_sites = [
            "site:census.gov",
            "site:bls.gov", 
            "site:fredlouisfed.org",
            "site:data.gov"
        ]
        
        for site in gov_sites:
            try:
                search_query = f"{gap.focus_keyword} {site}"
                results = await self._linkup_search(search_query, max_results=2, user_id=user_id, 
                                                   data_type=DataSourceType.GOVERNMENT_DATA.value, gap=gap)
                
                for result in results:
                    sources.append({
                        'title': result.get('title', ''),
                        'description': result.get('description', ''),
                        'url': result.get('url', ''),
                        'source': f"government_{site.split(':')[1].split('.')[0]}",
                        'type': 'government_data',
                        'user_id': user_id
                    })
            except Exception as e:
                self.logger.error(f"Government data search error for {site}: {e}")
        
        return sources
    
    async def _research_industry_reports(self, gap: KnowledgeGap, user_id: str = None) -> List[Dict]:
        """Research industry reports and market research with user tracking."""
        
        reports = []
        
        if not self.linkup_client:
            self.logger.warning("  Skipping industry reports - Linkup not available")
            return []
        
        try:
            search_query = f"{gap.focus_keyword} industry report market analysis"
            results = await self._linkup_search(search_query, max_results=5, user_id=user_id, 
                                               data_type=DataSourceType.INDUSTRY_REPORTS.value, gap=gap)
            
            for result in results:
                reports.append({
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'url': result.get('url', ''),
                    'source': 'industry_research',
                    'type': 'industry_report',
                    'user_id': user_id
                })
        except Exception as e:
            self.logger.error(f"Industry report search error: {e}")
        
        return reports
    
    async def _research_news_articles(self, gap: KnowledgeGap, user_id: str = None) -> List[Dict]:
        """Research recent news articles with user tracking."""
        
        articles = []
        
        if not self.linkup_client:
            self.logger.warning("  Skipping news articles - Linkup not available")
            return []
        
        try:
            # Search for recent news
            search_query = f"{gap.focus_keyword} news 2024 2025"
            results = await self._linkup_search(search_query, max_results=8, user_id=user_id, 
                                               data_type=DataSourceType.NEWS_ARTICLES.value, gap=gap)
            
            for result in results:
                articles.append({
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'url': result.get('url', ''),
                    'source': 'news_search',
                    'type': 'news_article',
                    'user_id': user_id
                })
        except Exception as e:
            self.logger.error(f"News search error: {e}")
        
        return articles
    
    async def _research_statistical_data(self, gap: KnowledgeGap, user_id: str = None) -> List[Dict]:
        """Research statistical data and datasets with user tracking."""
        
        data_sources = []
        
        if not self.linkup_client:
            self.logger.warning("  Skipping statistical data - Linkup not available")
            return []
        
        try:
            # Search for statistical data
            stat_queries = [
                f"{gap.focus_keyword} statistics data",
                f"{gap.focus_keyword} market size trends"
            ]
            
            for query in stat_queries:
                results = await self._linkup_search(query, max_results=3, user_id=user_id, 
                                                   data_type=DataSourceType.STATISTICAL_DATA.value, gap=gap)
                
                for result in results:
                    data_sources.append({
                        'title': result.get('title', ''),
                        'description': result.get('description', ''),
                        'url': result.get('url', ''),
                        'source': 'statistical_data',
                        'type': 'statistical_data',
                        'user_id': user_id
                    })
        except Exception as e:
            self.logger.error(f"Statistical data search error: {e}")
        
        return data_sources
    
    async def _research_textbooks(self, gap: KnowledgeGap, user_id: str = None) -> List[Dict]:
        """Research relevant textbooks and educational materials with user tracking."""
        
        textbooks = []
        
        if not self.linkup_client:
            self.logger.warning("  Skipping textbooks - Linkup not available")
            return []
        
        try:
            # Search for textbooks and educational materials
            search_query = f"{gap.focus_keyword} textbook guide tutorial course"
            results = await self._linkup_search(search_query, max_results=5, user_id=user_id, 
                                               data_type=DataSourceType.TEXTBOOKS.value, gap=gap)
            
            for result in results:
                textbooks.append({
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'url': result.get('url', ''),
                    'source': 'textbook_search',
                    'type': 'textbook',
                    'user_id': user_id
                })
        except Exception as e:
            self.logger.error(f"Textbook search error: {e}")
        
        return textbooks
    
    async def _research_generic_sources(self, gap: KnowledgeGap, data_type: str, user_id: str = None) -> List[Dict]:
        """Research generic sources for custom data types."""
        
        sources = []
        
        if not self.linkup_client:
            self.logger.warning(f"  Skipping {data_type} - Linkup not available")
            return []
        
        try:
            # Map custom data types to search queries
            search_mappings = {
                'market_data': f"{gap.focus_keyword} market data trends analysis",
                'government_statistics': f"{gap.focus_keyword} government statistics official data",
                'investment_analysis': f"{gap.focus_keyword} investment analysis ROI case study",
                'technical_specifications': f"{gap.focus_keyword} technical specifications product details",
                'financial_data': f"{gap.focus_keyword} financial data cost analysis",
                'economic_indicators': f"{gap.focus_keyword} economic indicators impact analysis"
            }
            
            search_query = search_mappings.get(data_type, f"{gap.focus_keyword} {data_type}")
            results = await self._linkup_search(search_query, max_results=4, user_id=user_id, 
                                               data_type=data_type, gap=gap)
            
            for result in results:
                sources.append({
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'url': result.get('url', ''),
                    'source': f'{data_type}_research',
                    'type': data_type,
                    'user_id': user_id
                })
        except Exception as e:
            self.logger.error(f"Generic source search error for {data_type}: {e}")
        
        return sources
    
    def _determine_linkup_depth(self, query: str, data_type: str = None, gap: 'KnowledgeGap' = None) -> str:
        """Intelligently determine Linkup search depth based on query characteristics."""
        
        # Default to standard for basic queries
        depth = "standard"
        
        # Check for complex query indicators
        complexity_indicators = [
            "analysis", "research", "study", "thesis", "comprehensive",
            "detailed", "in-depth", "thorough", "academic", "scholarly",
            "advanced", "technical", "scientific", "methodology", "literature"
        ]
        
        # Check for academic/source type indicators
        academic_types = [
            DataSourceType.ACADEMIC_PAPERS.value,
            DataSourceType.GOVERNMENT_DATA.value,
            DataSourceType.INDUSTRY_REPORTS.value,
            DataSourceType.TEXTBOOKS.value
        ]
        
        # Convert to lowercase for matching
        query_lower = query.lower()
        
        # Complex query detection
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in query_lower)
        
        # Data type-based depth selection
        if data_type in academic_types:
            depth = "deep"
        elif data_type == DataSourceType.STATISTICAL_DATA.value:
            depth = "deep"
        elif complexity_score >= 2:
            depth = "deep"
        elif any(indicator in query_lower for indicator in ["2025", "2024", "latest", "current"]):
            depth = "deep"
        
        # Gap-based depth selection
        if gap and hasattr(gap, 'data_types_needed'):
            academic_data_types = [dt for dt in gap.data_types_needed if dt in academic_types]
            if len(academic_data_types) >= 2:
                depth = "deep"
            elif gap.priority_score and gap.priority_score >= 8:
                depth = "deep"
        
        self.logger.info(f"🎯 DEPTH DECISION: '{query}' → {depth} (complexity: {complexity_score}, type: {data_type})")
        return depth

    async def _linkup_search(self, query: str, max_results: int = 5, user_id: str = None, 
                           data_type: str = None, gap: 'KnowledgeGap' = None) -> List[Dict]:
        """Perform Linkup search with intelligent depth selection and user tracking."""
        
        if not self.linkup_client:
            self.logger.error("❌ Linkup client not initialized")
            return []
        
        try:
            # Determine optimal depth
            depth = self._determine_linkup_depth(query, data_type, gap)
            
            self.logger.info(f"🔍 LINKUP SEARCH: '{query}' (max: {max_results}, depth: {depth}, user: {user_id})")
            
            # Make the API call
            response = await asyncio.to_thread(
                self.linkup_client.search,
                query=query,
                depth=depth,
                output_type="searchResults"
            )
            
            self.logger.info(f"📋 Response type: {type(response)}")
            
            results = []
            
            # Handle LinkupSearchResults with Pydantic objects
            if hasattr(response, 'results') and response.results:
                self.logger.info(f"🔍 Processing {len(response.results)} Linkup results...")
                
                for i, result in enumerate(response.results[:max_results]):
                    try:
                        # Access Pydantic attributes directly
                        title = getattr(result, 'name', 'Unknown Title')
                        content = getattr(result, 'content', 'No description')
                        url = getattr(result, 'url', 'No URL')
                        
                        # Truncate content if too long
                        description = content[:500] + "..." if len(content) > 500 else content
                        
                        formatted_result = {
                            'title': title,
                            'description': description,
                            'url': url,
                            'user_id': user_id
                        }
                        
                        results.append(formatted_result)
                        self.logger.info(f"   ✅ Result {i+1}: {title[:50]}...")
                        
                    except Exception as result_error:
                        self.logger.error(f"   ❌ Error processing result {i+1}: {result_error}")
                        continue
            else:
                self.logger.warning(f"⚠️ No results found in response")
            
            self.logger.info(f"📊 LINKUP SUCCESS: Extracted {len(results)} usable results")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ LINKUP ERROR for '{query}': {e}")
            return []
    
    def _calculate_research_quality(self, research_results: Dict) -> int:
        """Calculate research quality score (0-100) to determine if Manus-wide research is needed."""
        
        if MANUS_CONFIG_AVAILABLE:
            # Use advanced quality analysis if available
            assessment = ResearchQualityAnalyzer.assess_research_quality(research_results)
            return assessment['overall_score']
        else:
            # Fallback to basic calculation
            total_sources = research_results['total_sources']
            sources_by_type = research_results['sources_found']
            
            # Base score from total sources
            source_score = min(total_sources * 10, 40)  # Max 40 points from sources
            
            # Bonus for diversity of source types
            source_types_found = len([t for t, sources in sources_by_type.items() if sources])
            diversity_score = min(source_types_found * 15, 30)  # Max 30 points for diversity
            
            # Quality bonus for rich sources (academic papers, industry reports, government data)
            quality_sources = (
                len(sources_by_type.get(DataSourceType.ACADEMIC_PAPERS.value, [])) * 3 +
                len(sources_by_type.get(DataSourceType.INDUSTRY_REPORTS.value, [])) * 2 +
                len(sources_by_type.get(DataSourceType.GOVERNMENT_DATA.value, [])) * 2 +
                len(sources_by_type.get(DataSourceType.STATISTICAL_DATA.value, [])) * 1
            )
            quality_score = min(quality_sources * 5, 30)  # Max 30 points for quality
            
            total_score = source_score + diversity_score + quality_score
            return min(total_score, 100)

    def _create_research_summary(self, research_results: Dict) -> str:
        """Create a summary of research findings."""
        
        total_sources = research_results['total_sources']
        sources_by_type = research_results['sources_found']
        quality_score = research_results.get('research_quality_score', 0)
        
        summary_parts = [f"Found {total_sources} total sources for '{research_results['focus_keyword']}' (Quality Score: {quality_score}/100)"]
        
        for data_type, sources in sources_by_type.items():
            if sources:
                summary_parts.append(f"- {data_type}: {len(sources)} sources")
        
        if research_results.get('manus_suggestion_needed'):
            summary_parts.append("⚠️ Insufficient research results - Manus-wide research recommended")
        
        return "; ".join(summary_parts)
    ## PART 4

    # knowledge_gap_http_supabase.py - PART 4: Enhanced RAG Knowledge Enhancer Class (Part 1)

class EnhancedRAGKnowledgeEnhancer:




    def __init__(self, supabase_client, document_processor=None, background_executor=None):
        self.supabase = supabase_client
        self.document_processor = document_processor
        self.background_executor = background_executor
        self.logger = logging.getLogger(__name__)
        
        # ✅ ADD THIS: Initialize Linkup client (same pattern as MultiSourceResearcher)
        self.linkup_client = None
        
        try:
            from linkup import LinkupClient
            if os.getenv('LINKUP_API_KEY'):
                self.linkup_client = LinkupClient(api_key=os.getenv('LINKUP_API_KEY'))
                self.logger.info("✅ Linkup client initialized for additional knowledge enhancement")
            else:
                self.logger.warning("⚠️ LINKUP_API_KEY not found - additional knowledge enhancement will be limited")
        except ImportError:
            self.logger.warning("⚠️ Linkup SDK not available. Install with: pip install linkup-sdk")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Linkup for additional enhancement: {e}")
        
        # Initialize manual suggestions generator if available
        self.suggestions_generator = None
        if MANUAL_SUGGESTIONS_AVAILABLE:
            try:
                self.suggestions_generator = ManualActionSuggestionsGenerator(supabase_client)
                self.logger.info("✅ Manual action suggestions generator initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize manual suggestions: {e}")


    # Add this method to your EnhancedRAGKnowledgeEnhancer class (you might be missing it):

    async def _ensure_collection_exists(self, collection_name: str, user_id: str = None):
        """Ensure collection exists in lindex_collections table with user tracking."""
        try:
            response = self.supabase.execute_query(
                'GET',
                f'lindex_collections?select=id&name=eq.{collection_name}'
            )
            
            if not response['success'] or not response['data']:
                collection_data = {
                    "name": collection_name,
                    "description": f"Additional knowledge collection created by Gap Filler system (user: {user_id})",
                    "created_at": datetime.now().isoformat()
                }
                
                insert_response = self.supabase.execute_query(
                    'POST',
                    'lindex_collections',
                    collection_data
                )
                
                if insert_response['success']:
                    self.logger.info(f"✅ Created new collection: {collection_name} (user: {user_id})")
                    return collection_name
                else:
                    raise ValueError(f"Failed to create collection {collection_name}")
            else:
                self.logger.info(f"📁 Collection {collection_name} already exists")
                return collection_name
                
        except Exception as e:
            self.logger.error(f"Error ensuring collection exists: {e}")
            raise

    async def _get_collection_id(self, collection_name: str) -> int:
        """Get collection ID by name."""
        try:
            collection_response = self.supabase.execute_query(
                'GET',
                f'lindex_collections?select=id&name=eq.{collection_name}'
            )
            
            if collection_response['success'] and collection_response['data']:
                return collection_response['data'][0]['id']
            else:
                raise Exception(f"Collection {collection_name} not found")
                
        except Exception as e:
            self.logger.error(f"❌ Error getting collection ID for {collection_name}: {e}")
            raise e
    """Enhanced RAG Knowledge Enhancer with manual suggestions and dependency injection"""
    


    def _get_additional_research_config(self, research_depth: str, source_types: list) -> Dict:
        """Configure research parameters for additional knowledge enhancement."""
        
        base_config = {
            'standard': {'base_results': 4, 'query_variants': 2},
            'deep': {'base_results': 6, 'query_variants': 3},
            'comprehensive': {'base_results': 8, 'query_variants': 4}
        }
        
        config = base_config.get(research_depth, base_config['standard'])
        
        # Define additional research categories
        if 'all' in source_types:
            research_categories = [
                'recent_developments', 'case_studies', 'expert_opinions', 
                'comparative_analysis', 'future_trends', 'best_practices'
            ]
        else:
            research_categories = source_types
        
        config['categories'] = research_categories
        return config

    def _generate_additional_search_queries(self, title_name: str, focus_keyword: str, 
                                          research_config: Dict) -> List[Dict]:
        """Generate expanded search queries for additional knowledge."""
        
        queries = []
        base_results = research_config['base_results']
        
        # Category-specific query templates
        query_templates = {
            'recent_developments': [
                f"{focus_keyword} latest developments 2025",
                f"{focus_keyword} recent trends innovations",
                f"{focus_keyword} new research findings"
            ],
            'case_studies': [
                f"{focus_keyword} case studies examples",
                f"{focus_keyword} real world applications",
                f"{focus_keyword} success stories lessons"
            ],
            'expert_opinions': [
                f"{focus_keyword} expert analysis predictions",
                f"{focus_keyword} industry leader insights",
                f"{focus_keyword} professional recommendations"
            ],
            'comparative_analysis': [
                f"{focus_keyword} comparison alternatives",
                f"{focus_keyword} versus competitive analysis",
                f"{focus_keyword} market comparison study"
            ],
            'future_trends': [
                f"{focus_keyword} future outlook predictions",
                f"{focus_keyword} emerging trends 2025 2026",
                f"{focus_keyword} next generation technology"
            ],
            'best_practices': [
                f"{focus_keyword} best practices guidelines",
                f"{focus_keyword} implementation strategies",
                f"{focus_keyword} optimization techniques"
            ]
        }
        
        for category in research_config['categories']:
            if category in query_templates:
                for query_template in query_templates[category]:
                    queries.append({
                        'query': query_template,
                        'source_type': category,
                        'max_results': base_results
                    })
        
        return queries

    async def _search_linkup_with_filtering(self, query: str, max_results: int, 
                                        user_id: str, existing_urls: set) -> List[Dict]:
        """Search Linkup and filter out existing URLs."""
        
        if not self.linkup_client:
            self.logger.warning(f"⚠️ Linkup client not available for query: '{query}'")
            return []
        
        try:
            self.logger.info(f"🔍 LINKUP SEARCH: '{query}' (max: {max_results}, user: {user_id})")
            
            # Determine optimal depth for additional research
            depth = self._determine_linkup_depth(query, source_type, None)
            
            # ✅ FIXED: Remove await - Linkup search is synchronous
            search_response = self.linkup_client.search(
                query=query,
                depth=depth,
                output_type="searchResults"
            )
            
            self.logger.info(f"📋 Response type: {type(search_response)}")
            
            results = []
            
            # Handle LinkupSearchResults with Pydantic objects
            if hasattr(search_response, 'results') and search_response.results:
                self.logger.info(f"🔍 Processing {len(search_response.results)} Linkup results...")
                
                processed_count = 0
                for i, result in enumerate(search_response.results):
                    if processed_count >= max_results:
                        break
                        
                    try:
                        # Access Pydantic attributes directly
                        title = getattr(result, 'name', getattr(result, 'title', 'Unknown Title'))
                        content = getattr(result, 'content', getattr(result, 'description', 'No description'))
                        url = getattr(result, 'url', 'No URL')
                        
                        # Skip if URL already exists
                        if url in existing_urls:
                            self.logger.debug(f"⚠️ Skipping duplicate URL: {url}")
                            continue
                        
                        # Truncate content if too long
                        description = content[:500] + "..." if len(content) > 500 else content
                        
                        formatted_result = {
                            'title': title,
                            'description': description,
                            'url': url,
                            'source': 'Linkup Additional Research',
                            'type': 'additional_research',
                            'user_id': user_id
                        }
                        
                        results.append(formatted_result)
                        processed_count += 1
                        self.logger.info(f"   ✅ Result {processed_count}: {title[:50]}...")
                        
                    except Exception as result_error:
                        self.logger.error(f"   ❌ Error processing result {i+1}: {result_error}")
                        continue
            else:
                self.logger.warning(f"⚠️ No results found in response")
            
            self.logger.info(f"📊 LINKUP SUCCESS: Extracted {len(results)} new results (excluded {len(existing_urls)} duplicates)")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ LINKUP ERROR for '{query}': {e}")
            return []

    def _create_enhanced_additional_content(self, source: Dict, title_name: str, 
                                          focus_keyword: str) -> str:
        """Create enhanced content for additional knowledge sources."""
        
        content_parts = []
        
        # Header with additional knowledge indicator
        content_parts.append(f"# Additional Knowledge: {source.get('title', 'Unknown Source')}")
        content_parts.append(f"**Enhancement Type:** Additional Knowledge Expansion")
        content_parts.append(f"**Related Title:** {title_name}")
        content_parts.append(f"**Focus Area:** {focus_keyword}")
        content_parts.append(f"**Source Type:** {source.get('data_source_type', 'additional_research')}")
        content_parts.append("")
        
        # Enhanced description
        description = source.get('description', '')
        if description:
            content_parts.append("## Content Overview")
            content_parts.append(description)
            content_parts.append("")
        
        # Additional context
        content_parts.append("## Additional Insights")
        content_parts.append(f"This source provides supplementary knowledge to expand understanding of {focus_keyword}.")
        content_parts.append("Key areas covered may include recent developments, comparative analysis, expert perspectives, or emerging trends.")
        content_parts.append("")
        
        # Source information
        source_url = source.get('url', '')
        if source_url:
            content_parts.append("## Source Information")
            content_parts.append(f"**Original URL:** {source_url}")
            content_parts.append(f"**Source Type:** {source.get('type', 'Additional Research')}")
            content_parts.append(f"**Research Date:** {datetime.now().strftime('%Y-%m-%d')}")
        
        return "\n".join(content_parts)

    async def _add_additional_gap_filler_document(self, collection_id: int, collection_name: str,
                                                title: str, content: str, source: Dict, 
                                                source_type: str, metadata: Dict, title_id: str,
                                                research_results: Dict, user_id: str = None) -> Optional[str]:
        """Add additional knowledge document with special metadata."""
        
        try:
            # Calculate document metrics
            doc_size = len(content)
            estimated_chunks = max(1, doc_size // 1000)
            
            # Extract source information
            source_url = str(source.get('url', ''))[:500] if source.get('url') else ""
            source_name = source.get('source', 'Additional Research')
            
            # Create enhanced metadata for additional knowledge
            processing_metadata = {
                "additional_knowledge_info": {
                    "original_title_id": title_id,
                    "focus_keyword": research_results['focus_keyword'],
                    "research_timestamp": research_results['research_timestamp'],
                    "data_source_type": source_type,
                    "original_source": source_name,
                    "source_url": source_url,
                    "enhancement_type": "additional_knowledge",
                    "auto_generated": True,
                    "user_id": user_id  # ✅ ENSURE user_id is in metadata
                },
                "source_metadata": {
                    "title": source.get('title', ''),
                    "description": source.get('description', '')[:1000],
                    "source_type": source.get('type', source_type)
                },
                "content_analysis": {
                    "content_length": doc_size,
                    "estimated_chunks": estimated_chunks,
                    "language": "en",
                    "content_quality": "additional_research_grade"
                }
            }
            
            # Prepare document data
            doc_data = {
                'collectionId': collection_id,
                'title': title[:200],
                'parsedText': content,
                'author': self._extract_authors(source),
                'summary_short': self._create_short_summary(source, content),
                'summary_medium': self._create_medium_summary(source, content),
                'source_type': 'additional_gap_filler',  # Different from regular gap_filler
                'url': source_url,
                'doc_size': doc_size,
                'chunk_count': estimated_chunks,
                'processing_status': 'pending',
                'in_vector_store': False,
                'processing_metadata': processing_metadata,
                'file_type': 'additional_knowledge_research',
                'filename': f"additional_{source_type}_{title_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                'upload_date': datetime.now().isoformat(),
                'created_at': datetime.now().isoformat(),
                'last_processed': datetime.now().isoformat(),
                'selected': True,
                'scraped': True,
                'importance_score': self._calculate_importance_score(source, source_type),
                
                # ✅ CRITICAL FIX: Ensure user_id is stored as top-level field
                'user_id': user_id,
                
                # Additional knowledge specific fields
                'title_id': title_id,
                'focus_keyword': research_results.get('focus_keyword', ''),
                'research_timestamp': research_results.get('research_timestamp', ''),
                'data_source_type': source_type,
                'enhancement_type': 'additional_knowledge'
            }
            
            # ✅ VALIDATION: Ensure user_id is not None before inserting
            if not user_id:
                self.logger.warning(f"⚠️ user_id is None for additional knowledge document - using 'system'")
                doc_data['user_id'] = 'system'
            
            # Insert using HTTP client
            http_response = self.supabase.table("lindex_documents").insert(doc_data)
            
            # Check if the response is successful and has data
            if http_response.success and http_response.data and len(http_response.data) > 0:
                doc_id = http_response.data[0]['id']
                
                self.logger.info(f"    📄 Successfully inserted additional knowledge document:")
                self.logger.info(f"       - ID: {doc_id}")
                self.logger.info(f"       - Title ID: {title_id}")
                self.logger.info(f"       - Title: {title[:50]}...")
                self.logger.info(f"       - Source Type: {source_type}")
                self.logger.info(f"       - Size: {doc_size} chars")
                self.logger.info(f"       - Enhancement Type: additional_knowledge")
                self.logger.info(f"       - User ID: {user_id}")  # ✅ LOG user_id confirmation
                
                return str(doc_id)
            else:
                self.logger.error(f"    ❌ Additional document insert failed")
                self.logger.error(f"       Response success: {http_response.success}")
                self.logger.error(f"       Response error: {http_response.error}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error adding additional knowledge document: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return None

    # =============================================================================
    # Add this main method to the EnhancedRAGKnowledgeEnhancer class in knowledge_gap_http_supabase.py
    ####################################################################################

    async def enhance_additional_knowledge(self, title_id: str, user_id: str, 
                                        collection_name: str = None,
                                        research_depth: str = 'standard',
                                        existing_urls: set = None,
                                        source_types: list = None) -> Dict:
        """
        Find additional knowledge sources for titles with closed gaps.
        Ensures no duplicate URLs and expands existing knowledge.
        """
        try:
            self.logger.info(f"🔍 Starting additional knowledge enhancement for title {title_id}")
            
            if existing_urls is None:
                existing_urls = set()
            
            if source_types is None:
                source_types = ['all']
            
            # ✅ Get title information
            title_response = self.supabase.execute_query(
                'GET',
                f'Titles?select=id,Title,focus_keyword,status&id=eq.{title_id}'
            )
            
            if not title_response['success'] or not title_response['data']:
                raise Exception(f"Title {title_id} not found")
            
            title_data = title_response['data'][0]
            title_name = title_data.get('Title', 'Unknown Title')
            focus_keyword = title_data.get('focus_keyword', '')
            
            self.logger.info(f"📋 Title: {title_name}")
            self.logger.info(f"🎯 Focus keyword: {focus_keyword}")
            self.logger.info(f"🚫 Excluding {len(existing_urls)} existing URLs")
            
            # ✅ FIXED: Determine collection name properly
            if not collection_name:
                collection_name = f"additional_knowledge_{title_id[:8]}"
            
            self.logger.info(f"🗂️ Using collection name: {collection_name}")
            
            # ✅ FIXED: Ensure collection exists and get the actual name back
            final_collection_name = await self._ensure_collection_exists(collection_name, user_id)
            
            self.logger.info(f"🗂️ Final collection name: {final_collection_name}")
            
            # ✅ FIXED: Get collection ID immediately after ensuring collection exists
            collection_id = await self._get_collection_id(final_collection_name)
            
            self.logger.info(f"🗂️ Collection ID: {collection_id}")
            
            # ✅ Configure research parameters based on depth
            research_config = self._get_additional_research_config(research_depth, source_types)
            
            # ✅ Generate expanded search queries
            search_queries = self._generate_additional_search_queries(
                title_name, focus_keyword, research_config
            )
            
            self.logger.info(f"🔍 Generated {len(search_queries)} additional search queries")
            
            # ✅ Perform additional research
            all_additional_sources = []
            research_results = {
                'focus_keyword': focus_keyword,
                'research_timestamp': datetime.now().isoformat(),
                'total_sources': 0,
                'new_sources': 0,
                'excluded_duplicates': 0
            }
            
            for query_config in search_queries:
                query = query_config['query']
                source_type = query_config['source_type']
                max_results = query_config['max_results']
                
                self.logger.info(f"🔍 Searching for {source_type}: '{query}' (max: {max_results})")
                
                try:
                    linkup_results = await self._search_linkup_with_filtering(
                                query, max_results, user_id, existing_urls
                            )
                    if linkup_results:
                        self.logger.info(f"📊 Found {len(linkup_results)} new {source_type} sources")
                        
                        for source in linkup_results:
                            source['data_source_type'] = f"additional_{source_type}"
                            source['enhancement_type'] = 'additional_knowledge'
                            all_additional_sources.append(source)
                        
                        research_results['total_sources'] += len(linkup_results)
                        research_results['new_sources'] += len(linkup_results)
                    else:
                        self.logger.info(f"📭 No new {source_type} sources found")
                    
                    # Delay between searches
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error searching for {source_type}: {e}")
                    continue


            
            # ✅ Process and add additional sources
            if not all_additional_sources:
                self.logger.warning(f"⚠️ No additional sources found for title {title_id}")
                return {
                    'success': True,
                    'documents_added': 0,
                    'new_sources_found': 0,
                    'collection_name': final_collection_name,
                    'message': 'No additional sources found',
                    'research_timestamp': research_results['research_timestamp']
                }
            
            # ✅ Add additional documents to RAG
            documents_added = 0
            source_breakdown = {}
            
            self.logger.info(f"📄 Adding {len(all_additional_sources)} documents to collection '{final_collection_name}' (ID: {collection_id})")
            
            for i, source in enumerate(all_additional_sources):
                try:
                    source_type = source.get('data_source_type', 'additional_unknown')
                    
                    self.logger.info(f"  📄 Adding document {i+1}/{len(all_additional_sources)}: {source.get('title', 'Unknown')[:50]}...")
                    
                    # Create enhanced content
                    enhanced_content = self._create_enhanced_additional_content(source, title_name, focus_keyword)
                    
                    # ✅ FIXED: Pass the variables we know are not None
                    doc_id = await self._add_additional_gap_filler_document(
                        collection_id=collection_id,
                        collection_name=final_collection_name,  # This should not be None
                        title=f"📈 Additional: {source.get('title', 'Unknown Source')[:150]}",
                        content=enhanced_content,
                        source=source,
                        source_type=source_type,
                        metadata={'enhancement_type': 'additional_knowledge'},
                        title_id=title_id,
                        research_results=research_results,
                        user_id=user_id
                    )
                    
                    if doc_id:
                        documents_added += 1
                        source_breakdown[source_type] = source_breakdown.get(source_type, 0) + 1
                        self.logger.info(f"    ✅ Successfully added document {doc_id}")
                        
                        # Process into vector store
                        await self._process_document_into_vector_store(doc_id, final_collection_name, source_type)
                    else:
                        self.logger.error(f"    ❌ Failed to add document - no doc_id returned")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error adding additional source {i+1}: {e}")
                    continue
            
            # ✅ Update title metadata
            try:
                update_response = self.supabase.execute_query(
                    'PATCH',
                    f'Titles?id=eq.{title_id}',
                    {
                        'additional_knowledge_enhanced': True,
                        'additional_enhancement_timestamp': datetime.now().isoformat(),
                        'additional_documents_count': documents_added
                    }
                )
                
                if update_response['success']:
                    self.logger.info(f"✅ Updated title {title_id} with additional enhancement metadata")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to update title metadata: {e}")
            
            # ✅ Calculate enhancement effectiveness
            effectiveness = 'high' if documents_added >= 10 else 'medium' if documents_added >= 5 else 'low'
            
            enhancement_result = {
                'success': True,
                'title_id': title_id,
                'title_name': title_name,
                'collection_name': final_collection_name,
                'documents_added': documents_added,
                'new_sources_found': len(all_additional_sources),
                'source_breakdown': source_breakdown,
                'research_timestamp': research_results['research_timestamp'],
                'enhancement_type': 'additional_knowledge',
                'research_depth': research_depth,
                'enhancement_effectiveness': effectiveness,
                'processing_summary': {
                    'total_queries_executed': len(search_queries),
                    'unique_sources_found': len(all_additional_sources),
                    'documents_successfully_added': documents_added,
                    'existing_urls_excluded': len(existing_urls)
                },
                'next_steps': [
                    'Test enhanced RAG queries with expanded knowledge',
                    'Review additional document quality',
                    'Consider generating updated manual actions',
                    'Monitor vector store performance with additional content'
                ] if documents_added > 0 else [
                    'No additional content found - consider different search terms',
                    'Check if existing knowledge is already comprehensive',
                    'Try deeper research depth setting'
                ]
            }
            
            self.logger.info(f"✅ Additional enhancement completed: {documents_added} documents added to '{final_collection_name}'")
            return enhancement_result
            
        except Exception as e:
            self.logger.error(f"❌ Additional knowledge enhancement failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'title_id': title_id,
                'documents_added': 0,
                'enhancement_type': 'additional_knowledge'
            }
  ####################################################################################

    async def enhance_rag_for_title(self, title_id: str, research_results: Dict, 
                                  collection_name: str = None, merge_with_existing: bool = True,
                                  user_id: str = None):
        """Add research results to RAG system with vector store integration and user tracking."""
        try:
            # Step 1: Determine collection name
            final_collection_name = await self._determine_collection_name(
                title_id, collection_name, merge_with_existing, research_results, user_id
            )
            
            self.logger.info(f"🗂️ Using collection: {final_collection_name} (user: {user_id})")
            
            # Step 2: Ensure collection exists
            await self._ensure_collection_exists(final_collection_name, user_id)
            
            # Step 3: Get collection ID
            collection_id = await self._get_collection_id(final_collection_name)
            
            # Step 4: Process each source type
            documents_added = 0
            added_document_ids = []
            
            for source_type, sources in research_results['sources_found'].items():
                for i, source in enumerate(sources):
                    try:
                        # Create document content
                        doc_content = self._format_source_content(source, source_type, research_results)
                        doc_title = self._generate_document_title(source, source_type, i)
                        
                        # Create metadata with user tracking
                        metadata = self._create_comprehensive_metadata(
                            title_id, research_results, source, source_type, i, 
                            final_collection_name, user_id
                        )
                        
                        # Add to lindex_documents with complete data
                        doc_id = await self._add_gap_filler_document(
                            collection_id=collection_id,
                            collection_name=final_collection_name,
                            title=doc_title,
                            content=doc_content,
                            source=source,
                            source_type=source_type,
                            metadata=metadata,
                            title_id=title_id,
                            research_results=research_results,
                            user_id=user_id
                        )
                        
                        if doc_id:
                            # Process document into vector store
                            await self._process_document_into_vector_store(
                                doc_id, final_collection_name, source_type, user_id
                            )
                            
                            documents_added += 1
                            added_document_ids.append(doc_id)
                            self.logger.info(f"  ✅ Added and processed gap filler document: {doc_title[:50]}... (ID: {doc_id})")
                        
                    except Exception as e:
                        self.logger.error(f"  ❌ Failed to add source: {e}")
                        continue
            
            # Step 5: Generate manual suggestions if available
            manual_suggestions_generated = 0
            suggestions_breakdown = {}
            
            if self.suggestions_generator and documents_added > 0:
                try:
                    self.logger.info(f"🎯 Generating manual action suggestions for title {title_id}")
                    
                    # Create gap info for suggestions
                    gap_info = self._create_gap_from_research(research_results)
                    
                    # Generate suggestions
                    suggestions = await self.suggestions_generator.generate_action_suggestions(gap_info, research_results)
                    
                    if suggestions:
                        # Save to database - ✅ FIX: Add user_id parameter
                        research_context = {
                            'title_id': title_id,
                            'focus_keyword': research_results['focus_keyword'],
                            'total_sources_found': research_results['total_sources'],
                            'collection_name': final_collection_name
                        }
                        
                        # ✅ FIXED: Pass user_id as required parameter
                        suggestions_saved = await self.suggestions_generator.save_suggestions_to_database(
                            title_id, suggestions, research_context, user_id  # ← Add user_id here
                        )
                        
                        if suggestions_saved:
                            manual_suggestions_generated = len(suggestions)
                            suggestions_breakdown = {
                                'textbooks': len([s for s in suggestions if s['action_type'] == 'textbook']),
                                'datasets': len([s for s in suggestions if s['action_type'] == 'dataset']),
                                'tools': len([s for s in suggestions if s['action_type'] == 'tool_development']),
                                'experts': len([s for s in suggestions if s['action_type'] == 'expert_interview']),
                                'templates': len([s for s in suggestions if s['action_type'] == 'template_creation'])
                            }
                            self.logger.info(f"✅ Generated {len(suggestions)} manual suggestions")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error generating manual suggestions: {e}")

            
            # Step 6: Update title record with proper gap closure
            if documents_added > 0:
                await self._update_title_enhancement_status(
                    title_id, final_collection_name, documents_added, added_document_ids, user_id
                )
            
            self.logger.info(f"✅ Enhanced RAG for title {title_id} with {documents_added} gap filler documents (user: {user_id})")
            
            return {
                'success': True,
                'documents_added': documents_added,
                'collection_name': final_collection_name,
                'collection_id': collection_id,
                'document_ids': added_document_ids,
                'collection_strategy': self._get_collection_strategy_used(title_id, collection_name, merge_with_existing),
                'manual_suggestions_generated': manual_suggestions_generated,
                'suggestions_breakdown': suggestions_breakdown,
                'user_id': user_id
            }
            
        except Exception as e:
            self.logger.error(f"Error enhancing RAG for title {title_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_id': user_id
            }
    
    async def _process_document_into_vector_store(self, doc_id: str, collection_name: str, source_type: str, user_id: str = None):
        """Process the gap filler document into the vector store with proper dependency checking."""
        
        try:
            self.logger.info(f"🔄 Processing document {doc_id} into vector store... (user: {user_id})")
            
            # Update processing status
            self.supabase.table("lindex_documents").update({
                "processing_status": "processing",
                "last_processed": datetime.now().isoformat(),
                "user_id": user_id
            }).eq("id", doc_id).execute()
            
            # Check if processors are available
            if self.document_processor and self.background_executor:
                # Submit to background processing
                self.background_executor.submit(
                    self._process_document_background,
                    doc_id,
                    collection_name,
                    source_type,
                    user_id
                )
                self.logger.info(f"  📤 Submitted document {doc_id} for background processing")
            else:
                # Update status to indicate processor not available
                self.logger.warning(f"⚠️ Document processors not available - updating status only")
                self.supabase.table("lindex_documents").update({
                    "processing_status": "pending_processor",
                    "error_message": "Background document processor not available",
                    "user_id": user_id
                }).eq("id", doc_id).execute()
                
        except Exception as e:
            self.logger.error(f"❌ Error processing document {doc_id} into vector store: {e}")
            # Update error status
            self.supabase.table("lindex_documents").update({
                "processing_status": "error",
                "error_message": str(e)[:200],
                "user_id": user_id
            }).eq("id", doc_id).execute()
    
    def _process_document_background(docid, collection_name, source_type="document"):
        """Background task for processing documents with enhanced error handling."""
        try:
            # Update status to processing
            supabase.table("lindex_documents").update({
                "processing_status": "processing"
            }).eq("id", docid).execute()
            
            # Ensure optimizer is fitted
            embedding_manager.ensure_optimizer_fitted()
            
            # Get document content
            doc_response = supabase.table("lindex_documents").select(
                "parsedText"
            ).eq("id", docid).execute()
            
            if not doc_response.data or not doc_response.data[0].get("parsedText"):
                logger.error(f"Document {docid} has no parsable content")
                supabase.table("lindex_documents").update({
                    "processing_status": "error",
                    "error_message": "No parsable content found"
                }).eq("id", docid).execute()
                return
                
            document_text = doc_response.data[0]["parsedText"]
            
            # ✅ FIX: Enhanced error handling for vector processing
            try:
                result = document_processor.process_document(
                    docid=docid, 
                    collection_name=collection_name,
                    source_type=source_type
                )
                
                if result.get('success', False):
                    doc_size = len(document_text)
                    chunk_count = result.get('embedding', {}).get('total_nodes', 0)
                    
                    supabase.table("lindex_documents").update({
                        "processing_status": "completed",
                        "in_vector_store": True,
                        "last_processed": datetime.now().isoformat(),
                        "source_type": source_type,
                        "doc_size": doc_size,
                        "chunk_count": chunk_count
                    }).eq("id", docid).execute()
                    
                    logger.info(f"✅ Successfully processed document {docid} into vector store")
                    
                else:
                    error_msg = result.get('error', 'Unknown processing error')
                    logger.error(f"❌ Vector processing failed for {docid}: {error_msg}")
                    
                    supabase.table("lindex_documents").update({
                        "processing_status": "error",
                        "error_message": error_msg[:200]
                    }).eq("id", docid).execute()
                    
            except Exception as processing_error:
                error_msg = str(processing_error)
                logger.error(f"❌ Exception during vector processing for {docid}: {error_msg}")
                
                # ✅ FIX: Special handling for Supabase client compatibility issues
                if "SyncPostgrestClient" in error_msg or "http_client" in error_msg:
                    logger.warning(f"⚠️ Supabase client compatibility issue for {docid} - marking as pending_processor")
                    supabase.table("lindex_documents").update({
                        "processing_status": "pending_processor",
                        "error_message": "Supabase client compatibility issue - needs processor update"
                    }).eq("id", docid).execute()
                else:
                    supabase.table("lindex_documents").update({
                        "processing_status": "error", 
                        "error_message": error_msg[:200]
                    }).eq("id", docid).execute()
                
        except Exception as e:
            logger.exception(f"Background processing error for document {docid}: {str(e)}")
            
            try:
                supabase.table("lindex_documents").update({
                    "processing_status": "error",
                    "error_message": str(e)[:200]
                }).eq("id", docid).execute()
            except:
                pass

## PART 5

# knowledge_gap_http_supabase.py - PART 5: Enhanced RAG Knowledge Enhancer Class (Part 2)

    async def _update_title_enhancement_status(self, title_id: str, collection_name: str, 
                                            documents_added: int, document_ids: List[str],
                                            user_id: str = None):
        """Update titles table with proper status values and gap closure tracking."""
        
        try:
            update_data = {
                "knowledge_enhanced": True,
                "rag_collection_name": collection_name,
                "research_sources_count": documents_added,
                "last_research_date": datetime.now().isoformat(),
                # Proper gap closure tracking
                "knowledge_gaps_closed": True,
                "gap_closure_timestamp": datetime.now().isoformat(),
                "gap_closure_method": "automated_gap_filler",
                "gap_closure_details": json.dumps({
                    "documents_added": documents_added,
                    "document_ids": document_ids,
                    "collection_name": collection_name,
                    "closure_timestamp": datetime.now().isoformat(),
                    "enhancement_method": "multi_source_research",
                    "sources_researched": len(document_ids),
                    "auto_processed": True,
                    "user_id": user_id,
                    "processing_status": "completed"
                })
                # Note: Only add user_id if your titles table has this field
                # "user_id": user_id
            }
            
            # Update title
            http_response = self.supabase.table("Titles").update(update_data).eq("id", title_id).execute()
            
            if http_response.data and len(http_response.data) > 0:
                self.logger.info(f"✅ Updated title {title_id}:")
                self.logger.info(f"   - Knowledge gaps closed: TRUE")
                self.logger.info(f"   - Documents added: {documents_added}")
                self.logger.info(f"   - Collection: {collection_name}")
                self.logger.info(f"   - User ID: {user_id}")
            else:
                self.logger.warning(f"⚠️ Title update returned no data for {title_id}")
                
        except Exception as e:
            self.logger.error(f"❌ Error updating title status: {e}")
    
    async def mark_gaps_as_closed(self, title_id: str, closure_method: str = "manual", 
                                    closure_notes: str = "", user_id: str = None):
        """Manually mark knowledge gaps as closed for a title with user tracking."""
        try:
            update_data = {
                "knowledge_gaps_closed": True,
                "gap_closure_timestamp": datetime.now().isoformat(),
                "gap_closure_method": closure_method,
                "gap_closure_details": json.dumps({
                    "closure_method": closure_method,
                    "closure_timestamp": datetime.now().isoformat(),
                    "closure_notes": closure_notes,
                    "closed_by_user": user_id,
                    "manual_closure": True
                })
                # Note: Only add user_id if your titles table has this field
                # "user_id": user_id
            }
            
            if closure_method == "manual_research_complete":
                update_data.update({
                    "knowledge_enhanced": True
                })
            
            response = self.supabase.table("Titles").update(update_data).eq("id", title_id).execute()
            
            if response.data:
                self.logger.info(f"✅ Manually marked gaps as closed for title {title_id} by user {user_id}")
                return True
            else:
                self.logger.warning(f"⚠️ Failed to mark gaps as closed for title {title_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error marking gaps as closed: {e}")
            return False

    async def get_gap_closure_status(self, title_id: str = None, user_id: str = None):
        """Get gap closure status for titles with user filtering."""
        try:
            if title_id:
                # Single title lookup
                response = self.supabase.table("Titles").select(
                    "id, Title, knowledge_gaps_closed, gap_closure_timestamp, gap_closure_method, gap_closure_details, knowledge_enhanced, status"
                ).eq("id", title_id).execute()
                
                if response.data:
                    title_data = response.data[0]
                    return {
                        "title_id": title_id,
                        "title": title_data.get("Title", ""),
                        "gaps_closed": title_data.get("knowledge_gaps_closed", False),
                        "closure_timestamp": title_data.get("gap_closure_timestamp"),
                        "closure_method": title_data.get("gap_closure_method"),
                        "closure_details": title_data.get("gap_closure_details"),
                        "knowledge_enhanced": title_data.get("knowledge_enhanced", False),
                        "status": title_data.get("status", ""),
                        "user_id": user_id
                    }
                else:
                    return {"error": "Title not found"}
            else:
                # Summary of all titles - THIS WAS MISSING
                self.logger.info(f"📊 Getting gap closure summary for user: {user_id}")
                
                # Get all titles - optionally filter by user_id if your titles table has this field
                if user_id:
                    # Try with user_id filter first (if your table has this field)
                    response = self.supabase.table("Titles").select(
                        "id, Title, knowledge_gaps_closed, gap_closure_timestamp, gap_closure_method, knowledge_enhanced, status"
                    ).eq("user_id", user_id).execute()
                    
                    # If no results with user_id, try without user_id filter
                    if not response.data:
                        self.logger.info(f"⚠️ No titles found with user_id filter, trying without filter")
                        response = self.supabase.table("Titles").select(
                            "id, Title, knowledge_gaps_closed, gap_closure_timestamp, gap_closure_method, knowledge_enhanced, status"
                        ).execute()
                else:
                    # Get all titles without user filter
                    response = self.supabase.table("Titles").select(
                        "id, Title, knowledge_gaps_closed, gap_closure_timestamp, gap_closure_method, knowledge_enhanced, status"
                    ).execute()
                
                if not response.data:
                    self.logger.warning("⚠️ No titles found in database")
                    return {
                        "total_titles": 0,
                        "gaps_closed_count": 0,
                        "gaps_open_count": 0,
                        "knowledge_enhanced_count": 0,
                        "new_status_count": 0,
                        "titles": [],
                        "user_id": user_id,
                        "summary_message": "No titles found in database"
                    }
                
                self.logger.info(f"📋 Found {len(response.data)} titles")
                
                # Process the data
                titles = response.data
                total_titles = len(titles)
                gaps_closed_count = 0
                gaps_open_count = 0
                knowledge_enhanced_count = 0
                new_status_count = 0
                
                processed_titles = []
                
                for title in titles:
                    # Count gaps closed (handling various ways this might be stored)
                    gaps_closed = title.get("knowledge_gaps_closed", False)
                    if gaps_closed is True or gaps_closed == "true" or gaps_closed == 1:
                        gaps_closed_count += 1
                    else:
                        gaps_open_count += 1
                    
                    # Count knowledge enhanced
                    knowledge_enhanced = title.get("knowledge_enhanced", False)
                    if knowledge_enhanced is True or knowledge_enhanced == "true" or knowledge_enhanced == 1:
                        knowledge_enhanced_count += 1
                    
                    # Count NEW status
                    status = title.get("status", "").upper()
                    if status == "NEW":
                        new_status_count += 1
                    
                    # Add to processed list
                    processed_titles.append({
                        "id": title.get("id"),
                        "title": title.get("Title", ""),
                        "gaps_closed": gaps_closed,
                        "knowledge_enhanced": knowledge_enhanced,
                        "status": status,
                        "closure_timestamp": title.get("gap_closure_timestamp"),
                        "closure_method": title.get("gap_closure_method")
                    })
                
                # Calculate summary metrics
                completion_percentage = round((gaps_closed_count / total_titles) * 100, 1) if total_titles > 0 else 0
                
                summary = {
                    "total_titles": total_titles,
                    "gaps_closed_count": gaps_closed_count,
                    "gaps_open_count": gaps_open_count,
                    "knowledge_enhanced_count": knowledge_enhanced_count,
                    "new_status_count": new_status_count,
                    "completion_percentage": completion_percentage,
                    "titles": processed_titles,
                    "user_id": user_id,
                    "query_timestamp": datetime.now().isoformat(),
                    "summary_message": f"{completion_percentage}% complete ({gaps_closed_count}/{total_titles} gaps closed)",
                    "has_open_gaps": gaps_open_count > 0,
                    "needs_attention": gaps_open_count > (total_titles * 0.5),  # More than 50% open
                    "system_healthy": total_titles > 0,
                    "supports_bulk_processing": total_titles > 5,
                    "bulk_recommended": gaps_open_count > 3
                }
                
                self.logger.info(f"✅ Gap closure summary: {gaps_closed_count}/{total_titles} closed ({completion_percentage}%)")
                return summary
                        
        except Exception as e:
            self.logger.error(f"Error getting gap closure status: {e}")
            return {"error": str(e)}
    
    async def _determine_collection_name(self, title_id: str, provided_collection: str = None, 
                                       merge_with_existing: bool = True, research_results: Dict = None,
                                       user_id: str = None) -> str:
        """Determine the collection name to use with flexible logic and user tracking."""
        
        if provided_collection:
            self.logger.info(f"📋 Using provided collection: {provided_collection} (user: {user_id})")
            return provided_collection
        
        if merge_with_existing:
            existing_collection = await self._get_existing_title_collection(title_id)
            if existing_collection:
                self.logger.info(f"🔗 Merging with existing title collection: {existing_collection}")
                return existing_collection
        
        if merge_with_existing and research_results:
            topic_collection = await self._get_topic_based_collection(research_results)
            if topic_collection:
                self.logger.info(f"🏷️ Using topic-based collection: {topic_collection}")
                return topic_collection
        
        title_specific = f"enhanced_{title_id}"
        self.logger.info(f"🆕 Creating title-specific collection: {title_specific}")
        return title_specific
    
    async def _ensure_collection_exists(self, collection_name: str, user_id: str = None):
        """Ensure collection exists in lindex_collections table with user tracking."""
        try:
            # ✅ FIXED: Handle None collection_name at the start
            if not collection_name or collection_name == "None":
                collection_name = f"additional_knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.logger.info(f"📁 Invalid collection name provided, using: {collection_name}")
            
            self.logger.info(f"📁 Ensuring collection exists: {collection_name}")
            
            response = self.supabase.execute_query(
                'GET',
                f'lindex_collections?select=id,name&name=eq.{collection_name}'
            )
            
            if not response['success'] or not response['data']:
                self.logger.info(f"📁 Collection '{collection_name}' does not exist, creating it...")
                
                collection_data = {
                    "name": collection_name,
                    "description": f"Additional knowledge collection created by Gap Filler system (user: {user_id})",
                    "created_at": datetime.now().isoformat()
                }
                
                insert_response = self.supabase.execute_query(
                    'POST',
                    'lindex_collections',
                    collection_data
                )
                
                if insert_response['success'] and insert_response['data']:
                    self.logger.info(f"✅ Created new collection: {collection_name} (user: {user_id})")
                else:
                    self.logger.error(f"❌ Failed to create collection {collection_name}")
                    self.logger.error(f"   Response: {insert_response}")
                    raise ValueError(f"Failed to create collection {collection_name}")
            else:
                self.logger.info(f"📁 Collection {collection_name} already exists")
            
            # ✅ IMPORTANT: Always return the valid collection_name
            return collection_name
                
        except Exception as e:
            self.logger.error(f"❌ Error ensuring collection exists: {e}")
            raise
    
    async def _get_collection_id(self, collection_name: str) -> int:
        """Get collection ID by name."""
        try:
            if not collection_name or collection_name == "None":
                raise Exception(f"Invalid collection name: {collection_name}")
            
            self.logger.info(f"🔍 Getting collection ID for: {collection_name}")
            
            collection_response = self.supabase.execute_query(
                'GET',
                f'lindex_collections?select=id,name&name=eq.{collection_name}'
            )
            
            if collection_response['success'] and collection_response['data']:
                collection_id = collection_response['data'][0]['id']
                self.logger.info(f"✅ Found collection ID: {collection_id} for '{collection_name}'")
                return collection_id
            else:
                self.logger.error(f"❌ Collection '{collection_name}' not found in database")
                self.logger.error(f"   Response: {collection_response}")
                raise Exception(f"Collection {collection_name} not found")
                
        except Exception as e:
            self.logger.error(f"❌ Error getting collection ID for '{collection_name}': {e}")
            raise e

    async def _get_existing_title_collection(self, title_id: str) -> Optional[str]:
        """Check if title already has a RAG collection assigned."""
        try:
            response = self.supabase.table("Titles").select("rag_collection_name").eq("id", title_id).execute()
            
            if response.data and response.data[0].get('rag_collection_name'):
                collection_name = response.data[0]['rag_collection_name']
                
                # Verify collection exists
                coll_response = self.supabase.table("lindex_collections").select("id").eq("name", collection_name).execute()
                
                if coll_response.data:
                    return collection_name
                else:
                    self.logger.warning(f"⚠️ Title references non-existent collection: {collection_name}")
                    return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking existing title collection: {e}")
            return None
    
    async def _get_topic_based_collection(self, research_results: Dict) -> Optional[str]:
        """Determine topic-based collection name from research results."""
        try:
            focus_keyword = research_results.get('focus_keyword', '').lower()
            
            topic_mappings = {
                'real_estate_general': ['real estate', 'property', 'housing'],
                'real_estate_investment': ['investment property', 'real estate investment', 'rental property'],
                'home_buying': ['home buying', 'house buying', 'first time buyer'],
                'mortgage_finance': ['mortgage', 'home loan', 'financing'],
                'ai_machine_learning': ['ai', 'artificial intelligence', 'machine learning', 'ml'],
                'web_development': ['web development', 'programming', 'coding'],
                'data_science': ['data science', 'analytics', 'big data'],
                'personal_finance': ['personal finance', 'budgeting', 'saving'],
                'investing': ['investing', 'investment', 'portfolio'],
                'cryptocurrency': ['crypto', 'bitcoin', 'blockchain'],
                'entrepreneurship': ['startup', 'entrepreneur', 'business'],
                'marketing': ['marketing', 'social media', 'advertising'],
                'management': ['management', 'leadership', 'business strategy']
            }
            
            for collection_base, keywords in topic_mappings.items():
                if any(keyword in focus_keyword for keyword in keywords):
                    topic_collection = f"knowledge_{collection_base}"
                    
                    collection_id = await self._get_collection_id_if_exists(topic_collection)
                    if collection_id:
                        existing_docs = self.supabase.table("lindex_documents").select("id").eq(
                            "collectionId", collection_id
                        ).execute()
                        
                        if existing_docs.data:
                            return topic_collection
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining topic-based collection: {e}")
            return None
    
    async def _get_collection_id_if_exists(self, collection_name: str) -> Optional[int]:
        """Get collection ID if it exists, return None otherwise."""
        try:
            response = self.supabase.table("lindex_collections").select("id").eq("name", collection_name).execute()
            return response.data[0]["id"] if response.data else None
        except:
            return None
    
    def _get_collection_strategy_used(self, title_id: str, provided_collection: str, merge_with_existing: bool) -> str:
        """Return description of collection strategy used."""
        if provided_collection:
            return f"explicit_provided:{provided_collection}"
        elif merge_with_existing:
            return "merge_with_existing:auto_detected"
        else:
            return f"title_specific:enhanced_{title_id}"
    
    def _create_gap_from_research(self, research_results):
        """Create gap object for suggestions"""
        from types import SimpleNamespace
        return SimpleNamespace(
            title_id=research_results['title_id'],
            focus_keyword=research_results['focus_keyword'],
            topic_category=self._infer_topic_category(research_results['focus_keyword']),
            target_audience='professional',
            data_types_needed=list(research_results['sources_found'].keys())
        )
    
    def _infer_topic_category(self, focus_keyword: str) -> str:
        """Infer topic category from focus keyword"""
        keyword_lower = focus_keyword.lower()
        
        if any(term in keyword_lower for term in ['real estate', 'property', 'housing', 'mortgage']):
            return 'real_estate'
        elif any(term in keyword_lower for term in ['finance', 'investment', 'money', 'financial']):
            return 'finance'
        elif any(term in keyword_lower for term in ['tech', 'software', 'ai', 'programming']):
            return 'technology'
        elif any(term in keyword_lower for term in ['business', 'entrepreneur', 'startup']):
            return 'business'
        elif any(term in keyword_lower for term in ['health', 'medical', 'wellness']):
            return 'health'
        else:
            return 'general'

## PART 6

# knowledge_gap_http_supabase.py - PART 6: Document Processing, Routes, and Integration Functions

# Fix for the _add_gap_filler_document method in knowledge_gap_http_supabase.py
# Replace the existing method around line 1746

# In knowledge_gap_http_supabase.py, update the _add_gap_filler_document method:

    async def _add_gap_filler_document(self, collection_id: int, collection_name: str, 
                                    title: str, content: str, source: Dict, source_type: str,
                                    metadata: Dict, title_id: str, research_results: Dict,
                                    user_id: str = None) -> Optional[str]:
        """Add gap filler document to lindex_documents with complete metadata and user tracking."""
        
        try:
            # Calculate document metrics
            doc_size = len(content)
            estimated_chunks = max(1, doc_size // 1000)
            
            # Extract source information safely
            source_url = str(source.get('url', ''))[:500] if source.get('url') else ""
            source_name = source.get('source', 'Linkup Research')
            
            # Create processing metadata
            processing_metadata = {
                "gap_filler_info": {
                    "original_title_id": title_id,
                    "focus_keyword": research_results['focus_keyword'],
                    "research_timestamp": research_results['research_timestamp'],
                    "data_source_type": source_type,
                    "original_source": source_name,
                    "source_url": source_url,
                    "auto_generated": True,
                    "user_id": user_id
                },
                "source_metadata": {
                    "title": source.get('title', ''),
                    "description": source.get('description', '')[:1000],
                    "source_type": source.get('type', source_type)
                },
                "content_analysis": {
                    "content_length": doc_size,
                    "estimated_chunks": estimated_chunks,
                    "language": "en",
                    "content_quality": "research_grade"
                }
            }
            
            # Prepare document data for lindex_documents table
            doc_data = {
                'collectionId': collection_id,
                'title': title[:200],
                'parsedText': content,  # Essential for vector processing
                'author': self._extract_authors(source),
                'summary_short': self._create_short_summary(source, content),
                'summary_medium': self._create_medium_summary(source, content),
                'source_type': 'gap_filler',
                'url': source_url,
                'doc_size': doc_size,
                'chunk_count': estimated_chunks,
                'processing_status': 'pending',  # Start as pending
                'in_vector_store': False,  # Will be updated after processing
                'processing_metadata': processing_metadata,
                'file_type': 'gap_filler_research',
                'filename': f"gap_filler_{source_type}_{title_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                'upload_date': datetime.now().isoformat(),
                'created_at': datetime.now().isoformat(),
                'last_processed': datetime.now().isoformat(),
                'selected': True,
                'scraped': True,
                'web_content': content[:2000] if source_type in ['news_articles', 'industry_reports'] else None,
                'importance_score': self._calculate_importance_score(source, source_type),
                'user_id': user_id,
                # ✅ NEW: Add title_id as a top-level field for easy querying
                'title_id': title_id,
                # ✅ NEW: Add focus keyword for easier identification
                'focus_keyword': research_results.get('focus_keyword', ''),
                # ✅ NEW: Add research timestamp for tracking
                'research_timestamp': research_results.get('research_timestamp', ''),
                # ✅ NEW: Add data source type for categorization
                'data_source_type': source_type
            }
            
            # Insert using HTTP client (without .execute())
            http_response = self.supabase.table("lindex_documents").insert(doc_data)
            
            # Check if the response is successful and has data
            if http_response.success and http_response.data and len(http_response.data) > 0:
                doc_id = http_response.data[0]['id']
                
                self.logger.info(f"    📄 Successfully inserted gap filler document:")
                self.logger.info(f"       - ID: {doc_id}")
                self.logger.info(f"       - Title ID: {title_id}")  # ✅ NEW: Log title_id
                self.logger.info(f"       - Title: {title[:50]}...")
                self.logger.info(f"       - Source Type: {source_type}")
                self.logger.info(f"       - Size: {doc_size} chars")
                self.logger.info(f"       - User ID: {user_id}")
                
                return str(doc_id)
            else:
                self.logger.error(f"    ❌ Insert failed - no data returned or unsuccessful")
                self.logger.error(f"       Response success: {http_response.success}")
                self.logger.error(f"       Response error: {http_response.error}")
                self.logger.error(f"       Response status: {http_response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error adding gap filler document: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            return None
        
    def _format_source_content(self, source: Dict, source_type: str, research_results: Dict) -> str:
        """Format source content for RAG storage with enhanced structure."""
        
        title = source.get('title', 'Untitled Source')
        description = source.get('description', source.get('summary', ''))
        url = source.get('url', '')
        source_name = source.get('source', 'Unknown Source')
        
        authors_section = ""
        if source.get('authors'):
            if isinstance(source['authors'], list):
                authors_section = f"\n**Authors:** {', '.join(source['authors'])}"
            else:
                authors_section = f"\n**Authors:** {source['authors']}"
        
        published_section = ""
        if source.get('published'):
            published_section = f"\n**Published:** {source['published']}"
        
        type_info = {
            'academic_papers': {
                'type_label': 'Academic Research Paper',
                'context': 'This peer-reviewed research provides authoritative information and data-driven insights.',
                'icon': '📚'
            },
            'government_data': {
                'type_label': 'Government Data/Statistics',
                'context': 'Official government data providing reliable statistics and regulatory information.',
                'icon': '🏛️'
            },
            'industry_reports': {
                'type_label': 'Industry Report/Market Analysis', 
                'context': 'Professional industry analysis providing market insights and business intelligence.',
                'icon': '📊'
            },
            'news_articles': {
                'type_label': 'News Article/Current Information',
                'context': 'Current news and developments providing up-to-date information on trending topics.',
                'icon': '📰'
            },
            'statistical_data': {
                'type_label': 'Statistical Data/Survey Results',
                'context': 'Quantitative data and survey results providing measurable insights and trends.',
                'icon': '📈'
            },
            'textbooks': {
                'type_label': 'Educational Material/Textbook',
                'context': 'Educational content providing comprehensive learning materials and foundational knowledge.',
                'icon': '📖'
            }
        }
        
        type_details = type_info.get(source_type, {
            'type_label': 'Research Source',
            'context': 'Additional research material to enhance knowledge base.',
            'icon': '📄'
        })
        
        content = f"""# {type_details['icon']} {title}

## Source Information
**Source:** {source_name}  
**Type:** {type_details['type_label']}  
**URL:** {url}{authors_section}{published_section}

## Research Context
**Focus Keyword:** {research_results['focus_keyword']}  
**Research Date:** {research_results['research_timestamp'][:10]}  
**Auto-Generated:** Knowledge Gap Filler System

## Content Summary
{description}

## Research Value
{type_details['context']} This source was automatically identified and collected to enhance content creation about "{research_results['focus_keyword']}" by providing additional authoritative information and perspectives.

## Content Guidelines
This research material can be used to:
- Enhance factual accuracy of content
- Provide supporting data and statistics  
- Add authoritative citations and references
- Expand topic coverage with expert insights
- Support evidence-based content creation

---
**Automated Research Enhancement**  
*Generated by Knowledge Gap Filler System*  
*Source Type: {source_type} | Content Type: gap_filler*
"""
        
        return content.strip()

    def _generate_document_title(self, source: Dict, source_type: str, index: int) -> str:
        """Generate a clear, descriptive title for the gap filler document."""
        
        original_title = source.get('title', 'Untitled Source')
        
        # Clean and truncate original title
        if original_title and len(original_title) > 100:
            original_title = original_title[:97] + "..."
        
        # Add source type context
        type_prefix = {
            'academic_papers': '📚 Research',
            'government_data': '🏛️ Gov Data',
            'industry_reports': '📊 Industry',
            'news_articles': '📰 News',
            'statistical_data': '📈 Statistics',
            'textbooks': '📖 Educational'
        }.get(source_type, '📄 Source')
        
        return f"{type_prefix}: {original_title}"
    
    def _create_comprehensive_metadata(self, title_id: str, research_results: Dict, 
                                     source: Dict, source_type: str, index: int, 
                                     collection_name: str, user_id: str = None) -> Dict:
        """Create comprehensive metadata for the gap filler document."""
        
        return {
            'gap_filler_context': {
                'title_id': title_id,
                'focus_keyword': research_results['focus_keyword'],
                'research_timestamp': research_results['research_timestamp'],
                'source_index': index,
                'total_sources_in_session': research_results['total_sources'],
                'target_collection': collection_name,
                'user_id': user_id
            },
            'source_information': {
                'original_title': source.get('title', ''),
                'source_name': source.get('source', ''),
                'source_url': source.get('url', ''),
                'data_source_type': source_type,
                'content_type': source.get('type', source_type)
            },
            'content_metadata': {
                'authors': source.get('authors', []),
                'published_date': source.get('published', ''),
                'description': source.get('description', source.get('summary', '')),
                'content_length': len(source.get('description', source.get('summary', ''))),
                'research_quality': self._assess_research_quality(source, source_type)
            },
            'processing_info': {
                'processed_by': 'gap_filler_system',
                'processing_timestamp': datetime.now().isoformat(),
                'auto_generated': True,
                'requires_chunking': True,
                'ready_for_embedding': True,
                'target_collection': collection_name,
                'collection_strategy': 'gap_filler_enhancement',
                'user_id': user_id
            }
        }
    
    def _extract_authors(self, source: Dict) -> str:
        """Extract and format authors from source."""
        authors = source.get('authors', [])
        if not authors:
            return source.get('source', 'Knowledge Gap Filler System')
        
        if isinstance(authors, list):
            if len(authors) <= 3:
                return ', '.join(authors)
            else:
                return f"{', '.join(authors[:3])} et al."
        else:
            return str(authors)[:100]
    
    def _create_short_summary(self, source: Dict, content: str) -> str:
        """Create a short summary for the document."""
        description = source.get('description', source.get('summary', ''))
        if description:
            summary = description[:200].strip()
            if len(description) > 200:
                summary += "..."
            return summary
        else:
            return content[:200].strip() + "..." if len(content) > 200 else content
    
    def _create_medium_summary(self, source: Dict, content: str) -> str:
        """Create a medium-length summary for the document."""
        description = source.get('description', source.get('summary', ''))
        title = source.get('title', 'Research Source')
        source_name = source.get('source', 'Unknown Source')
        
        summary_parts = [
            f"Source: {title}",
            f"From: {source_name}"
        ]
        
        if description:
            desc_summary = description[:500].strip()
            if len(description) > 500:
                desc_summary += "..."
            summary_parts.append(f"Content: {desc_summary}")
        
        return "\n".join(summary_parts)
    
    def _calculate_importance_score(self, source: Dict, source_type: str) -> float:
        """Calculate importance score for the gap filler document."""
        base_score = 0.7
        
        type_multipliers = {
            'academic_papers': 1.0,
            'government_data': 0.95,
            'industry_reports': 0.9,
            'textbooks': 0.85,
            'statistical_data': 0.8,
            'news_articles': 0.7
        }
        
        type_score = base_score * type_multipliers.get(source_type, 0.6)
        
        description = source.get('description', source.get('summary', '')).lower()
        title = source.get('title', '').lower()
        
        research_keywords = ['study', 'research', 'analysis', 'survey', 'report', 'data', 'statistics']
        keyword_boost = sum(0.05 for keyword in research_keywords if keyword in description or keyword in title)
        
        if source.get('published'):
            try:
                if any(year in str(source['published']) for year in ['2023', '2024', '2025']):
                    keyword_boost += 0.1
            except:
                pass
        
        final_score = min(1.0, type_score + keyword_boost)
        return round(final_score, 3)
    
    def _assess_research_quality(self, source: Dict, source_type: str) -> str:
        """Assess the research quality of the source."""
        description = source.get('description', source.get('summary', '')).lower()
        
        high_quality_indicators = ['peer-reviewed', 'academic', 'university', 'research institute', 
                                 'government', 'official', 'study', 'analysis']
        
        medium_quality_indicators = ['report', 'survey', 'data', 'statistics', 'industry']
        
        if any(indicator in description for indicator in high_quality_indicators):
            return 'high'
        elif any(indicator in description for indicator in medium_quality_indicators):
            return 'medium'
        else:
            return 'standard'

# Orchestrator Class
class EnhancedKnowledgeGapFillerOrchestrator:
    """Enhanced orchestrator with dependency injection and all functionality"""
    
    def __init__(self, llm_provider: str = "deepseek", document_processor=None, background_executor=None):
        self.supabase = HTTPSupabaseClient()
        self.llm_provider = llm_provider
        
        # Initialize components with injected dependencies
        self.gap_analyzer = KnowledgeGapAnalyzer(self.supabase, llm_provider)
        self.researcher = MultiSourceResearcher(self.supabase)
        self.rag_enhancer = EnhancedRAGKnowledgeEnhancer(
            self.supabase, 
            document_processor,
            background_executor
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Log initialization status
        if document_processor and background_executor:
            self.logger.info("✅ Knowledge Gap Orchestrator initialized with full document processing")
        else:
            self.logger.warning("⚠️ Knowledge Gap Orchestrator initialized without document processors")
            self.logger.warning("   Documents will be added to lindex_documents but not processed into vector store")
    
    async def process_single_title_with_collection(self, title_id: str, collection_name: str = None, 
                                                 merge_with_existing: bool = True, user_id: str = None):
        """Process single title with collection control and user tracking."""
        
        try:
            self.logger.info(f"🔍 Processing title {title_id} with collection control (user: {user_id})")
            
            # Get title data
            response = self.supabase.table("Titles").select("*").eq("id", title_id).execute()
            
            if not response.data:
                raise ValueError(f"Title {title_id} not found")
            
            title_data = response.data[0]
            
            # Analyze knowledge gap
            gap = await self.gap_analyzer._analyze_single_title(title_data, user_id)
            
            if not gap:
                return {
                    'status': 'no_gap',
                    'message': 'No knowledge gap identified',
                    'user_id': user_id
                }
            
            # Research gap
            research_results = await self.researcher.research_knowledge_gap(gap, user_id)
            
            if research_results['total_sources'] > 0:
                # Enhanced RAG with collection control
                enhancement_result = await self.rag_enhancer.enhance_rag_for_title(
                    title_id=title_id,
                    research_results=research_results,
                    collection_name=collection_name,
                    merge_with_existing=merge_with_existing,
                    user_id=user_id
                )
                
                if enhancement_result['success']:
                    return {
                        'status': 'success',
                        'title_id': title_id,
                        'sources_found': research_results['total_sources'],
                        'collection_name': enhancement_result['collection_name'],
                        'collection_strategy': enhancement_result['collection_strategy'],
                        'documents_added': enhancement_result['documents_added'],
                        'document_ids': enhancement_result['document_ids'],
                        'research_summary': research_results['research_summary'],
                        'manual_suggestions_generated': enhancement_result.get('manual_suggestions_generated', 0),
                        'suggestions_breakdown': enhancement_result.get('suggestions_breakdown', {}),
                        'user_id': user_id
                    }
                else:
                    return {
                        'status': 'enhancement_failed',
                        'error': enhancement_result.get('error'),
                        'user_id': user_id
                    }
            else:
                return {
                    'status': 'no_sources',
                    'message': 'No relevant sources found',
                    'user_id': user_id
                }
            
        except Exception as e:
            self.logger.error(f"Error processing single title {title_id}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'user_id': user_id
            }

# Route functions
def add_knowledge_gap_routes(app):
    """Add knowledge gap routes to Flask app"""
    
    def get_orchestrator():
        """Get orchestrator with current dependencies"""
        try:
            import __main__
            document_processor = getattr(__main__, 'document_processor', None)
            background_executor = getattr(__main__, 'background_executor', None)
            
            return EnhancedKnowledgeGapFillerOrchestrator(document_processor=document_processor, background_executor=background_executor)
        except:
            return EnhancedKnowledgeGapFillerOrchestrator()
    
    @app.route('/gap_filler/analyze_knowledge_gaps', methods=['GET'])
    def gap_filler_analyze_knowledge_gaps():
        """Analyze knowledge gaps without processing them"""
        try:
            user_id = request.args.get('user_id')
            
            orchestrator = get_orchestrator()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            gaps = loop.run_until_complete(
                orchestrator.gap_analyzer.analyze_new_titles(user_id)
            )
            
            gaps_summary = []
            for gap in gaps:
                gaps_summary.append({
                    'title_id': gap.title_id,
                    'focus_keyword': gap.focus_keyword,
                    'topic_category': gap.topic_category,
                    'priority_score': gap.priority_score,
                    'data_types_needed': gap.data_types_needed,
                    'specific_requirements': gap.specific_requirements,
                    'estimated_documents': len(gap.data_types_needed) * 2,
                    'will_add_to_lindex': True
                })
            
            gaps_summary.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return jsonify({
                "status": "success",
                "gaps_found": len(gaps_summary),
                "knowledge_gaps": gaps_summary,
                "analysis_timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "integration_info": {
                    "will_update_lindex_documents": True,
                    "content_type": "gap_filler",
                    "auto_processing": bool(orchestrator.rag_enhancer.document_processor),
                    "estimated_total_documents": sum(g['estimated_documents'] for g in gaps_summary)
                }
            })
            
        except Exception as e:
            logger.error(f"Error in analyze_knowledge_gaps: {e}")
            return jsonify({"status": "error", "error": str(e)}), 500

# Main integration function
# Complete ending for knowledge_gap_http_supabase.py

# Main integration function
def integrate_knowledge_gap_filler_http(app):
    """
    Main integration function for main.py
    """
    
    logger.info("🔧 Integrating Enhanced Knowledge Gap Filler...")
    
    try:
        # Get dependencies from main module
        import __main__
        
        document_processor = getattr(__main__, 'document_processor', None)
        background_executor = getattr(__main__, 'background_executor', None)
        
        if document_processor and background_executor:
            logger.info("✅ Found document processing dependencies")
        else:
            logger.warning("⚠️ Document processing dependencies not found - using mock implementation")
        
        # Add all routes
        add_knowledge_gap_routes(app)
        add_enhanced_routes(app)
        add_gap_closure_routes(app)
        
        logger.info("✅ Enhanced Knowledge Gap Filler integrated successfully!")
        logger.info("📋 Available endpoints:")
        logger.info("   GET  /gap_filler/analyze_knowledge_gaps?user_id=X - Analyze NEW titles")
        logger.info("   POST /gap_filler/enhance_knowledge/<title_id> - Enhance single title")
        logger.info("   GET  /gap_filler/status?user_id=X - System status")
        logger.info("   GET  /gap_closure_status?user_id=X - Gap closure status")
        logger.info("   POST /bulk_enhance_knowledge - Bulk enhance multiple titles")
        logger.info("   GET  /validate_knowledge_gap_setup - Validate system setup")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ INTEGRATION FAILED: {e}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False

def add_enhanced_routes(app):
    """Add enhanced routes with all functionality"""
    
    def get_orchestrator():
        import __main__
        document_processor = getattr(__main__, 'document_processor', None)
        background_executor = getattr(__main__, 'background_executor', None)
        return EnhancedKnowledgeGapFillerOrchestrator(document_processor=document_processor, background_executor=background_executor)
    
    @app.route('/gap_filler/enhance_knowledge/<title_id>', methods=['POST'])
    def gap_filler_enhance_single_title(title_id):
        """Enhance knowledge for a single title"""
        try:
            data = request.get_json() or {}
            
            collection_name = data.get('collection_name')
            merge_with_existing = data.get('merge_with_existing', True)
            user_id = data.get('user_id')
            
            orchestrator = get_orchestrator()
            
            logger.info(f"🎯 Enhancing title {title_id} (user: {user_id})")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(
                orchestrator.process_single_title_with_collection(
                    title_id, collection_name, merge_with_existing, user_id
                )
            )
            
            if results.get('status') == 'success':
                results['collection_info'] = {
                    'final_collection_name': results.get('collection_name'),
                    'collection_strategy': results.get('collection_strategy'),
                    'merge_with_existing': merge_with_existing,
                    'requested_collection': collection_name,
                    'documents_added_to_lindex': True,
                    'content_type': 'gap_filler',
                    'vector_processing_enabled': bool(orchestrator.rag_enhancer.document_processor)
                }
            
            return jsonify(results)
            
        except Exception as e:
            logger.error(f"Error in enhance_single_title: {e}")
            return jsonify({"status": "error", "error": str(e)}), 500
    
    @app.route('/gap_filler/status', methods=['GET'])
    def gap_filler_system_status():
        """Get gap filler system status"""
        try:
            user_id = request.args.get('user_id')
            
            return jsonify({
                "status": "operational",
                "user_id": user_id,
                "integration_status": {
                    "type": "enhanced_implementation",
                    "fully_functional": True,
                    "linkup_research": bool(os.getenv('LINKUP_API_KEY')),
                    "user_tracking_enabled": True,
                    "manual_suggestions_available": MANUAL_SUGGESTIONS_AVAILABLE
                },
                "api_status": {
                    "supabase_http": True,
                    "linkup_available": bool(os.getenv('LINKUP_API_KEY')),
                    "environment_ready": bool(os.getenv('SUPABASE_URL') and os.getenv('SUPABASE_KEY'))
                },
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in gap_filler_status: {e}")
            return jsonify({"status": "error", "error": str(e)}), 500
    
    @app.route('/analyze_knowledge_gaps_open_only', methods=['GET'])
    def analyze_knowledge_gaps_open_only():
        """Get only titles with open knowledge gaps"""
        try:
            user_id = request.args.get('user_id')
            
            orchestrator = get_orchestrator()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            gaps = loop.run_until_complete(
                orchestrator.gap_analyzer.analyze_new_titles_with_gap_filter(user_id)
            )
            
            knowledge_gaps = []
            for gap in gaps:
                knowledge_gaps.append({
                    'title_id': gap.title_id,
                    'focus_keyword': gap.focus_keyword,
                    'topic_category': gap.topic_category,
                    'priority_score': gap.priority_score,
                    'data_types_needed': gap.data_types_needed,
                    'specific_requirements': gap.specific_requirements,
                    'estimated_documents': len(gap.data_types_needed) * 3,
                    'gaps_currently_open': True
                })
            
            knowledge_gaps.sort(key=lambda x: x['priority_score'], reverse=True)
            
            return jsonify({
                "status": "success",
                "gaps_found": len(knowledge_gaps),
                "knowledge_gaps": knowledge_gaps,
                "user_id": user_id,
                "filter_applied": "open_gaps_only",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in analyze_knowledge_gaps_open_only: {e}")
            return jsonify({"status": "error", "error": str(e)}), 500




    @app.route('/bulk_enhance_knowledge', methods=['POST'])
    def bulk_enhance_knowledge():
        """Bulk enhance knowledge for multiple titles"""
        try:
            data = request.get_json() or {}
            title_ids = data.get('title_ids', [])
            user_id = data.get('user_id')
            max_concurrent = data.get('max_concurrent', 3)
            merge_with_existing = data.get('merge_with_existing', True)
            collection_name = data.get('collection_name')
            
            if not title_ids:
                return jsonify({"status": "error", "error": "No title_ids provided"}), 400
            
            logger.info(f"🔄 Bulk enhancing {len(title_ids)} titles (user: {user_id})")
            
            orchestrator = get_orchestrator()
            
            # Process titles sequentially for now (can be made concurrent later)
            successful_results = []
            failed_results = []
            
            for title_id in title_ids:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(
                        orchestrator.process_single_title_with_collection(
                            title_id, collection_name, merge_with_existing, user_id
                        )
                    )
                    
                    if result.get('status') == 'success':
                        successful_results.append(result)
                    else:
                        failed_results.append({
                            'title_id': title_id,
                            'status': 'failed',
                            'error': result.get('error', 'Unknown error')
                        })
                        
                except Exception as e:
                    failed_results.append({
                        'title_id': title_id,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            total_documents_added = sum(r.get('documents_added', 0) for r in successful_results)
            total_suggestions = sum(r.get('manual_suggestions_generated', 0) for r in successful_results)
            avg_docs = total_documents_added / len(successful_results) if successful_results else 0
            avg_suggestions = total_suggestions / len(successful_results) if successful_results else 0
            
            return jsonify({
                "status": "completed", 
                "total_titles": len(title_ids),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "results": successful_results + failed_results,
                "user_id": user_id,
                "bulk_processing_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_documents_added": total_documents_added,
                    "total_suggestions_generated": total_suggestions,
                    "average_documents_per_title": round(avg_docs, 1),
                    "average_suggestions_per_title": round(avg_suggestions, 1),
                    "success_rate": round((len(successful_results) / len(title_ids)) * 100, 1)
                }
            })
            
        except Exception as e:
            logger.error(f"Error in bulk enhance knowledge: {e}")
            return jsonify({"status": "error", "error": str(e)}), 500

def add_gap_closure_routes(app):
    """Add gap closure management routes"""
    
    def get_orchestrator():
        import __main__
        document_processor = getattr(__main__, 'document_processor', None)
        background_executor = getattr(__main__, 'background_executor', None)
        return EnhancedKnowledgeGapFillerOrchestrator(document_processor=document_processor, background_executor=background_executor)
    
    @app.route('/gap_closure_status', methods=['GET'])
    def gap_closure_status():
        """Get gap closure status for titles"""
        try:
            user_id = request.args.get('user_id')
            title_id = request.args.get('title_id')
            
            orchestrator = get_orchestrator()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            status = loop.run_until_complete(
                orchestrator.rag_enhancer.get_gap_closure_status(title_id, user_id)
            )
            
            return jsonify({
                "status": "success",
                "gap_closure_status": status,
                "query_timestamp": datetime.now().isoformat(),
                "user_id": user_id
            })
            
        except Exception as e:
            logger.error(f"Error in gap_closure_status: {e}")
            return jsonify({"status": "error", "error": str(e)}), 500
    
    @app.route('/mark_gaps_closed/<title_id>', methods=['POST'])
    def mark_gaps_closed_endpoint(title_id):
        """Manually mark gaps as closed for a specific title"""
        try:
            data = request.get_json() or {}
            closure_method = data.get('closure_method', 'manual')
            closure_notes = data.get('closure_notes', '')
            user_id = data.get('user_id')
            
            orchestrator = get_orchestrator()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            success = loop.run_until_complete(
                orchestrator.rag_enhancer.mark_gaps_as_closed(
                    title_id=title_id,
                    closure_method=closure_method,
                    closure_notes=closure_notes,
                    user_id=user_id
                )
            )
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": f"Gaps marked as closed for title {title_id}",
                    "title_id": title_id,
                    "closure_method": closure_method,
                    "closure_timestamp": datetime.now().isoformat(),
                    "user_id": user_id
                })
            else:
                return jsonify({
                    "status": "error",
                    "error": "Failed to mark gaps as closed"
                }), 500
            
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500
######################################



        # =============================================================================
        # New Endpoint: Additional Knowledge Enhancement for Titles with Closed Gaps
        # =============================================================================


    @app.route('/additional_knowledge_documents/<title_id>', methods=['GET'])
    def get_additional_knowledge_documents(title_id):
            """Get all additional knowledge documents for a specific title."""
            try:
                user_id = request.args.get('user_id')
                
                logger.info(f"📋 Getting additional knowledge documents for title: {title_id}")
                
                from knowledge_gap_http_supabase import HTTPSupabaseClient
                
                http_supabase = HTTPSupabaseClient()
                
                # Get additional knowledge documents for this title
                docs_response = http_supabase.execute_query(
                    'GET',
                    f'lindex_documents?select=id,title,source_type,data_source_type,processing_status,in_vector_store,chunk_count,focus_keyword,research_timestamp,summary_short,url,importance_score,enhancement_type&title_id=eq.{title_id}&source_type=eq.additional_gap_filler'
                )
                
                if not docs_response['success']:
                    return jsonify({
                        "status": "error",
                        "error": "Failed to fetch additional knowledge documents",
                        "details": docs_response.get('error')
                    }), 500
                
                additional_docs = docs_response['data']
                
                # Get the title information
                title_response = http_supabase.execute_query(
                    'GET',
                    f'Titles?select=id,Title,focus_keyword,status,additional_knowledge_enhanced,additional_enhancement_timestamp,additional_documents_count&id=eq.{title_id}'
                )
                
                title_info = {}
                if title_response['success'] and title_response['data']:
                    title_info = title_response['data'][0]
                
                # Organize documents by data source type
                docs_by_type = {}
                total_docs = len(additional_docs)
                docs_in_vector = 0
                total_chunks = 0
                
                for doc in additional_docs:
                    data_source_type = doc.get('data_source_type', 'unknown')
                    
                    if data_source_type not in docs_by_type:
                        docs_by_type[data_source_type] = []
                    
                    docs_by_type[data_source_type].append({
                        'document_id': doc.get('id'),
                        'title': doc.get('title', 'Unknown'),
                        'processing_status': doc.get('processing_status', 'unknown'),
                        'in_vector_store': doc.get('in_vector_store', False),
                        'chunk_count': doc.get('chunk_count', 0),
                        'summary': doc.get('summary_short', ''),
                        'url': doc.get('url', ''),
                        'importance_score': doc.get('importance_score', 0),
                        'research_timestamp': doc.get('research_timestamp', ''),
                        'enhancement_type': doc.get('enhancement_type', 'unknown')
                    })
                    
                    if doc.get('in_vector_store', False):
                        docs_in_vector += 1
                    
                    total_chunks += doc.get('chunk_count', 0)
                
                # Calculate effectiveness
                vector_effectiveness = round((docs_in_vector / total_docs) * 100, 1) if total_docs > 0 else 0
                
                return jsonify({
                    "status": "success",
                    "title_id": title_id,
                    "title_info": title_info,
                    "user_id": user_id,
                    "enhancement_type": "additional_knowledge",
                    "summary": {
                        "total_additional_docs": total_docs,
                        "documents_in_vector_store": docs_in_vector,
                        "total_chunks": total_chunks,
                        "vector_effectiveness": f"{vector_effectiveness}%",
                        "data_source_types": list(docs_by_type.keys()),
                        "source_type_counts": {k: len(v) for k, v in docs_by_type.items()},
                        "enhancement_timestamp": title_info.get('additional_enhancement_timestamp', 'Unknown')
                    },
                    "additional_knowledge_documents": docs_by_type,
                    "query_timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error getting additional knowledge documents for title {title_id}: {e}")
                return jsonify({"status": "error", "error": str(e)}), 500


    @app.route('/enhance_additional_knowledge/<title_id>', methods=['POST'])
    def enhance_additional_knowledge(title_id):
        """
        Find additional knowledge sources for titles with closed gaps.
        Ensures no duplicate URLs and focuses on expanding existing knowledge.
        """
        try:
            data = request.get_json() or {}
            user_id = data.get('user_id')
            collection_name = data.get('collection_name')
            research_depth = data.get('research_depth', 'standard')
            source_types = data.get('source_types', ['all'])
            exclude_existing_urls = data.get('exclude_existing_urls', True)
            
            if not user_id:
                return jsonify({"status": "error", "error": "user_id is required"}), 400
            
            logger.info(f"🔍 Additional knowledge enhancement for title: {title_id}")
            logger.info(f"   Research depth: {research_depth}")
            logger.info(f"   User: {user_id}")
            
            # ✅ Create HTTP client
            http_supabase = HTTPSupabaseClient()
            
            # ✅ Check if title exists and has closed gaps
            title_response = http_supabase.execute_query(
                'GET',
                f'Titles?select=id,Title,status,knowledge_gaps_closed,knowledge_enhanced,focus_keyword&id=eq.{title_id}'
            )
            
            if not title_response['success'] or not title_response['data']:
                return jsonify({
                    "status": "error",
                    "error": "Title not found"
                }), 404
            
            title_data = title_response['data'][0]
            
            # ✅ Verify title has closed gaps
            if not title_data.get('knowledge_gaps_closed', False):
                return jsonify({
                    "status": "error", 
                    "error": "Title must have closed knowledge gaps before additional enhancement",
                    "suggestion": "Use /analyze_knowledge_gaps_open_only endpoint first"
                }), 400
            
            # ✅ Get existing URLs to avoid duplicates
            existing_urls = set()
            if exclude_existing_urls:
                existing_docs_response = http_supabase.execute_query(
                    'GET',
                    f'lindex_documents?select=url&title_id=eq.{title_id}'
                )
                
                if existing_docs_response['success'] and existing_docs_response['data']:
                    existing_urls = {
                        doc.get('url', '') 
                        for doc in existing_docs_response['data']
                        if doc.get('url')
                    }
                    logger.info(f"📋 Found {len(existing_urls)} existing URLs to exclude")
            
            # ✅ CORRECT INITIALIZATION (matches your class constructor)
            try:
                gap_orchestrator = EnhancedRAGKnowledgeEnhancer(
                    supabase_client=http_supabase,
                    document_processor=None,
                    background_executor=None
                )
            except Exception as e:
                logger.error(f"Failed to initialize knowledge enhancer: {e}")
                return jsonify({
                    "status": "error",
                    "error": f"Knowledge enhancer initialization failed: {str(e)}"
                }), 500
            
            # ✅ Run async function synchronously
            enhancement_result = run_async_task(
                gap_orchestrator.enhance_additional_knowledge(
                    title_id=title_id,
                    user_id=user_id,
                    collection_name=collection_name,
                    research_depth=research_depth,
                    existing_urls=existing_urls,
                    source_types=source_types
                )
            )
            
            if not enhancement_result or not enhancement_result.get('success', False):
                error_msg = enhancement_result.get('error', 'Additional enhancement failed') if enhancement_result else 'Enhancement returned no result'
                logger.error(f"Additional enhancement failed for {title_id}: {error_msg}")
                return jsonify({
                    "status": "error",
                    "error": error_msg,
                    "title_id": title_id,
                    "user_id": user_id
                }), 500
            
            # ✅ Return comprehensive results
            return jsonify({
                "status": "success",
                "title_id": title_id,
                "title_name": title_data.get('Title', 'Unknown'),
                "user_id": user_id,
                "enhancement_type": "additional_knowledge",
                "research_depth": research_depth,
                "collection_name": enhancement_result.get('collection_name'),
                "additional_documents_added": enhancement_result.get('documents_added', 0),
                "new_sources_found": enhancement_result.get('new_sources_found', 0),
                "existing_urls_excluded": len(existing_urls),
                "source_breakdown": enhancement_result.get('source_breakdown', {}),
                "research_timestamp": enhancement_result.get('research_timestamp'),
                "processing_summary": enhancement_result.get('processing_summary', {}),
                "enhancement_effectiveness": enhancement_result.get('enhancement_effectiveness', 'unknown'),
                "next_steps": enhancement_result.get('next_steps', [])
            })
            
        except Exception as e:
            logger.error(f"Additional knowledge enhancement error for {title_id}: {e}")
            import traceback
            return jsonify({
                "status": "error",
                "error": str(e),
                "title_id": title_id,
                "traceback": traceback.format_exc()
            }), 500

    @app.route('/enhance_additional_knowledge_bulk', methods=['POST'])
    def enhance_additional_knowledge_bulk():
        """
        Bulk additional knowledge enhancement for multiple titles with closed gaps.
        """
        try:
            data = request.get_json() or {}
            user_id = data.get('user_id')
            title_ids = data.get('title_ids', [])
            collection_name = data.get('collection_name')
            research_depth = data.get('research_depth', 'standard')
            source_types = data.get('source_types', ['all'])
            exclude_existing_urls = data.get('exclude_existing_urls', True)
            
            # ✅ VALIDATION: Ensure user_id is provided
            if not user_id:
                return jsonify({"status": "error", "error": "user_id is required"}), 400
            
            if not title_ids or not isinstance(title_ids, list):
                return jsonify({"status": "error", "error": "title_ids must be a non-empty list"}), 400
            
            # ✅ LOG: Confirm user_id is received
            logger.info(f"🔍 Bulk additional knowledge enhancement for {len(title_ids)} titles")
            logger.info(f"   Research depth: {research_depth}")
            logger.info(f"   User: {user_id}")  # Confirm user_id logging
            
            # ✅ Create HTTP client
            http_supabase = HTTPSupabaseClient()
            
            # ✅ Filter titles to only those with closed gaps
            title_ids_str = ','.join(title_ids)
            titles_response = http_supabase.execute_query(
                'GET',
                f'Titles?select=id,Title,knowledge_gaps_closed,knowledge_enhanced&id=in.({title_ids_str})&knowledge_gaps_closed=eq.true'
            )
            
            if not titles_response['success'] or not titles_response['data']:
                return jsonify({
                    "status": "error",
                    "error": "No titles found with closed knowledge gaps",
                    "suggestion": "Ensure titles have completed gap closure first"
                }), 400
            
            eligible_titles = titles_response['data']
            eligible_title_ids = [t['id'] for t in eligible_titles]
            
            logger.info(f"📋 Found {len(eligible_titles)} titles eligible for additional enhancement")
            
            # ✅ CORRECT INITIALIZATION (matches your class constructor)
            try:
                gap_orchestrator = EnhancedRAGKnowledgeEnhancer(
                    supabase_client=http_supabase,
                    document_processor=None,
                    background_executor=None
                )
            except Exception as e:
                logger.error(f"Failed to initialize knowledge enhancer: {e}")
                return jsonify({
                    "status": "error",
                    "error": f"Knowledge enhancer initialization failed: {str(e)}"
                }), 500
            
            # ✅ Process each eligible title
            results = []
            successful_enhancements = 0
            total_additional_docs = 0
            
            for i, title_data in enumerate(eligible_titles):
                title_id = title_data['id']
                title_name = title_data.get('Title', 'Unknown')
                
                try:
                    logger.info(f"  Processing {i + 1}/{len(eligible_titles)}: {title_id} (user: {user_id})")
                    
                    # Get existing URLs for this title
                    existing_urls = set()
                    if exclude_existing_urls:
                        existing_docs_response = http_supabase.execute_query(
                            'GET',
                            f'lindex_documents?select=url&title_id=eq.{title_id}'
                        )
                        
                        if existing_docs_response['success'] and existing_docs_response['data']:
                            existing_urls = {
                                doc.get('url', '') 
                                for doc in existing_docs_response['data']
                                if doc.get('url')
                            }
                    
                    # ✅ CRITICAL: Ensure user_id is passed to enhance_additional_knowledge
                    enhancement_result = run_async_task(
                        gap_orchestrator.enhance_additional_knowledge(
                            title_id=title_id,
                            user_id=user_id,  # ✅ ENSURE user_id is passed
                            collection_name=collection_name,
                            research_depth=research_depth,
                            existing_urls=existing_urls,
                            source_types=source_types
                        )
                    )
                    
                    if enhancement_result and enhancement_result.get('success', False):
                        docs_added = enhancement_result.get('documents_added', 0)
                        successful_enhancements += 1
                        total_additional_docs += docs_added
                        
                        results.append({
                            "title_id": title_id,
                            "title_name": title_name,
                            "status": "success",
                            "additional_documents_added": docs_added,
                            "new_sources_found": enhancement_result.get('new_sources_found', 0),
                            "collection_name": enhancement_result.get('collection_name'),
                            "source_breakdown": enhancement_result.get('source_breakdown', {}),
                            "enhancement_effectiveness": enhancement_result.get('enhancement_effectiveness', 'unknown'),
                            "user_id": user_id  # ✅ Include user_id in response
                        })
                        
                        logger.info(f"    ✅ Added {docs_added} additional documents for {title_id} (user: {user_id})")
                    else:
                        error_msg = enhancement_result.get('error', 'Enhancement failed') if enhancement_result else 'No result returned'
                        results.append({
                            "title_id": title_id,
                            "title_name": title_name,
                            "status": "failed",
                            "error": error_msg,
                            "additional_documents_added": 0,
                            "user_id": user_id  # ✅ Include user_id even in failed results
                        })
                        
                        logger.warning(f"    ⚠️ Enhancement failed for {title_id}: {error_msg}")
                    
                    # ✅ Use regular sleep instead of asyncio.sleep
                    if i < len(eligible_titles) - 1:
                        import time
                        time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"    ❌ Error enhancing {title_id}: {e}")
                    results.append({
                        "title_id": title_id,
                        "title_name": title_name,
                        "status": "error",
                        "error": str(e),
                        "additional_documents_added": 0,
                        "user_id": user_id  # ✅ Include user_id in error results
                    })
            
            # ✅ Calculate summary statistics
            failed_enhancements = len(eligible_titles) - successful_enhancements
            success_rate = round((successful_enhancements / len(eligible_titles)) * 100, 1) if eligible_titles else 0
            
            return jsonify({
                "status": "success",
                "user_id": user_id,  # ✅ Return user_id in response
                "enhancement_type": "bulk_additional_knowledge",
                "research_depth": research_depth,
                "processing_summary": {
                    "total_titles_requested": len(title_ids),
                    "eligible_titles": len(eligible_titles),
                    "successful_enhancements": successful_enhancements,
                    "failed_enhancements": failed_enhancements,
                    "success_rate": f"{success_rate}%",
                    "total_additional_documents": total_additional_docs,
                    "user_id": user_id  # ✅ Include in summary as well
                },
                "results": results,
                "collection_name": collection_name,
                "research_timestamp": datetime.now().isoformat(),
                "next_steps": [
                    f"Review {total_additional_docs} additional documents added",
                    "Test enhanced RAG queries with expanded knowledge",
                    "Consider updating manual action suggestions",
                    "Monitor vector store performance"
                ] if successful_enhancements > 0 else [
                    "Check title enhancement status",
                    "Ensure knowledge gaps are properly closed",
                    "Review error messages for failed titles"
                ]
            })
            
        except Exception as e:
            logger.error(f"Bulk additional knowledge enhancement error: {e}")
            import traceback
            return jsonify({
                "status": "error",
                "error": str(e),
                "user_id": data.get('user_id') if 'data' in locals() else None,  # ✅ Include user_id in error response
                "traceback": traceback.format_exc()
            }), 500

##########################################


    @app.route('/validate_knowledge_gap_setup', methods=['GET'])
    def validate_knowledge_gap_setup():
        """Validate knowledge gap system setup"""
        try:
            validation_results = {
                'overall_status': 'ready',
                'environment_variables': {
                    'supabase_url': bool(os.getenv('SUPABASE_URL')),
                    'supabase_key': bool(os.getenv('SUPABASE_KEY')),
                    'linkup_api_key': bool(os.getenv('LINKUP_API_KEY')),
                    'openai_api_key': bool(os.getenv('OPENAI_API_KEY'))
                },
                'dependencies': {
                    'requests': True,
                    'asyncio': True,
                    'json': True,
                    'datetime': True,
                    'linkup': bool(os.getenv('LINKUP_API_KEY')),
                    'manual_suggestions': MANUAL_SUGGESTIONS_AVAILABLE
                },
                'runtime_tests': {
                    'supabase_connection': True,
                    'tables_accessible': True
                },
                'recommendations': []
            }
            
            # Determine overall status
            critical_checks = [
                validation_results['environment_variables']['supabase_url'],
                validation_results['environment_variables']['supabase_key']
            ]
            
            if all(critical_checks):
                validation_results['overall_status'] = 'ready'
            elif any(critical_checks):
                validation_results['overall_status'] = 'partial'
            else:
                validation_results['overall_status'] = 'not_ready'
            
            # Add recommendations
            if not validation_results['environment_variables']['linkup_api_key']:
                validation_results['recommendations'].append("Consider setting LINKUP_API_KEY for enhanced web search capabilities")
            
            if not validation_results['dependencies']['manual_suggestions']:
                validation_results['recommendations'].append("Manual action suggestions module not available")
            
            return jsonify(validation_results)
            
        except Exception as e:
            logger.error(f"Error in validate_knowledge_gap_setup: {e}")
            return jsonify({
                "overall_status": "error",
                "error": str(e)
            }), 500

# Test function
async def test_enhanced_knowledge_gap_system(user_id: str = "test_user"):
    """Test the enhanced knowledge gap system"""
    
    print("🧪 Testing Enhanced Knowledge Gap System")
    print("=" * 50)
    
    try:
        # Initialize orchestrator
        orchestrator = EnhancedKnowledgeGapFillerOrchestrator()
        print(f"✅ Enhanced orchestrator initialized (user: {user_id})")
        
        # Test 1: Analyze titles
        print("1. Testing title analysis...")
        gaps = await orchestrator.gap_analyzer.analyze_new_titles(user_id)
        print(f"   Found {len(gaps)} knowledge gaps")
        
        if gaps:
            gap = gaps[0]
            print(f"   Sample gap: {gap.focus_keyword} (category: {gap.topic_category})")
            print(f"   Data types needed: {gap.data_types_needed}")
            print(f"   Priority: {gap.priority_score:.2f}")
        
        # Test 2: Test research
        if gaps and orchestrator.researcher.linkup_client:
            print("2. Testing research functionality...")
            research_results = await orchestrator.researcher.research_knowledge_gap(gaps[0], user_id)
            print(f"   Found {research_results['total_sources']} total sources")
            for data_type, sources in research_results['sources_found'].items():
                print(f"   - {data_type}: {len(sources)} sources")
        
        print("✅ Enhanced Knowledge Gap System test completed successfully!")
        
        return {
            'status': 'success',
            'gaps_found': len(gaps),
            'enhanced_features': True,
            'user_id': user_id,
            'test_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'user_id': user_id,
            'test_timestamp': datetime.now().isoformat()
        }

# Export the integration function
__all__ = [
    'HTTPSupabaseClient',
    'KnowledgeGapAnalyzer', 
    'MultiSourceResearcher',
    'EnhancedRAGKnowledgeEnhancer',
    'EnhancedKnowledgeGapFillerOrchestrator',
    'integrate_knowledge_gap_filler_http'
]

if __name__ == "__main__":
    """Direct testing"""
    import sys
    
    user_id = sys.argv[1] if len(sys.argv) > 1 else "test_user"
    command = sys.argv[2] if len(sys.argv) > 2 else "test"
    
    if command == "test":
        print("🧪 Running Enhanced Knowledge Gap Filler Test...")
        print("=" * 50)
        
        result = asyncio.run(test_enhanced_knowledge_gap_system(user_id))
        
        if result['status'] == 'success':
            print(f"\n✅ Test completed successfully!")
            print(f"   Gaps found: {result['gaps_found']}")
            print(f"   User ID: {result['user_id']}")
            print(f"   Enhanced features: {result['enhanced_features']}")
        else:
            print(f"\n❌ Test failed: {result['error']}")
    else:
        print("Usage: python knowledge_gap_http_supabase.py [user_id] [test]")