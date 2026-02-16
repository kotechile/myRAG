# manual_action_suggestions.py - Add-on for Knowledge Gap Filler

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class ActionType(Enum):
    TEXTBOOK = "textbook"
    DATASET = "dataset" 
    TOOL_DEVELOPMENT = "tool_development"
    EXPERT_INTERVIEW = "expert_interview"
    COURSE_CREATION = "course_creation"
    TEMPLATE_CREATION = "template_creation"
    PARTNERSHIP = "partnership"
    RESEARCH_STUDY = "research_study"
    MANUS_WIDE_RESEARCH = "manus_wide_research"

class ActionPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ManualActionSuggestionsGenerator:
    """Generates manual action suggestions that require human intervention"""
    
    def __init__(self, supabase_client, linkup_client=None):
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)
        self.linkup_client = linkup_client
        
        # Suggestion templates by topic category
        self.suggestion_templates = {
            'real_estate': {
                'textbooks': [
                    "Real Estate Investment Analysis and Valuation by William Poorvu",
                    "The Millionaire Real Estate Investor by Gary Keller",
                    "Commercial Real Estate Analysis and Investments by David Geltner"
                ],
                'datasets': [
                    "Local MLS (Multiple Listing Service) historical data",
                    "County property tax records and assessments",
                    "Rental yield data from local rental markets",
                    "Property appreciation rates by neighborhood"
                ],
                'tools': [
                    "Mortgage Payment Calculator with PMI and taxes",
                    "Investment Property ROI Calculator", 
                    "Property Comparison Tool with neighborhood stats",
                    "Rent vs Buy Calculator for your area",
                    "Property Cash Flow Analyzer"
                ],
                'expert_interviews': [
                    "Local real estate agents specializing in investment properties",
                    "Property management company owners",
                    "Real estate attorneys familiar with local laws",
                    "Successful local real estate investors"
                ]
            },
            'finance': {
                'textbooks': [
                    "A Random Walk Down Wall Street by Burton Malkiel",
                    "The Intelligent Investor by Benjamin Graham",
                    "Personal Finance for Dummies by Eric Tyson"
                ],
                'datasets': [
                    "Historical stock market performance data (S&P 500, etc.)",
                    "Federal Reserve economic indicators",
                    "Consumer spending and savings rate data",
                    "Credit score distribution and lending rates"
                ],
                'tools': [
                    "Retirement Savings Calculator with inflation",
                    "Debt Payoff Strategy Calculator",
                    "Emergency Fund Calculator",
                    "Investment Portfolio Rebalancing Tool",
                    "Tax-Loss Harvesting Calculator"
                ],
                'expert_interviews': [
                    "Certified Financial Planners (CFP)",
                    "Tax professionals and CPAs",
                    "Portfolio managers at investment firms",
                    "Financial advisors specializing in your audience"
                ]
            },
            'technology': {
                'textbooks': [
                    "Clean Code by Robert Martin",
                    "System Design Interview by Alex Xu",
                    "The Pragmatic Programmer by David Thomas"
                ],
                'datasets': [
                    "GitHub repository statistics and trends",
                    "Stack Overflow developer survey data",
                    "Technology adoption rates by industry",
                    "Programming language popularity trends"
                ],
                'tools': [
                    "Code Complexity Analyzer",
                    "API Response Time Calculator",
                    "Database Query Optimizer",
                    "Cloud Cost Calculator for AWS/Azure",
                    "Security Vulnerability Scanner"
                ],
                'expert_interviews': [
                    "Senior software engineers at major tech companies",
                    "DevOps engineers with cloud expertise",
                    "Cybersecurity specialists",
                    "Tech startup founders"
                ]
            },
            'health': {
                'textbooks': [
                    "Nutrition: Science and Applications by Lori Smolin",
                    "ACSM's Guidelines for Exercise Testing by ACSM",
                    "The Body Keeps the Score by Bessel van der Kolk"
                ],
                'datasets': [
                    "CDC health statistics and disease prevalence",
                    "FDA nutrition database and food safety data",
                    "Clinical trial results from ClinicalTrials.gov",
                    "WHO global health statistics"
                ],
                'tools': [
                    "BMI and Body Fat Percentage Calculator",
                    "Calorie and Macro Nutrient Tracker",
                    "Exercise Heart Rate Zone Calculator",
                    "Medication Interaction Checker",
                    "Symptom Assessment Tool"
                ],
                'expert_interviews': [
                    "Registered dietitians and nutritionists",
                    "Licensed physical therapists",
                    "Mental health counselors",
                    "Sports medicine physicians"
                ]
            },
            'business': {
                'textbooks': [
                    "Good to Great by Jim Collins",
                    "The Lean Startup by Eric Ries",
                    "Blue Ocean Strategy by W. Chan Kim"
                ],
                'datasets': [
                    "Small Business Administration (SBA) statistics",
                    "Industry-specific market research reports",
                    "Consumer behavior and purchasing data",
                    "Business failure and success rate statistics"
                ],
                'tools': [
                    "Business Plan Financial Projections Calculator",
                    "Customer Acquisition Cost (CAC) Calculator",
                    "Break-even Analysis Tool",
                    "Market Size Calculator",
                    "ROI and Payback Period Calculator"
                ],
                'expert_interviews': [
                    "Successful entrepreneurs in your industry",
                    "Business consultants and advisors",
                    "Venture capitalists and angel investors",
                    "Marketing and sales professionals"
                ]
            }
        }
    
    async def generate_action_suggestions(self, gap, research_results: Dict) -> List[Dict]:
        """Generate manual action suggestions based on knowledge gap and research results"""
        
        suggestions = []
        
        try:
            # Check if Manus-wide research is needed first
            if research_results.get('manus_suggestion_needed', False):
                manus_suggestions = await self.suggest_manus_wide_research(gap, research_results)
                suggestions.extend(manus_suggestions)
            
            # Use Linkup to find sources not readily available online
            if self.linkup_client:
                linkup_suggestions = await self._suggest_from_linkup(gap, research_results)
                suggestions.extend(linkup_suggestions)
            else:
                # Fallback to template-based suggestions if Linkup not available
                self.logger.warning("Linkup client not available - using template-based suggestions")
                suggestions.extend(await self._suggest_textbooks(gap, research_results))
                suggestions.extend(await self._suggest_datasets(gap, research_results))
            
            # Generate other types of suggestions (tools, interviews, templates)
            suggestions.extend(await self._suggest_tool_development(gap, research_results))
            suggestions.extend(await self._suggest_expert_interviews(gap, research_results))
            suggestions.extend(await self._suggest_templates_and_resources(gap, research_results))
            
            # Score and prioritize suggestions
            scored_suggestions = self._score_suggestions(suggestions, gap)
            
            return scored_suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating action suggestions: {e}")
            return []
    
    async def _suggest_from_linkup(self, gap, research_results: Dict) -> List[Dict]:
        """Use Linkup to find books, articles, datasets, and other sources not readily available online"""
        
        suggestions = []
        focus_keyword = gap.focus_keyword
        topic_category = gap.topic_category
        
        try:
            # Check what sources were already found
            sources_found = research_results.get('sources_found', {})
            total_sources = research_results.get('total_sources', 0)
            research_quality = research_results.get('research_quality_score', 0)
            
            # Only suggest if research quality is low or specific data types are missing
            if research_quality >= 70 and total_sources >= 10:
                self.logger.info(f"Research quality good ({research_quality}), skipping Linkup suggestions")
                return suggestions
            
            # Search for books and textbooks
            book_suggestions = await self._find_books_via_linkup(focus_keyword, topic_category, gap)
            suggestions.extend(book_suggestions)
            
            # Search for academic papers and research studies
            academic_suggestions = await self._find_academic_sources_via_linkup(focus_keyword, topic_category, gap, sources_found)
            suggestions.extend(academic_suggestions)
            
            # Search for specialized datasets
            dataset_suggestions = await self._find_datasets_via_linkup(focus_keyword, topic_category, gap, sources_found)
            suggestions.extend(dataset_suggestions)
            
            # Search for industry reports and market research
            report_suggestions = await self._find_industry_reports_via_linkup(focus_keyword, topic_category, gap, sources_found)
            suggestions.extend(report_suggestions)
            
            self.logger.info(f"✅ Generated {len(suggestions)} Linkup-based suggestions for {focus_keyword}")
            
        except Exception as e:
            self.logger.error(f"Error generating Linkup suggestions: {e}")
        
        return suggestions
    
    async def _find_books_via_linkup(self, focus_keyword: str, topic_category: str, gap) -> List[Dict]:
        """Use Linkup to find relevant books and textbooks"""
        
        suggestions = []
        
        try:
            # Search for books on the topic
            book_queries = [
                f"{focus_keyword} textbook comprehensive guide",
                f"{focus_keyword} book reference manual",
                f"best books about {focus_keyword}"
            ]
            
            for query in book_queries[:2]:  # Limit to 2 queries to control costs
                results = await self._linkup_search_safe(query, max_results=3)
                
                for result in results:
                    title = result.get('title', '')
                    url = result.get('url', '')
                    description = result.get('description', result.get('summary', ''))
                    
                    # Filter for book-related sources (Amazon, Goodreads, publisher sites, etc.)
                    if any(indicator in url.lower() for indicator in ['amazon', 'goodreads', 'publisher', 'book', 'textbook', 'library']):
                        # Extract book title and author if possible
                        resource_name = title
                        if 'author' in result:
                            resource_name = f"{title} by {result.get('author')}"
                        
                        suggestions.append({
                            'action_type': ActionType.TEXTBOOK.value,
                            'title': f"Read: {title}",
                            'description': f"Comprehensive book covering {focus_keyword}. {description[:200] if description else 'Found via Linkup research.'}",
                            'resource_name': resource_name,
                            'estimated_effort_hours': 20,
                            'difficulty_level': 'intermediate',
                            'expected_benefit': f"Deep understanding of {focus_keyword} from authoritative source",
                            'cost_estimate': '$30-80',
                            'implementation_notes': f"Purchase or borrow from library. URL: {url}",
                            'content_enhancement_potential': 'Provides authoritative quotes, frameworks, and structured approaches to the topic',
                            'linkup_source': url,
                            'linkup_title': title
                        })
                        
                        if len(suggestions) >= 3:  # Limit to 3 book suggestions
                            break
                
                if len(suggestions) >= 3:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error finding books via Linkup: {e}")
        
        return suggestions
    
    async def _find_academic_sources_via_linkup(self, focus_keyword: str, topic_category: str, gap, sources_found: Dict) -> List[Dict]:
        """Use Linkup to find academic papers and research studies"""
        
        suggestions = []
        
        # Skip if academic papers were already found
        if 'academic_papers' in sources_found and len(sources_found.get('academic_papers', [])) >= 3:
            return suggestions
        
        try:
            academic_queries = [
                f"{focus_keyword} research study academic paper",
                f"{focus_keyword} peer-reviewed journal article",
                f"{focus_keyword} scholarly research"
            ]
            
            for query in academic_queries[:2]:  # Limit queries
                results = await self._linkup_search_safe(query, max_results=3)
                
                for result in results:
                    title = result.get('title', '')
                    url = result.get('url', '')
                    description = result.get('description', result.get('summary', ''))
                    
                    # Filter for academic sources
                    if any(indicator in url.lower() for indicator in ['pubmed', 'arxiv', 'scholar', 'jstor', 'researchgate', 'academia', 'edu', 'journal']):
                        suggestions.append({
                            'action_type': ActionType.RESEARCH_STUDY.value,
                            'title': f"Research study: {title}",
                            'description': f"Academic research paper on {focus_keyword}. {description[:200] if description else 'Found via Linkup research.'}",
                            'resource_name': title,
                            'estimated_effort_hours': 4,
                            'difficulty_level': 'intermediate',
                            'expected_benefit': f"Peer-reviewed research and data on {focus_keyword}",
                            'cost_estimate': '$0-50',
                            'implementation_notes': f"Access via academic database or library. URL: {url}",
                            'content_enhancement_potential': 'Adds academic credibility and research-backed insights',
                            'linkup_source': url,
                            'linkup_title': title
                        })
                        
                        if len(suggestions) >= 2:  # Limit to 2 academic suggestions
                            break
                
                if len(suggestions) >= 2:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error finding academic sources via Linkup: {e}")
        
        return suggestions
    
    async def _find_datasets_via_linkup(self, focus_keyword: str, topic_category: str, gap, sources_found: Dict) -> List[Dict]:
        """Use Linkup to find specialized datasets"""
        
        suggestions = []
        
        # Skip if statistical data was already found
        if 'statistical_data' in sources_found and len(sources_found.get('statistical_data', [])) >= 3:
            return suggestions
        
        try:
            dataset_queries = [
                f"{focus_keyword} dataset download data",
                f"{focus_keyword} statistics database",
                f"{focus_keyword} data source CSV"
            ]
            
            for query in dataset_queries[:2]:  # Limit queries
                results = await self._linkup_search_safe(query, max_results=3)
                
                for result in results:
                    title = result.get('title', '')
                    url = result.get('url', '')
                    description = result.get('description', result.get('summary', ''))
                    
                    # Filter for dataset sources
                    if any(indicator in url.lower() or indicator in title.lower() for indicator in ['dataset', 'data', 'csv', 'database', 'kaggle', 'data.gov', 'statistics']):
                        suggestions.append({
                            'action_type': ActionType.DATASET.value,
                            'title': f"Dataset: {title}",
                            'description': f"Specialized dataset for {focus_keyword}. {description[:200] if description else 'Found via Linkup research.'}",
                            'resource_name': title,
                            'estimated_effort_hours': 8,
                            'difficulty_level': 'intermediate',
                            'expected_benefit': f"Data-driven insights and statistics for {focus_keyword}",
                            'cost_estimate': '$0-500',
                            'implementation_notes': f"Download and analyze dataset. URL: {url}",
                            'content_enhancement_potential': 'Provides exclusive statistics and data visualizations',
                            'linkup_source': url,
                            'linkup_title': title
                        })
                        
                        if len(suggestions) >= 2:  # Limit to 2 dataset suggestions
                            break
                
                if len(suggestions) >= 2:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error finding datasets via Linkup: {e}")
        
        return suggestions
    
    async def _find_industry_reports_via_linkup(self, focus_keyword: str, topic_category: str, gap, sources_found: Dict) -> List[Dict]:
        """Use Linkup to find industry reports and market research"""
        
        suggestions = []
        
        # Only suggest for professional/entrepreneur audience
        if gap.target_audience not in ['professional', 'entrepreneur']:
            return suggestions
        
        # Skip if industry reports were already found
        if 'industry_reports' in sources_found and len(sources_found.get('industry_reports', [])) >= 3:
            return suggestions
        
        try:
            report_queries = [
                f"{focus_keyword} industry report market research",
                f"{focus_keyword} market analysis report",
                f"{focus_keyword} industry benchmark study"
            ]
            
            for query in report_queries[:2]:  # Limit queries
                results = await self._linkup_search_safe(query, max_results=3)
                
                for result in results:
                    title = result.get('title', '')
                    url = result.get('url', '')
                    description = result.get('description', result.get('summary', ''))
                    
                    # Filter for report sources (may require purchase)
                    if any(indicator in url.lower() or indicator in title.lower() for indicator in ['report', 'research', 'analysis', 'market', 'gartner', 'forrester', 'mckinsey']):
                        suggestions.append({
                            'action_type': ActionType.DATASET.value,  # Industry reports are datasets
                            'title': f"Industry report: {title}",
                            'description': f"Comprehensive industry report on {focus_keyword}. {description[:200] if description else 'Found via Linkup research.'}",
                            'resource_name': title,
                            'estimated_effort_hours': 6,
                            'difficulty_level': 'intermediate',
                            'expected_benefit': f"Industry insights and market intelligence for {focus_keyword}",
                            'cost_estimate': '$200-2000',
                            'implementation_notes': f"Purchase or access via subscription. URL: {url}",
                            'content_enhancement_potential': 'Provides authoritative industry data and competitive analysis',
                            'linkup_source': url,
                            'linkup_title': title
                        })
                        
                        if len(suggestions) >= 2:  # Limit to 2 report suggestions
                            break
                
                if len(suggestions) >= 2:
                    break
                    
        except Exception as e:
            self.logger.error(f"Error finding industry reports via Linkup: {e}")
        
        return suggestions
    
    async def _linkup_search_safe(self, query: str, max_results: int = 3) -> List[Dict]:
        """Safely perform Linkup search with error handling"""
        
        if not self.linkup_client:
            return []
        
        try:
            import asyncio
            
            # Use standard depth for cost efficiency (suggestions are supplementary)
            depth = "standard"
            output_type = "searchResults"
            
            self.logger.info(f"🔍 LINKUP SUGGESTION SEARCH: '{query}' (max: {max_results}, depth: {depth})")
            
            # Make the API call
            response = await asyncio.to_thread(
                self.linkup_client.search,
                query=query,
                depth=depth,
                output_type=output_type
            )
            
            results = []
            
            # Process searchResults
            if hasattr(response, 'results') and response.results:
                for result in response.results[:max_results]:
                    try:
                        title = getattr(result, 'name', getattr(result, 'title', 'Unknown Title'))
                        content = getattr(result, 'content', getattr(result, 'description', ''))
                        url = getattr(result, 'url', '')
                        author = getattr(result, 'author', None)
                        
                        results.append({
                            'title': title,
                            'description': content,
                            'summary': content[:500] + "..." if len(content) > 500 else content,
                            'url': url,
                            'author': author
                        })
                    except Exception as e:
                        self.logger.warning(f"Error processing Linkup result: {e}")
                        continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Linkup search for suggestions: {e}")
            return []
    
    async def _suggest_textbooks(self, gap, research_results: Dict) -> List[Dict]:
        """Suggest relevant textbooks to read"""
        
        suggestions = []
        topic_category = gap.topic_category
        
        # Get base textbook suggestions for the category
        base_textbooks = self.suggestion_templates.get(topic_category, {}).get('textbooks', [])
        
        for textbook in base_textbooks:
            suggestions.append({
                'action_type': ActionType.TEXTBOOK.value,
                'title': f"Read: {textbook}",
                'description': f"Comprehensive textbook covering {gap.focus_keyword} fundamentals and advanced concepts",
                'resource_name': textbook,
                'estimated_effort_hours': 20,
                'difficulty_level': 'intermediate',
                'expected_benefit': f"Deep understanding of {gap.focus_keyword} principles and methodologies",
                'cost_estimate': '$30-80',
                'implementation_notes': 'Purchase book and dedicate 1-2 hours daily for reading and note-taking',
                'content_enhancement_potential': 'Provides authoritative quotes, frameworks, and structured approaches to the topic'
            })
        
        # Add specific textbook suggestions based on research gaps
        if 'academic_papers' in gap.data_types_needed:
            suggestions.append({
                'action_type': ActionType.TEXTBOOK.value,
                'title': f"Find academic textbooks on {gap.focus_keyword}",
                'description': f"Research university-level textbooks that cover {gap.focus_keyword} from an academic perspective",
                'resource_name': f"Academic texts in {topic_category}",
                'estimated_effort_hours': 5,
                'difficulty_level': 'beginner',
                'expected_benefit': 'Access to peer-reviewed theories and research methodologies',
                'cost_estimate': '$50-150',
                'implementation_notes': 'Check university bookstores, library databases, and academic publishers',
                'content_enhancement_potential': 'Adds academic credibility and theoretical foundation to content'
            })
        
        return suggestions
    
    async def _suggest_datasets(self, gap, research_results: Dict) -> List[Dict]:
        """Suggest relevant datasets to research and acquire based on gap analysis"""
        
        suggestions = []
        topic_category = gap.topic_category
        focus_keyword = gap.focus_keyword.lower()
        
        # Check what data was actually found in research
        sources_found = research_results.get('sources_found', {})
        total_sources = research_results.get('total_sources', 0)
        research_quality = research_results.get('research_quality_score', 0)
        
        # Analyze focus keyword to determine appropriate dataset types
        is_product_name = self._is_product_or_brand_name(focus_keyword)
        is_simple_query = len(focus_keyword.split()) <= 2 and not any(char in focus_keyword for char in ['rate', 'cost', 'price', 'comparison', 'vs'])
        needs_benchmarking = self._needs_benchmarking_data(focus_keyword, topic_category, gap.target_audience)
        needs_statistical_data = 'statistical_data' in gap.data_types_needed
        needs_market_data = 'industry_reports' in gap.data_types_needed or 'market_data' in gap.data_types_needed
        
        # Only suggest base category datasets if research quality is low or specific data types are missing
        if research_quality < 50 or needs_statistical_data:
            base_datasets = self.suggestion_templates.get(topic_category, {}).get('datasets', [])
            
            # Filter to only suggest relevant datasets
            for dataset in base_datasets:
                # Skip generic suggestions if we have a specific product/query
                if is_product_name and 'general' in dataset.lower():
                    continue
                    
                suggestions.append({
                    'action_type': ActionType.DATASET.value,
                    'title': f"Acquire dataset: {dataset}",
                    'description': f"Obtain and analyze {dataset} to provide data-driven insights for {gap.focus_keyword}",
                    'resource_name': dataset,
                    'estimated_effort_hours': 8,
                    'difficulty_level': 'intermediate',
                    'expected_benefit': f'Unique data insights and statistics to support content about {gap.focus_keyword}',
                    'cost_estimate': '$0-500',
                    'implementation_notes': 'Research data sources, download/purchase dataset, perform analysis in Excel/Python',
                    'content_enhancement_potential': 'Provides exclusive statistics and data visualizations'
                })
        
        # Suggest specific datasets based on focus keyword and gap requirements
        if needs_benchmarking and not is_product_name and not is_simple_query:
            # Generate contextual benchmarking suggestion
            benchmark_title, benchmark_desc, benchmark_resource = self._generate_benchmarking_suggestion(
                gap.focus_keyword, topic_category
            )
            
            suggestions.append({
                'action_type': ActionType.DATASET.value,
                'title': benchmark_title,
                'description': benchmark_desc,
                'resource_name': benchmark_resource,
                'estimated_effort_hours': 12,
                'difficulty_level': 'advanced',
                'expected_benefit': f'Competitive analysis and industry positioning insights for {gap.focus_keyword}',
                'cost_estimate': '$200-1000',
                'implementation_notes': self._generate_benchmarking_implementation_notes(gap.focus_keyword, topic_category),
                'content_enhancement_potential': 'Enables comparative analysis and industry-specific recommendations'
            })
        
        # Suggest specific datasets based on data types needed
        if needs_market_data and not is_product_name:
            market_suggestion = self._generate_market_data_suggestion(gap.focus_keyword, topic_category)
            if market_suggestion:
                suggestions.append(market_suggestion)
        
        # Suggest specialized datasets based on focus keyword analysis
        specialized_suggestions = self._generate_specialized_dataset_suggestions(gap, research_results)
        suggestions.extend(specialized_suggestions)
        
        return suggestions
    
    def _is_product_or_brand_name(self, focus_keyword: str) -> bool:
        """Determine if focus keyword is a specific product or brand name"""
        # Common product naming patterns
        product_keywords = ['model', 'pro', 'max', 'plus', 'premium', 'edition', 'series', 'version']
        has_product_keyword = any(keyword in focus_keyword for keyword in product_keywords)
        
        # Check for model numbers/versions (t6, t5, v1, etc.)
        words = focus_keyword.split()
        has_model_number = any(word.lower() in ['t6', 't5', 't4', 'v1', 'v2', 'v3'] or 
                              (len(word) <= 3 and word[0].isupper() and word[1:].isdigit()) 
                              for word in words)
        
        # Check for brand names (common ones)
        common_brands = ['honeywell', 'nest', 'ecobee', 'amazon', 'google', 'apple', 'samsung']
        has_brand = any(brand in focus_keyword.lower() for brand in common_brands)
        
        # Check for Brand Model format (e.g., "Honeywell T6")
        is_brand_model_format = (len(words) == 2 and 
                                any(word.isupper() or (word[0].isupper() and len(word) <= 3) 
                                    for word in words))
        
        return has_brand or has_model_number or has_product_keyword or is_brand_model_format
    
    def _needs_benchmarking_data(self, focus_keyword: str, topic_category: str, target_audience: str) -> bool:
        """Determine if benchmarking data is relevant for this gap"""
        
        # Benchmarking is not relevant for simple queries or product names
        if len(focus_keyword.split()) <= 2:
            return False
        
        # Benchmarking is relevant for business/finance topics with professional audience
        if topic_category in ['business', 'finance'] and target_audience in ['professional', 'entrepreneur']:
            # But not for simple queries like "how find"
            if any(word in focus_keyword for word in ['how', 'what', 'why', 'when', 'where']):
                return len(focus_keyword.split()) > 3  # Only if it's a complex query
            
            # Relevant for topics about rates, costs, performance, strategies
            benchmarking_keywords = ['rate', 'cost', 'price', 'performance', 'strategy', 'analysis', 
                                   'comparison', 'benchmark', 'roi', 'return', 'investment', 'market']
            return any(keyword in focus_keyword for keyword in benchmarking_keywords)
        
        return False
    
    def _generate_benchmarking_suggestion(self, focus_keyword: str, topic_category: str) -> tuple:
        """Generate contextual benchmarking suggestion title, description, and resource name"""
        
        # Extract the core topic from focus keyword
        core_topic = focus_keyword
        
        # Create specific title based on topic
        if 'rate' in focus_keyword.lower() or 'cost' in focus_keyword.lower():
            title = f"Industry benchmarking data for {focus_keyword}"
            description = f"Collect current industry rates, costs, and performance benchmarks for {focus_keyword} to provide accurate comparative analysis"
            resource = f"{topic_category} industry rate and cost benchmarks"
        elif 'performance' in focus_keyword.lower() or 'efficiency' in focus_keyword.lower():
            title = f"Performance benchmarking data for {focus_keyword}"
            description = f"Gather industry performance metrics and efficiency benchmarks related to {focus_keyword}"
            resource = f"{topic_category} performance benchmarks"
        else:
            title = f"Market analysis and benchmarking data for {focus_keyword}"
            description = f"Collect comprehensive market data, industry benchmarks, and competitive analysis for {focus_keyword}"
            resource = f"{topic_category} market benchmarks and analysis"
        
        return title, description, resource
    
    def _generate_benchmarking_implementation_notes(self, focus_keyword: str, topic_category: str) -> str:
        """Generate specific implementation notes for benchmarking data collection"""
        
        if topic_category == 'finance':
            return f"Research current {focus_keyword} rates from Federal Reserve, financial institutions, and industry reports. Compare across different providers and regions."
        elif topic_category == 'business':
            return f"Contact industry associations for {focus_keyword} benchmarks. Purchase market research reports from Gartner, Forrester, or industry-specific analysts. Conduct surveys if needed."
        elif topic_category == 'real_estate':
            return f"Obtain local MLS data, county records, and real estate market reports for {focus_keyword}. Compare with national averages and similar markets."
        else:
            return f"Contact relevant industry associations, purchase market research reports, and conduct surveys to gather benchmarking data for {focus_keyword}"
    
    def _generate_market_data_suggestion(self, focus_keyword: str, topic_category: str) -> Optional[Dict]:
        """Generate market data suggestion if relevant"""
        
        # Only suggest for topics that would benefit from market data
        market_keywords = ['market', 'industry', 'trend', 'growth', 'size', 'forecast', 'outlook']
        if not any(keyword in focus_keyword.lower() for keyword in market_keywords):
            return None
        
        return {
            'action_type': ActionType.DATASET.value,
            'title': f"Market research data for {focus_keyword}",
            'description': f"Acquire comprehensive market research data including market size, growth trends, and forecasts for {focus_keyword}",
            'resource_name': f"{topic_category} market research data",
            'estimated_effort_hours': 10,
            'difficulty_level': 'intermediate',
            'expected_benefit': f'Current market intelligence and trend analysis for {focus_keyword}',
            'cost_estimate': '$300-1500',
            'implementation_notes': 'Purchase market research reports from industry analysts, access industry databases, review government economic data',
            'content_enhancement_potential': 'Provides authoritative market data and trend analysis'
        }
    
    def _generate_specialized_dataset_suggestions(self, gap, research_results: Dict) -> List[Dict]:
        """Generate specialized dataset suggestions based on gap analysis"""
        
        suggestions = []
        focus_keyword = gap.focus_keyword.lower()
        
        # Check what's missing from research
        sources_found = research_results.get('sources_found', {})
        
        # Suggest historical data if relevant
        if any(word in focus_keyword for word in ['rate', 'price', 'cost', 'trend', 'historical']):
            if 'statistical_data' not in sources_found or len(sources_found.get('statistical_data', [])) < 2:
                suggestions.append({
                    'action_type': ActionType.DATASET.value,
                    'title': f"Historical data and trends for {gap.focus_keyword}",
                    'description': f"Collect historical data showing trends, changes, and patterns for {gap.focus_keyword} over time",
                    'resource_name': f"Historical {gap.focus_keyword} data",
                    'estimated_effort_hours': 6,
                    'difficulty_level': 'intermediate',
                    'expected_benefit': f'Historical context and trend analysis for {gap.focus_keyword}',
                    'cost_estimate': '$0-300',
                    'implementation_notes': f'Access government databases, historical records, and time-series data sources for {gap.focus_keyword}',
                    'content_enhancement_potential': 'Provides historical context and trend visualization'
                })
        
        # Suggest comparison data if relevant
        if 'comparison' in focus_keyword or 'vs' in focus_keyword or 'versus' in focus_keyword:
            suggestions.append({
                'action_type': ActionType.DATASET.value,
                'title': f"Comparative analysis data for {gap.focus_keyword}",
                'description': f"Gather data comparing different options, providers, or approaches for {gap.focus_keyword}",
                'resource_name': f"Comparison data for {gap.focus_keyword}",
                'estimated_effort_hours': 8,
                'difficulty_level': 'intermediate',
                'expected_benefit': f'Data-driven comparison to help readers make informed decisions about {gap.focus_keyword}',
                'cost_estimate': '$100-500',
                'implementation_notes': f'Research multiple sources, compile comparison tables, verify data accuracy across providers',
                'content_enhancement_potential': 'Enables objective comparison and decision-making support'
            })
        
        return suggestions
    
    async def _suggest_tool_development(self, gap, research_results: Dict) -> List[Dict]:
        """Suggest simple tools that would add value for readers"""
        
        suggestions = []
        topic_category = gap.topic_category
        
        # Get base tool suggestions for the category
        base_tools = self.suggestion_templates.get(topic_category, {}).get('tools', [])
        
        for tool in base_tools:
            # Estimate development complexity
            complexity = self._estimate_tool_complexity(tool)
            
            suggestions.append({
                'action_type': ActionType.TOOL_DEVELOPMENT.value,
                'title': f"Develop: {tool}",
                'description': f"Create an interactive {tool} to help readers with {gap.focus_keyword}",
                'resource_name': tool,
                'estimated_effort_hours': complexity['hours'],
                'difficulty_level': complexity['difficulty'],
                'expected_benefit': 'Interactive tool that provides immediate value to readers',
                'cost_estimate': complexity['cost'],
                'implementation_notes': f"Build using {complexity['tech_stack']}. Include input validation and mobile responsiveness",
                'content_enhancement_potential': 'Embedded calculator/tool increases engagement and provides practical value',
                'technical_requirements': complexity['requirements']
            })
        
        # Add content-specific tool suggestions
        if 'comparison' in gap.focus_keyword.lower() or 'vs' in gap.focus_keyword.lower():
            suggestions.append({
                'action_type': ActionType.TOOL_DEVELOPMENT.value,
                'title': f"Interactive Comparison Tool for {gap.focus_keyword}",
                'description': f"Side-by-side comparison tool that helps users evaluate different options in {gap.focus_keyword}",
                'resource_name': f"{gap.focus_keyword} comparison tool",
                'estimated_effort_hours': 16,
                'difficulty_level': 'intermediate',
                'expected_benefit': 'Helps readers make informed decisions with visual comparisons',
                'cost_estimate': '$500-1500',
                'implementation_notes': 'Create dynamic comparison table with filtering and sorting capabilities',
                'content_enhancement_potential': 'Increases time on page and provides actionable decision-making support',
                'technical_requirements': 'Frontend: HTML/CSS/JavaScript, Backend: Optional for data storage'
            })
        
        return suggestions
    
    async def _suggest_expert_interviews(self, gap, research_results: Dict) -> List[Dict]:
        """Suggest expert interviews to conduct"""
        
        suggestions = []
        topic_category = gap.topic_category
        
        # Get base expert suggestions for the category
        base_experts = self.suggestion_templates.get(topic_category, {}).get('expert_interviews', [])
        
        for expert_type in base_experts:
            suggestions.append({
                'action_type': ActionType.EXPERT_INTERVIEW.value,
                'title': f"Interview: {expert_type}",
                'description': f"Conduct interview with {expert_type} to gain insider insights on {gap.focus_keyword}",
                'resource_name': expert_type,
                'estimated_effort_hours': 6,
                'difficulty_level': 'intermediate',
                'expected_benefit': 'Exclusive expert quotes and insights not available elsewhere',
                'cost_estimate': '$0-300',
                'implementation_notes': 'Prepare interview questions, schedule 30-60 minute session, record and transcribe',
                'content_enhancement_potential': 'Adds authority through expert opinions and exclusive insights',
                'interview_topics': self._generate_interview_topics(gap, expert_type)
            })
        
        return suggestions
    
    async def _suggest_templates_and_resources(self, gap, research_results: Dict) -> List[Dict]:
        """Suggest templates and downloadable resources to create"""
        
        suggestions = []
        
        # Template suggestions based on topic
        if 'guide' in gap.focus_keyword.lower() or 'how to' in gap.focus_keyword.lower():
            suggestions.append({
                'action_type': ActionType.TEMPLATE_CREATION.value,
                'title': f"Create checklist template for {gap.focus_keyword}",
                'description': f"Develop a step-by-step checklist that readers can follow for {gap.focus_keyword}",
                'resource_name': f"{gap.focus_keyword} checklist",
                'estimated_effort_hours': 3,
                'difficulty_level': 'beginner',
                'expected_benefit': 'Practical takeaway that increases content value and lead generation',
                'cost_estimate': '$0-50',
                'implementation_notes': 'Create PDF template with checkboxes and action items',
                'content_enhancement_potential': 'Downloadable resource for email capture and increased engagement'
            })
        
        # Planning templates
        if gap.topic_category in ['business', 'finance', 'real_estate']:
            suggestions.append({
                'action_type': ActionType.TEMPLATE_CREATION.value,
                'title': f"Create planning template for {gap.focus_keyword}",
                'description': f"Develop a planning worksheet that helps readers organize their {gap.focus_keyword} strategy",
                'resource_name': f"{gap.focus_keyword} planning template",
                'estimated_effort_hours': 4,
                'difficulty_level': 'beginner',
                'expected_benefit': 'Actionable planning tool that provides immediate value',
                'cost_estimate': '$0-100',
                'implementation_notes': 'Design Excel/Google Sheets template with formulas and guidance',
                'content_enhancement_potential': 'Lead magnet and practical application of content concepts'
            })
        
        return suggestions

    async def suggest_manus_wide_research(self, gap, research_results: Dict) -> List[Dict]:
        """Suggest Manus-wide research when standard research yields insufficient results"""
        
        suggestions = []
        
        # Only suggest if research quality is low
        if research_results.get('research_quality_score', 0) >= 30:
            return suggestions
        
        # High-priority Manus-wide research suggestion
        suggestions.append({
            'action_type': ActionType.MANUS_WIDE_RESEARCH.value,
            'title': f"Enable Manus-wide research for {gap.focus_keyword}",
            'description': f"Standard research yielded insufficient results for {gap.focus_keyword}. Enable Manus-wide research to access advanced AI-powered search across proprietary databases, expert networks, and specialized knowledge sources.",
            'resource_name': "Manus-wide Research System",
            'estimated_effort_hours': 2,
            'difficulty_level': 'beginner',
            'expected_benefit': 'Access to advanced research capabilities, proprietary data sources, and expert-level insights not available through standard search',
            'cost_estimate': '$0',
            'implementation_notes': 'Enable Manus-wide research mode in system settings or contact system administrator to activate advanced research capabilities',
            'content_enhancement_potential': 'Significantly improves content depth and authority by accessing exclusive research sources and expert insights',
            'research_quality_score': research_results.get('research_quality_score', 0),
            'manus_features': [
                'AI-powered semantic search across proprietary databases',
                'Access to expert networks and specialized knowledge bases',
                'Advanced data synthesis and analysis capabilities',
                'Real-time market intelligence and trend analysis',
                'Cross-platform knowledge integration'
            ]
        })
        
        return suggestions
    
    def _estimate_tool_complexity(self, tool_name: str) -> Dict:
        """Estimate development complexity for a tool"""
        
        tool_lower = tool_name.lower()
        
        # Simple calculators
        if 'calculator' in tool_lower and any(word in tool_lower for word in ['simple', 'basic', 'payment', 'loan']):
            return {
                'hours': 8,
                'difficulty': 'beginner',
                'cost': '$200-500',
                'tech_stack': 'HTML, CSS, JavaScript',
                'requirements': 'Basic web development skills, formula implementation'
            }
        
        # Advanced calculators with multiple inputs
        elif 'calculator' in tool_lower and any(word in tool_lower for word in ['roi', 'investment', 'analysis']):
            return {
                'hours': 16,
                'difficulty': 'intermediate', 
                'cost': '$500-1200',
                'tech_stack': 'HTML, CSS, JavaScript, Chart.js for visualizations',
                'requirements': 'Intermediate web development, financial formula knowledge'
            }
        
        # Comparison tools
        elif 'comparison' in tool_lower or 'analyzer' in tool_lower:
            return {
                'hours': 20,
                'difficulty': 'intermediate',
                'cost': '$800-2000',
                'tech_stack': 'HTML, CSS, JavaScript, possibly React/Vue.js',
                'requirements': 'Frontend framework knowledge, data management skills'
            }
        
        # Interactive tools with data
        elif any(word in tool_lower for word in ['tracker', 'planner', 'optimizer']):
            return {
                'hours': 24,
                'difficulty': 'advanced',
                'cost': '$1200-3000',
                'tech_stack': 'Full-stack development (React/Vue + Node.js/Python)',
                'requirements': 'Full-stack development skills, database management'
            }
        
        # Default complexity
        else:
            return {
                'hours': 12,
                'difficulty': 'intermediate',
                'cost': '$400-1000',
                'tech_stack': 'HTML, CSS, JavaScript',
                'requirements': 'Web development skills, domain knowledge'
            }
    
    def _generate_interview_topics(self, gap, expert_type: str) -> List[str]:
        """Generate relevant interview topics for an expert"""
        
        base_topics = [
            f"Current trends and challenges in {gap.focus_keyword}",
            f"Common mistakes people make with {gap.focus_keyword}",
            f"Best practices for {gap.target_audience} in {gap.focus_keyword}",
            f"Future outlook and predictions for {gap.focus_keyword}",
            f"Tools and resources you recommend for {gap.focus_keyword}"
        ]
        
        # Add expert-specific topics
        if 'financial' in expert_type.lower():
            base_topics.extend([
                "Most effective strategies for different income levels",
                "How to balance risk and return in current market conditions"
            ])
        elif 'real estate' in expert_type.lower():
            base_topics.extend([
                "Market conditions and timing considerations",
                "Local vs national market factors to consider"
            ])
        elif 'technology' in expert_type.lower():
            base_topics.extend([
                "Emerging technologies and their impact",
                "Skills gap and learning recommendations"
            ])
        
        return base_topics
    
    def _score_suggestions(self, suggestions: List[Dict], gap) -> List[Dict]:
        """Score and prioritize suggestions based on impact and feasibility"""
        
        for suggestion in suggestions:
            # Calculate impact score (0-100)
            impact_score = self._calculate_impact_score(suggestion, gap)
            
            # Calculate feasibility score (0-100)
            feasibility_score = self._calculate_feasibility_score(suggestion)
            
            # Combined priority score
            priority_score = (impact_score * 0.6) + (feasibility_score * 0.4)
            
            suggestion['impact_score'] = impact_score
            suggestion['feasibility_score'] = feasibility_score
            suggestion['priority_score'] = priority_score
            suggestion['priority_level'] = self._determine_priority_level(priority_score)
        
        # Sort by priority score (highest first)
        suggestions.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return suggestions
    
    def _calculate_impact_score(self, suggestion: Dict, gap) -> int:
        """Calculate impact score based on content enhancement potential"""
        
        score = 50  # Base score
        
        # Action type impact
        if suggestion['action_type'] == ActionType.TOOL_DEVELOPMENT.value:
            score += 25  # Tools provide high engagement
        elif suggestion['action_type'] == ActionType.EXPERT_INTERVIEW.value:
            score += 20  # Expert insights add authority
        elif suggestion['action_type'] == ActionType.DATASET.value:
            score += 15  # Data adds credibility
        
        # Difficulty level impact (easier = higher impact for audience)
        if suggestion['difficulty_level'] == 'beginner':
            score += 15
        elif suggestion['difficulty_level'] == 'intermediate':
            score += 10
        
        # Target audience alignment
        if gap.target_audience == 'beginner' and suggestion['difficulty_level'] == 'beginner':
            score += 10
        elif gap.target_audience == 'professional' and suggestion['difficulty_level'] == 'advanced':
            score += 10
        
        return min(100, score)
    
    def _calculate_feasibility_score(self, suggestion: Dict) -> int:
        """Calculate feasibility score based on effort and cost"""
        
        score = 50  # Base score
        
        # Effort hours (lower is better)
        hours = suggestion.get('estimated_effort_hours', 10)
        if hours <= 4:
            score += 30
        elif hours <= 8:
            score += 20
        elif hours <= 16:
            score += 10
        else:
            score -= 10
        
        # Cost estimate (lower is better)
        cost_str = suggestion.get('cost_estimate', '$500')
        if '$0' in cost_str or 'free' in cost_str.lower():
            score += 20
        elif any(amount in cost_str for amount in ['$0-50', '$0-100']):
            score += 15
        elif any(amount in cost_str for amount in ['$50-200', '$100-500']):
            score += 10
        elif any(amount in cost_str for amount in ['$500-1000']):
            score += 5
        else:
            score -= 5
        
        return min(100, max(0, score))
    
    def _determine_priority_level(self, priority_score: float) -> str:
        """Determine priority level based on score"""
        
        if priority_score >= 80:
            return ActionPriority.CRITICAL.value
        elif priority_score >= 65:
            return ActionPriority.HIGH.value
        elif priority_score >= 45:
            return ActionPriority.MEDIUM.value
        else:
            return ActionPriority.LOW.value
    
    async def save_suggestions_to_database(self, title_id: str, suggestions: List[Dict], 
                                        research_context: Dict, user_id: str) -> bool:
        """Save manual action suggestions to Supabase table with user_id"""
        
        try:
            # Prepare suggestions for database insertion
            suggestions_to_insert = []
            
            for suggestion in suggestions:
                suggestion_record = {
                    'title_id': title_id,
                    'user_id': user_id,  # ADD THIS LINE
                    'action_type': suggestion['action_type'],
                    'title': suggestion['title'],
                    'description': suggestion['description'],
                    'resource_name': suggestion['resource_name'],
                    'estimated_effort_hours': suggestion['estimated_effort_hours'],
                    'difficulty_level': suggestion['difficulty_level'],
                    'expected_benefit': suggestion['expected_benefit'],
                    'cost_estimate': suggestion['cost_estimate'],
                    'implementation_notes': suggestion['implementation_notes'],
                    'content_enhancement_potential': suggestion['content_enhancement_potential'],
                    'impact_score': suggestion['impact_score'],
                    'feasibility_score': suggestion['feasibility_score'],
                    'priority_score': suggestion['priority_score'],
                    'priority_level': suggestion['priority_level'],
                    'status': 'suggested',
                    'created_at': datetime.now().isoformat(),
                    'research_context': json.dumps(research_context),
                    'additional_data': json.dumps({
                        'technical_requirements': suggestion.get('technical_requirements'),
                        'interview_topics': suggestion.get('interview_topics'),
                        'original_focus_keyword': research_context.get('focus_keyword'),
                        'linkup_source': suggestion.get('linkup_source'),
                        'linkup_title': suggestion.get('linkup_title')
                    })
                }
                suggestions_to_insert.append(suggestion_record)
            
            # Insert into manual_action_suggestions table
            result = self.supabase.execute_query('POST', 'manual_action_suggestions', suggestions_to_insert)
            
            if result['success']:
                self.logger.info(f"✅ Saved {len(suggestions_to_insert)} manual action suggestions for title {title_id}, user {user_id}")
                return True
            else:
                raise Exception(f"Failed to save suggestions: {result.get('error')}")
            
        except Exception as e:
            self.logger.error(f"Error saving manual action suggestions: {e}")
            return False

# SQL for creating the manual_action_suggestions table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS manual_action_suggestions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    title_id TEXT NOT NULL,
    action_type TEXT NOT NULL CHECK (action_type IN ('textbook', 'dataset', 'tool_development', 'expert_interview', 'course_creation', 'template_creation', 'partnership', 'research_study', 'manus_wide_research')),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    resource_name TEXT NOT NULL,
    estimated_effort_hours INTEGER NOT NULL,
    difficulty_level TEXT NOT NULL CHECK (difficulty_level IN ('beginner', 'intermediate', 'advanced')),
    expected_benefit TEXT NOT NULL,
    cost_estimate TEXT NOT NULL,
    implementation_notes TEXT NOT NULL,
    content_enhancement_potential TEXT NOT NULL,
    impact_score INTEGER NOT NULL CHECK (impact_score >= 0 AND impact_score <= 100),
    feasibility_score INTEGER NOT NULL CHECK (feasibility_score >= 0 AND feasibility_score <= 100),
    priority_score DECIMAL(5,2) NOT NULL CHECK (priority_score >= 0 AND priority_score <= 100),
    priority_level TEXT NOT NULL CHECK (priority_level IN ('low', 'medium', 'high', 'critical')),
    status TEXT NOT NULL DEFAULT 'suggested' CHECK (status IN ('suggested', 'in_progress', 'completed', 'rejected')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    research_context JSONB,
    additional_data JSONB,
    notes TEXT,
    
    -- Foreign key constraint (optional, depends on your titles table structure)
    CONSTRAINT fk_title_id FOREIGN KEY (title_id) REFERENCES titles(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_manual_action_suggestions_title_id ON manual_action_suggestions(title_id);
CREATE INDEX IF NOT EXISTS idx_manual_action_suggestions_action_type ON manual_action_suggestions(action_type);
CREATE INDEX IF NOT EXISTS idx_manual_action_suggestions_priority_level ON manual_action_suggestions(priority_level);
CREATE INDEX IF NOT EXISTS idx_manual_action_suggestions_status ON manual_action_suggestions(status);
CREATE INDEX IF NOT EXISTS idx_manual_action_suggestions_priority_score ON manual_action_suggestions(priority_score DESC);
"""