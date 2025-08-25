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
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.logger = logging.getLogger(__name__)
        
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
            
            # Generate other types of suggestions
            suggestions.extend(await self._suggest_textbooks(gap, research_results))
            suggestions.extend(await self._suggest_datasets(gap, research_results))
            suggestions.extend(await self._suggest_tool_development(gap, research_results))
            suggestions.extend(await self._suggest_expert_interviews(gap, research_results))
            suggestions.extend(await self._suggest_templates_and_resources(gap, research_results))
            
            # Score and prioritize suggestions
            scored_suggestions = self._score_suggestions(suggestions, gap)
            
            return scored_suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating action suggestions: {e}")
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
        """Suggest relevant datasets to research and acquire"""
        
        suggestions = []
        topic_category = gap.topic_category
        
        # Get base dataset suggestions for the category
        base_datasets = self.suggestion_templates.get(topic_category, {}).get('datasets', [])
        
        for dataset in base_datasets:
            suggestions.append({
                'action_type': ActionType.DATASET.value,
                'title': f"Acquire dataset: {dataset}",
                'description': f"Obtain and analyze {dataset} to provide data-driven insights for {gap.focus_keyword}",
                'resource_name': dataset,
                'estimated_effort_hours': 8,
                'difficulty_level': 'intermediate',
                'expected_benefit': 'Unique data insights and statistics to support content claims',
                'cost_estimate': '$0-500',
                'implementation_notes': 'Research data sources, download/purchase dataset, perform analysis in Excel/Python',
                'content_enhancement_potential': 'Provides exclusive statistics and data visualizations'
            })
        
        # Add specific dataset suggestions based on audience and topic
        if gap.target_audience in ['professional', 'entrepreneur']:
            suggestions.append({
                'action_type': ActionType.DATASET.value,
                'title': f"Industry benchmarking data for {gap.focus_keyword}",
                'description': f"Collect industry-specific performance metrics and benchmarks related to {gap.focus_keyword}",
                'resource_name': f"{topic_category} industry benchmarks",
                'estimated_effort_hours': 12,
                'difficulty_level': 'advanced',
                'expected_benefit': 'Competitive analysis and industry positioning insights',
                'cost_estimate': '$200-1000',
                'implementation_notes': 'Contact industry associations, purchase market research reports, conduct surveys',
                'content_enhancement_potential': 'Enables comparative analysis and industry-specific recommendations'
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
                        'original_focus_keyword': research_context.get('focus_keyword')
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