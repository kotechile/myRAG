"""
Manus-wide Research Configuration
Configuration settings for advanced research capabilities
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ManusResearchConfig:
    """Configuration for Manus-wide research system"""
    
    # Research quality thresholds
    quality_threshold_low: int = 30
    quality_threshold_medium: int = 60
    quality_threshold_high: int = 85
    
    # Source diversity requirements
    min_source_types: int = 3
    min_total_sources: int = 8
    min_quality_sources: int = 2  # Academic papers, industry reports, etc.
    
    # Manus research capabilities
    enable_proprietary_databases: bool = True
    enable_expert_networks: bool = True
    enable_advanced_analysis: bool = True
    enable_real_time_intelligence: bool = True
    enable_cross_platform_integration: bool = True
    
    # API configurations
    advanced_search_apis: List[str] = None
    expert_network_apis: List[str] = None
    proprietary_data_sources: List[str] = None
    
    # Rate limiting and performance
    max_concurrent_searches: int = 5
    search_timeout_seconds: int = 30
    retry_attempts: int = 3
    
    # Content enhancement settings
    enable_semantic_analysis: bool = True
    enable_sentiment_analysis: bool = True
    enable_trend_analysis: bool = True
    enable_competitive_intelligence: bool = True
    
    def __post_init__(self):
        if self.advanced_search_apis is None:
            self.advanced_search_apis = [
                "perplexity_ai",
                "anthropic_claude", 
                "openai_gpt4",
                "google_gemini"
            ]
        
        if self.expert_network_apis is None:
            self.expert_network_apis = [
                "linkedin_sales_navigator",
                "crunchbase_pro",
                "pitchbook",
                "gartner_research"
            ]
        
        if self.proprietary_data_sources is None:
            self.proprietary_data_sources = [
                "bloomberg_terminal",
                "reuters_eikon",
                "capital_iq",
                "factset"
            ]

class ManusResearchSettings:
    """Runtime settings for Manus-wide research"""
    
    @staticmethod
    def get_config() -> ManusResearchConfig:
        """Get Manus research configuration from environment"""
        
        return ManusResearchConfig(
            quality_threshold_low=int(os.getenv('MANUS_QUALITY_THRESHOLD_LOW', 30)),
            quality_threshold_medium=int(os.getenv('MANUS_QUALITY_THRESHOLD_MEDIUM', 60)),
            quality_threshold_high=int(os.getenv('MANUS_QUALITY_THRESHOLD_HIGH', 85)),
            min_source_types=int(os.getenv('MANUS_MIN_SOURCE_TYPES', 3)),
            min_total_sources=int(os.getenv('MANUS_MIN_TOTAL_SOURCES', 8)),
            min_quality_sources=int(os.getenv('MANUS_MIN_QUALITY_SOURCES', 2)),
            enable_proprietary_databases=os.getenv('MANUS_ENABLE_PROPRIETARY', 'true').lower() == 'true',
            enable_expert_networks=os.getenv('MANUS_ENABLE_EXPERT_NETWORKS', 'true').lower() == 'true',
            enable_advanced_analysis=os.getenv('MANUS_ENABLE_ADVANCED_ANALYSIS', 'true').lower() == 'true',
            enable_real_time_intelligence=os.getenv('MANUS_ENABLE_REAL_TIME', 'true').lower() == 'true',
            enable_cross_platform_integration=os.getenv('MANUS_ENABLE_CROSS_PLATFORM', 'true').lower() == 'true',
            max_concurrent_searches=int(os.getenv('MANUS_MAX_CONCURRENT_SEARCHES', 5)),
            search_timeout_seconds=int(os.getenv('MANUS_SEARCH_TIMEOUT', 30)),
            retry_attempts=int(os.getenv('MANUS_RETRY_ATTEMPTS', 3)),
            enable_semantic_analysis=os.getenv('MANUS_ENABLE_SEMANTIC', 'true').lower() == 'true',
            enable_sentiment_analysis=os.getenv('MANUS_ENABLE_SENTIMENT', 'true').lower() == 'true',
            enable_trend_analysis=os.getenv('MANUS_ENABLE_TREND', 'true').lower() == 'true',
            enable_competitive_intelligence=os.getenv('MANUS_ENABLE_COMPETITIVE', 'true').lower() == 'true'
        )
    
    @staticmethod
    def get_feature_flags() -> Dict[str, bool]:
        """Get feature flags for Manus-wide research capabilities"""
        
        return {
            'manus_wide_research_enabled': os.getenv('MANUS_WIDE_RESEARCH_ENABLED', 'true').lower() == 'true',
            'advanced_search_enabled': os.getenv('MANUS_ADVANCED_SEARCH_ENABLED', 'true').lower() == 'true',
            'expert_network_enabled': os.getenv('MANUS_EXPERT_NETWORK_ENABLED', 'true').lower() == 'true',
            'proprietary_data_enabled': os.getenv('MANUS_PROPRIETARY_DATA_ENABLED', 'true').lower() == 'true',
            'real_time_intelligence': os.getenv('MANUS_REAL_TIME_INTELLIGENCE', 'true').lower() == 'true'
        }

# Research quality assessment utilities
class ResearchQualityAnalyzer:
    """Analyze research quality and determine when Manus-wide research is needed"""
    
    @staticmethod
    def assess_research_quality(research_results: Dict) -> Dict[str, Any]:
        """Comprehensive assessment of research quality"""
        
        config = ManusResearchSettings.get_config()
        
        total_sources = research_results.get('total_sources', 0)
        sources_by_type = research_results.get('sources_found', {})
        
        # Calculate various quality metrics
        source_count_score = min(total_sources * 10, 40)
        
        source_types_found = len([t for t, sources in sources_by_type.items() if sources])
        diversity_score = min(source_types_found * 15, 30)
        
        quality_sources = (
            len(sources_by_type.get('academic_papers', [])) * 3 +
            len(sources_by_type.get('industry_reports', [])) * 2 +
            len(sources_by_type.get('government_data', [])) * 2 +
            len(sources_by_type.get('statistical_data', [])) * 1
        )
        quality_score = min(quality_sources * 5, 30)
        
        overall_score = source_count_score + diversity_score + quality_score
        
        # Determine recommendation
        if overall_score < config.quality_threshold_low:
            recommendation = "manus_wide_research_required"
            priority = "critical"
        elif overall_score < config.quality_threshold_medium:
            recommendation = "manus_wide_research_recommended"
            priority = "high"
        elif overall_score < config.quality_threshold_high:
            recommendation = "additional_research_suggested"
            priority = "medium"
        else:
            recommendation = "research_quality_sufficient"
            priority = "low"
        
        return {
            'overall_score': min(overall_score, 100),
            'source_count_score': source_count_score,
            'diversity_score': diversity_score,
            'quality_score': quality_score,
            'recommendation': recommendation,
            'priority': priority,
            'manus_research_needed': recommendation in ['manus_wide_research_required', 'manus_wide_research_recommended']
        }
    
    @staticmethod
    def generate_quality_report(research_results: Dict) -> str:
        """Generate a detailed quality report"""
        
        assessment = ResearchQualityAnalyzer.assess_research_quality(research_results)
        
        report = f"""
# Research Quality Report

## Summary
- **Overall Score:** {assessment['overall_score']}/100
- **Recommendation:** {assessment['recommendation'].replace('_', ' ').title()}
- **Priority:** {assessment['priority'].title()}

## Detailed Metrics
- **Source Count Score:** {assessment['source_count_score']}/40
- **Diversity Score:** {assessment['diversity_score']}/30  
- **Quality Score:** {assessment['quality_score']}/30

## Manus-wide Research Recommendation
{assessment['manus_research_needed'] and 'Yes - Enable Manus-wide research for better results' or 'No - Current research quality is sufficient'}
"""
        
        return report.strip()

# Example usage and testing
if __name__ == "__main__":
    # Test configuration loading
    config = ManusResearchSettings.get_config()
    print("Manus Research Configuration:")
    print(f"Quality Threshold Low: {config.quality_threshold_low}")
    print(f"Min Source Types: {config.min_source_types}")
    print(f"Advanced APIs: {config.advanced_search_apis}")
    
    # Test feature flags
    flags = ManusResearchSettings.get_feature_flags()
    print("\nFeature Flags:")
    for flag, enabled in flags.items():
        print(f"  {flag}: {enabled}")
    
    # Test quality assessment
    sample_results = {
        'total_sources': 5,
        'sources_found': {
            'academic_papers': [],
            'industry_reports': [1, 2],
            'news_articles': [1, 2, 3]
        }
    }
    
    assessment = ResearchQualityAnalyzer.assess_research_quality(sample_results)
    print("\nQuality Assessment:")
    print(f"Score: {assessment['overall_score']}/100")
    print(f"Recommendation: {assessment['recommendation']}")
    print(f"Manus Research Needed: {assessment['manus_research_needed']}")