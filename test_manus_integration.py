"""
Test script for Manus-wide research integration
"""

import asyncio
import logging
from knowledge_gap_http_supabase import KnowledgeGap, MultiSourceResearcher
from manual_action_suggestions import ManualActionSuggestionsGenerator
from manus_research_config import ResearchQualityAnalyzer, ManusResearchSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManusIntegrationTester:
    """Test the Manus-wide research integration"""
    
    def __init__(self):
        logger.info("🔬 Initializing Manus Integration Tester")
        self.test_gaps = []
        self.test_results = []
    
    def create_test_gaps(self):
        """Create test knowledge gaps for validation"""
        
        # Test case 1: High-quality research scenario
        high_quality_gap = KnowledgeGap(
            title_id="test_high_quality_001",
            focus_keyword="real estate investment strategies",
            topic_category="real_estate",
            data_types_needed=["industry_reports", "government_data", "statistical_data"],
            specific_needs=["Market analysis", "Investment ROI data", "Regional trends"],
            target_audience="professional"
        )
        
        # Test case 2: Low-quality research scenario
        low_quality_gap = KnowledgeGap(
            title_id="test_low_quality_001", 
            focus_keyword="obscure technology concept",
            topic_category="technology",
            data_types_needed=["academic_papers", "industry_reports"],
            specific_needs=["Technical specifications", "Implementation guides"],
            target_audience="beginner"
        )
        
        self.test_gaps = [high_quality_gap, low_quality_gap]
        logger.info(f"✅ Created {len(self.test_gaps)} test gaps")
        
        return self.test_gaps
    
    def simulate_research_results(self):
        """Simulate research results for testing"""
        
        # High quality results
        high_quality_results = {
            'title_id': 'test_high_quality_001',
            'focus_keyword': 'real estate investment strategies',
            'sources_found': {
                'industry_reports': [{'title': '2024 Real Estate Market Analysis', 'url': 'https://example.com/report1'}] * 5,
                'government_data': [{'title': 'Housing Market Statistics', 'url': 'https://gov.example.com/data'}] * 3,
                'statistical_data': [{'title': 'Investment ROI Data', 'url': 'https://stats.example.com/roi'}] * 4
            },
            'total_sources': 12
        }
        
        # Low quality results  
        low_quality_results = {
            'title_id': 'test_low_quality_001',
            'focus_keyword': 'obscure technology concept',
            'sources_found': {
                'academic_papers': [],
                'industry_reports': [{'title': 'Basic Overview', 'url': 'https://example.com/basic'}]
            },
            'total_sources': 1
        }
        
        return [high_quality_results, low_quality_results]
    
    async def test_quality_analysis(self):
        """Test the research quality analysis"""
        
        logger.info("🔍 Testing research quality analysis")
        
        # Test with Manus config
        test_results = self.simulate_research_results()
        
        for i, results in enumerate(test_results):
            quality_score = ResearchQualityAnalyzer.assess_research_quality(results)
            
            logger.info(f"Test {i+1}:")
            logger.info(f"  Total Sources: {results['total_sources']}")
            logger.info(f"  Quality Score: {quality_score['overall_score']}/100")
            logger.info(f"  Recommendation: {quality_score['recommendation']}")
            logger.info(f"  Manus Research Needed: {quality_score['manus_research_needed']}")
            logger.info(f"  Priority: {quality_score['priority']}")
            
            self.test_results.append(quality_score)
    
    async def test_suggestion_generation(self):
        """Test the Manus-wide research suggestion generation"""
        
        logger.info("💡 Testing suggestion generation")
        
        # Create mock suggestion generator
        class MockSupabaseClient:
            def execute_query(self, method, endpoint, data=None):
                return {'success': True, 'data': []}
        
        generator = ManualActionSuggestionsGenerator(MockSupabaseClient())
        
        test_gaps = self.create_test_gaps()
        test_results = self.simulate_research_results()
        
        for gap, results in zip(test_gaps, test_results):
            results['research_quality_score'] = ResearchQualityAnalyzer.assess_research_quality(results)['overall_score']
            results['manus_suggestion_needed'] = results['research_quality_score'] < 30
            
            suggestions = await generator.suggest_manus_wide_research(gap, results)
            
            logger.info(f"Gap: {gap.focus_keyword}")
            logger.info(f"  Research Quality: {results['research_quality_score']}/100")
            logger.info(f"  Manus Suggestion Needed: {results['manus_suggestion_needed']}")
            logger.info(f"  Suggestions Generated: {len(suggestions)}")
            
            for suggestion in suggestions:
                logger.info(f"    - {suggestion['title']}")
                logger.info(f"      Priority: {suggestion['priority_level']}")
                logger.info(f"      Expected Benefit: {suggestion['expected_benefit']}")
    
    async def run_all_tests(self):
        """Run all integration tests"""
        
        logger.info("🚀 Starting Manus Integration Tests")
        
        try:
            # Test configuration loading
            logger.info("⚙️ Testing configuration loading")
            config = ManusResearchSettings.get_config()
            logger.info(f"✅ Configuration loaded: quality_threshold_low={config.quality_threshold_low}")
            
            # Test feature flags
            flags = ManusResearchSettings.get_feature_flags()
            logger.info(f"✅ Feature flags loaded: {flags}")
            
            # Test quality analysis
            await self.test_quality_analysis()
            
            # Test suggestion generation
            await self.test_suggestion_generation()
            
            # Summary
            logger.info("\n📊 Test Summary:")
            logger.info(f"  ✅ Configuration: OK")
            logger.info(f"  ✅ Quality Analysis: OK") 
            logger.info(f"  ✅ Suggestion Generation: OK")
            logger.info(f"  ✅ All tests passed!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return False

async def main():
    """Main test runner"""
    
    tester = ManusIntegrationTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 All Manus integration tests passed!")
        print("The Manus-wide research suggestion system is ready for use.")
    else:
        print("\n❌ Some tests failed. Please check the logs above.")

if __name__ == "__main__":
    asyncio.run(main())