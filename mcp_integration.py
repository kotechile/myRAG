# mcp_integration.py - Integration layer for your existing MCP services

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add paths to your other services
current_dir = Path(__file__).parent
rag_path = current_dir.parent / "rag-system"
research_path = current_dir.parent / "web_research"

sys.path.append(str(rag_path))
sys.path.append(str(research_path))

print(f"🔧 Looking for services at:")
print(f"   RAG: {rag_path}")
print(f"   Research: {research_path}")

@dataclass
class MCPServiceConfig:
    """Configuration for MCP services"""
    rag_service_timeout: int = 30
    research_service_timeout: int = 120

class RAGMCPClient:
    """Client for your existing RAG MCP service"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        # Import your RAG functions directly
        try:
            from fastmcp_rag_server import query_hybrid_enhanced, get_collection_info, list_collections
            self.query_hybrid_enhanced = query_hybrid_enhanced
            self.get_collection_info = get_collection_info
            self.list_collections = list_collections
            print("✅ Successfully imported RAG functions")
        except ImportError as e:
            print(f"⚠️ RAG import warning: {e}")
            self.query_hybrid_enhanced = None
            self.get_collection_info = None
            self.list_collections = None
    
    def test_connection(self):
        """Test if RAG service is accessible"""
        if not self.query_hybrid_enhanced:
            return {"status": "error", "error": "RAG functions not available"}
        
        try:
            # Try to list collections
            result_str = self.list_collections()
            result = json.loads(result_str) if isinstance(result_str, str) else result_str
            return {"status": "success", "message": "RAG service connected", "collections": result}
        except Exception as e:
            return {"status": "error", "error": f"RAG connection test failed: {str(e)}"}
    
    def analyze_content_gaps(self, collection_name: str, topic_areas: List[str]) -> Dict[str, Any]:
        """Analyze content gaps across multiple topic areas"""
        if not self.query_hybrid_enhanced:
            return {"status": "error", "error": "RAG function not available"}
        
        print(f"🔍 Analyzing content gaps for {len(topic_areas)} topics...")
        
        gap_analysis = {
            "collection_name": collection_name,
            "topic_areas_analyzed": topic_areas,
            "gaps_by_topic": {},
            "overall_coverage": {}
        }
        
        for i, topic in enumerate(topic_areas, 1):
            print(f"  {i}/{len(topic_areas)}: {topic}")
            
            query = f"What comprehensive information, strategies, approaches, and detailed coverage exists about {topic}? Include all methods, techniques, and expert guidance available."
            
            try:
                result_str = self.query_hybrid_enhanced(query, collection_name, top_k=15)
                result = json.loads(result_str) if isinstance(result_str, str) else result_str
                
                if result.get("status") == "success":
                    chunks_used = result.get("chunks_used", 0)
                    docs_searched = result.get("documents_searched", 0)
                    response_length = len(result.get("response", ""))
                    
                    # Determine coverage depth
                    if chunks_used >= 12 and docs_searched >= 3 and response_length > 1000:
                        coverage_depth = "high"
                    elif chunks_used >= 6 and docs_searched >= 2 and response_length > 500:
                        coverage_depth = "medium"
                    elif chunks_used >= 2 and response_length > 200:
                        coverage_depth = "low"
                    else:
                        coverage_depth = "minimal"
                    
                    gap_analysis["gaps_by_topic"][topic] = {
                        "existing_coverage": result.get("response", "")[:500] + "...",  # Truncate for display
                        "chunks_found": chunks_used,
                        "documents_covering": docs_searched,
                        "coverage_depth": coverage_depth,
                        "response_length": response_length
                    }
                    
                    print(f"     ✅ {coverage_depth.upper()} coverage ({chunks_used} chunks)")
                else:
                    gap_analysis["gaps_by_topic"][topic] = {
                        "error": result.get("error", "Unknown error"),
                        "coverage_depth": "none"
                    }
                    print(f"     ❌ Error: {result.get('error', 'Unknown')}")
            except Exception as e:
                gap_analysis["gaps_by_topic"][topic] = {
                    "error": str(e),
                    "coverage_depth": "error"
                }
                print(f"     ❌ Exception: {str(e)}")
        
        # Calculate overall coverage summary
        coverage_summary = {}
        for topic, data in gap_analysis["gaps_by_topic"].items():
            depth = data.get("coverage_depth", "none")
            coverage_summary[depth] = coverage_summary.get(depth, 0) + 1
        
        gap_analysis["overall_coverage"] = {
            "total_topics_analyzed": len(topic_areas),
            "coverage_distribution": coverage_summary,
            "content_gap_opportunities": [topic for topic, data in gap_analysis["gaps_by_topic"].items() 
                                        if data.get("coverage_depth") in ["low", "minimal", "none"]]
        }
        
        return gap_analysis

class ResearchMCPClient:
    """Client for your existing CrewAI research MCP service"""
    
    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        try:
            from agents import run_research
            self.run_research = run_research
            print("✅ Successfully imported research function")
        except ImportError as e:
            print(f"⚠️ Research import warning: {e}")
            self.run_research = None
    
    def test_connection(self):
        """Test if research service is accessible"""
        if not self.run_research:
            return {"status": "error", "error": "Research function not available"}
        
        try:
            # Try a simple test query
            result = self.run_research("test connection")
            return {"status": "success", "message": "Research service connected", "test_result_length": len(str(result))}
        except Exception as e:
            return {"status": "error", "error": f"Research connection test failed: {str(e)}"}
    
    def search_trends(self, domain: str, keywords: str, timeframe: str = "30d") -> Dict[str, Any]:
        """Search for current trends using your CrewAI research system"""
        if not self.run_research:
            return {"status": "error", "error": "Research function not available"}
        
        try:
            query = f"What are the current trends and emerging topics in {domain} specifically related to {keywords} over the last {timeframe}? Include market insights, popular discussions, and content opportunities."
            
            print(f"🔍 Researching trends: {query[:100]}...")
            result = self.run_research(query)
            
            return {
                "status": "success",
                "domain": domain,
                "keywords": keywords,
                "research_result": str(result),
                "trending_topics": self._extract_trending_topics(str(result))
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _extract_trending_topics(self, research_result: str) -> List[str]:
        """Extract trending topics from research result"""
        lines = research_result.lower().split('\n')
        trends = []
        for line in lines:
            if any(keyword in line for keyword in ['trend', 'trending', 'popular', 'emerging']):
                if ':' in line:
                    trend = line.split(':')[0].strip()
                    if len(trend) < 100:
                        trends.append(trend)
        return trends[:10]

class IntegratedTopicDiscovery:
    """Integrated topic discovery using both MCP services"""
    
    def __init__(self, config: MCPServiceConfig = None):
        if config is None:
            config = MCPServiceConfig()
        
        self.rag_client = RAGMCPClient(config.rag_service_timeout)
        self.research_client = ResearchMCPClient(config.research_service_timeout)
        self.config = config
    
    def test_connections(self):
        """Test both service connections"""
        print("🧪 Testing service connections...")
        
        rag_test = self.rag_client.test_connection()
        research_test = self.research_client.test_connection()
        
        print(f"RAG Service: {rag_test['status']} - {rag_test.get('message', rag_test.get('error'))}")
        print(f"Research Service: {research_test['status']} - {research_test.get('message', research_test.get('error'))}")
        
        return {
            "rag_service": rag_test,
            "research_service": research_test,
            "both_working": rag_test["status"] == "success" and research_test["status"] == "success"
        }
    
    def discover_topics(self, domain: str, collection_name: str, 
                       focus_areas: List[str], competitors: List[str]) -> Dict[str, Any]:
        """Comprehensive topic discovery using both services"""
        
        print(f"🔍 Starting integrated topic discovery for {domain}")
        
        # Step 1: Test connections
        connection_test = self.test_connections()
        if not connection_test["both_working"]:
            return {
                "status": "error",
                "error": "One or both services not working",
                "connection_test": connection_test
            }
        
        # Step 2: Analyze existing knowledge base
        print("📚 Analyzing existing knowledge base...")
        rag_analysis = self.rag_client.analyze_content_gaps(collection_name, focus_areas)
        
        # Step 3: Research trends (simplified for now)
        print("📈 Researching current trends...")
        trend_keywords = ", ".join(focus_areas[:3])  # Limit to avoid long queries
        trend_analysis = self.research_client.search_trends(domain, trend_keywords)
        
        # Step 4: Simple synthesis
        opportunities = []
        for topic, data in rag_analysis.get("gaps_by_topic", {}).items():
            coverage_depth = data.get("coverage_depth", "none")
            
            # Score based on content gap
            if coverage_depth in ["minimal", "none"]:
                gap_score = 9.0
            elif coverage_depth == "low":
                gap_score = 7.0
            elif coverage_depth == "medium":
                gap_score = 4.0
            else:
                gap_score = 2.0
            
            opportunities.append({
                "topic": topic,
                "content_gap_score": gap_score,
                "trend_score": 5.0,  # Default for now
                "competitive_opportunity": 5.0,  # Default for now
                "priority_ranking": gap_score * 0.6 + 5.0 * 0.4,  # Weighted score
                "coverage_depth": coverage_depth,
                "existing_chunks": data.get("chunks_found", 0)
            })
        
        # Sort by priority
        opportunities.sort(key=lambda x: x["priority_ranking"], reverse=True)
        
        return {
            "status": "completed",
            "domain": domain,
            "collection_analyzed": collection_name,
            "focus_areas": focus_areas,
            "raw_analysis": {
                "rag_content_gaps": rag_analysis,
                "trend_research": trend_analysis
            },
            "synthesized_opportunities": {
                "ranked_opportunities": opportunities,
                "top_5_recommendations": opportunities[:5],
                "content_strategy_insights": [
                    "Focus on topics with minimal existing coverage",
                    "Prioritize trending topics for better engagement",
                    "Create comprehensive guides for high-gap topics"
                ],
                "next_steps": [
                    "Select top 3-5 topics for immediate content development",
                    "Conduct fresh research on selected topics",
                    "Create detailed content briefs"
                ]
            },
            "execution_summary": {
                "total_topics_analyzed": len(focus_areas),
                "low_coverage_topics": len([t for t in opportunities if t["content_gap_score"] >= 7.0]),
                "opportunities_identified": len(opportunities)
            }
        }

# Test function
def test_integration():
    """Test the integration"""
    print("🧪 Testing MCP Integration...")
    
    discovery = IntegratedTopicDiscovery()
    test_result = discovery.test_connections()
    
    if test_result["both_working"]:
        print("✅ All services working! Ready for topic discovery.")
        return True
    else:
        print("❌ Some services not working. Check the errors above.")
        return False

if __name__ == "__main__":
    test_integration()
