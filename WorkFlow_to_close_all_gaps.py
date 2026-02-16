#!/usr/bin/env python3
"""
Complete Knowledge Gap Closure Workflow
Processes all NEW titles with open gaps and fills them automatically
"""

import asyncio
import logging
from datetime import datetime
from knowledge_gap_http_supabase import EnhancedKnowledgeGapFillerOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GapClosureWorkflow:
    """Complete workflow for closing knowledge gaps."""
    
    def __init__(self):
        self.orchestrator = EnhancedKnowledgeGapFillerOrchestrator()
        
    async def process_all_open_gaps(self, collection_strategy="auto_detect", 
                                  merge_with_existing=True, dry_run=False):
        """Process all titles with open knowledge gaps."""
        
        logger.info("🚀 Starting complete gap closure workflow")
        logger.info(f"   Collection strategy: {collection_strategy}")
        logger.info(f"   Merge with existing: {merge_with_existing}")
        logger.info(f"   Dry run mode: {dry_run}")
        
        # Step 1: Analyze titles with open gaps
        logger.info("📊 Step 1: Analyzing titles with open knowledge gaps...")
        gaps = await self.orchestrator.gap_analyzer.analyze_new_titles_with_gap_filter()
        
        if not gaps:
            logger.info("✅ No titles with open knowledge gaps found!")
            return {
                "status": "completed",
                "message": "No open gaps to process",
                "gaps_found": 0,
                "titles_processed": 0
            }
        
        logger.info(f"🔍 Found {len(gaps)} titles with open knowledge gaps")
        
        # Step 2: Process gaps efficiently using batch consolidation
        results = {
            "gaps_found": len(gaps),
            "titles_processed": 0,
            "successful_closures": 0,
            "failed_closures": 0,
            "details": [],
            "batch_processing": False,
            "api_calls_saved": 0
        }
        
        # Use batch processing for multiple gaps (consolidates API calls)
        if len(gaps) > 1 and not dry_run:
            logger.info(f"🤖 Using batch processing with LLM consolidation for {len(gaps)} gaps")
            logger.info(f"   This will reduce API calls by grouping related gaps")
            
            try:
                # Batch research all gaps at once
                batch_results = await self.orchestrator.researcher.research_gaps_batch(
                    gaps=gaps,
                    user_id=None,
                    use_consolidation=True
                )
                
                results["batch_processing"] = True
                logger.info(f"✅ Batch research complete, processing results for each gap...")
                
                # Process each gap's research results
                for i, gap in enumerate(gaps, 1):
                    logger.info(f"📋 Processing {i}/{len(gaps)}: {gap.focus_keyword} (ID: {gap.title_id})")
                    
                    research_results = batch_results.get(gap.title_id)
                    if not research_results:
                        logger.warning(f"   ⚠️ No research results for {gap.focus_keyword}")
                        results["failed_closures"] += 1
                        results["details"].append({
                            "title_id": gap.title_id,
                            "focus_keyword": gap.focus_keyword,
                            "status": "no_research_results",
                            "error": "No results from batch processing"
                        })
                        continue
                    
                    # Continue with existing processing logic...
                    if research_results['total_sources'] == 0:
                        logger.warning(f"   ⚠️ No sources found for {gap.focus_keyword}")
                        results["details"].append({
                            "title_id": gap.title_id,
                            "focus_keyword": gap.focus_keyword,
                            "status": "no_sources_found",
                            "error": "No relevant sources found during research"
                        })
                        results["failed_closures"] += 1
                        continue
                    
                    logger.info(f"   📚 Found {research_results['total_sources']} sources")
                    
                    # Step 2b: Enhance RAG with collection control
                    collection_name = None
                    if collection_strategy != "auto_detect":
                        collection_name = collection_strategy
                    
                    logger.info(f"   🔗 Adding to RAG system...")
                    enhancement_result = await self.orchestrator.rag_enhancer.enhance_rag_for_title(
                        title_id=gap.title_id,
                        research_results=research_results,
                        collection_name=collection_name,
                        merge_with_existing=merge_with_existing
                    )
                    
                    if enhancement_result['success']:
                        logger.info(f"   ✅ Successfully closed gaps for {gap.focus_keyword}")
                        logger.info(f"      📊 Added {enhancement_result['documents_added']} documents")
                        logger.info(f"      🗂️ Collection: {enhancement_result['collection_name']}")
                        
                        results["successful_closures"] += 1
                        results["details"].append({
                            "title_id": gap.title_id,
                            "focus_keyword": gap.focus_keyword,
                            "status": "successfully_closed",
                            "documents_added": enhancement_result['documents_added'],
                            "collection_name": enhancement_result['collection_name'],
                            "sources_researched": research_results['total_sources']
                        })
                    else:
                        logger.error(f"   ❌ Failed to enhance RAG: {enhancement_result.get('error')}")
                        results["failed_closures"] += 1
                        results["details"].append({
                            "title_id": gap.title_id,
                            "focus_keyword": gap.focus_keyword,
                            "status": "enhancement_failed",
                            "error": enhancement_result.get('error')
                        })
                    
                    results["titles_processed"] += 1
                    
            except Exception as e:
                logger.error(f"❌ Batch processing failed: {e}")
                import traceback
                logger.error(f"   Traceback: {traceback.format_exc()}")
                logger.info("   🔄 Falling back to individual gap processing")
                results["batch_processing"] = False
                # Fall through to individual processing
        
        # Individual processing (fallback or when batch fails)
        if not results["batch_processing"]:
            logger.info(f"📋 Processing {len(gaps)} gap(s) individually")
            for i, gap in enumerate(gaps, 1):
                logger.info(f"📋 Processing {i}/{len(gaps)}: {gap.focus_keyword} (ID: {gap.title_id})")
                
                try:
                    if dry_run:
                        logger.info(f"   🧪 DRY RUN: Would process title {gap.title_id}")
                        results["details"].append({
                            "title_id": gap.title_id,
                            "focus_keyword": gap.focus_keyword,
                            "status": "dry_run_skipped",
                            "data_types_needed": gap.data_types_needed
                        })
                        continue
                    
                    # Step 2a: Research the gap
                    logger.info(f"   🔍 Researching knowledge gap...")
                    research_results = await self.orchestrator.researcher.research_knowledge_gap(gap)
                    
                    # Check if Manus-wide research is needed due to insufficient results
                    if research_results.get('manus_suggestion_needed', False):
                        logger.warning(f"   ⚠️ Insufficient research results for {gap.focus_keyword} (Quality Score: {research_results.get('research_quality_score', 0)}/100)")
                        
                        # Generate Manus-wide research suggestion
                        if hasattr(self.orchestrator, 'suggestions_generator') and self.orchestrator.suggestions_generator:
                            try:
                                manus_suggestions = await self.orchestrator.suggestions_generator.suggest_manus_wide_research(gap, research_results)
                                if manus_suggestions:
                                    # Save suggestion to database
                                    await self.orchestrator.suggestions_generator.save_suggestions_to_database(
                                        title_id=gap.title_id,
                                        suggestions=manus_suggestions,
                                        research_context={
                                            'focus_keyword': gap.focus_keyword,
                                            'research_quality_score': research_results.get('research_quality_score', 0),
                                            'total_sources': research_results.get('total_sources', 0),
                                            'research_summary': research_results.get('research_summary', '')
                                        },
                                        user_id=None
                                    )
                                    logger.info(f"   💡 Generated Manus-wide research suggestion for low-quality results")
                            except Exception as e:
                                logger.error(f"   ❌ Error generating Manus suggestion: {e}")
                    
                    if research_results['total_sources'] == 0:
                        logger.warning(f"   ⚠️ No sources found for {gap.focus_keyword}")
                        results["details"].append({
                            "title_id": gap.title_id,
                            "focus_keyword": gap.focus_keyword,
                            "status": "no_sources_found",
                            "error": "No relevant sources found during research"
                        })
                        results["failed_closures"] += 1
                        continue
                    
                    logger.info(f"   📚 Found {research_results['total_sources']} sources")
                    
                    # Step 2b: Enhance RAG with collection control
                    collection_name = None
                    if collection_strategy != "auto_detect":
                        collection_name = collection_strategy
                    
                    logger.info(f"   🔗 Adding to RAG system...")
                    enhancement_result = await self.orchestrator.rag_enhancer.enhance_rag_for_title(
                        title_id=gap.title_id,
                        research_results=research_results,
                        collection_name=collection_name,
                        merge_with_existing=merge_with_existing
                    )
                    
                    if enhancement_result['success']:
                        logger.info(f"   ✅ Successfully closed gaps for {gap.focus_keyword}")
                        logger.info(f"      📊 Added {enhancement_result['documents_added']} documents")
                        logger.info(f"      🗂️ Collection: {enhancement_result['collection_name']}")
                        
                        results["successful_closures"] += 1
                        results["details"].append({
                            "title_id": gap.title_id,
                            "focus_keyword": gap.focus_keyword,
                            "status": "successfully_closed",
                            "documents_added": enhancement_result['documents_added'],
                            "collection_name": enhancement_result['collection_name'],
                            "sources_researched": research_results['total_sources']
                        })
                    else:
                        logger.error(f"   ❌ Failed to enhance RAG: {enhancement_result.get('error')}")
                        results["failed_closures"] += 1
                        results["details"].append({
                            "title_id": gap.title_id,
                            "focus_keyword": gap.focus_keyword,
                            "status": "enhancement_failed",
                            "error": enhancement_result.get('error')
                        })
                    
                    results["titles_processed"] += 1
                
            except Exception as e:
                logger.error(f"   ❌ Error processing {gap.title_id}: {e}")
                results["failed_closures"] += 1
                results["details"].append({
                    "title_id": gap.title_id,
                    "focus_keyword": gap.focus_keyword,
                    "status": "processing_error",
                    "error": str(e)
                })
        
        # Step 3: Summary
        logger.info("📈 Gap closure workflow completed!")
        logger.info(f"   📊 Gaps found: {results['gaps_found']}")
        logger.info(f"   ✅ Successfully closed: {results['successful_closures']}")
        logger.info(f"   ❌ Failed: {results['failed_closures']}")
        logger.info(f"   📋 Processed: {results['titles_processed']}")
        
        results["status"] = "completed"
        results["completion_timestamp"] = datetime.now().isoformat()
        
        return results
    
    async def get_workflow_status(self):
        """Get current status of gap closure across all titles."""
        
        try:
            # Get comprehensive status
            status = await self.orchestrator.rag_enhancer.get_gap_closure_status()
            
            # Add workflow insights
            if "total_titles" in status:
                workflow_insights = {
                    "workflow_progress": {
                        "completion_percentage": round(
                            (status["gaps_closed_count"] / status["total_titles"]) * 100, 1
                        ) if status["total_titles"] > 0 else 0,
                        "remaining_work": status["gaps_open_count"],
                        "new_titles_pending": status["new_status_count"]
                    },
                    "recommendations": []
                }
                
                # Add recommendations
                if status["gaps_open_count"] > 0:
                    workflow_insights["recommendations"].append(
                        f"Run gap closure workflow for {status['gaps_open_count']} remaining titles"
                    )
                
                if status["new_status_count"] > status["gaps_open_count"]:
                    workflow_insights["recommendations"].append(
                        "Some NEW titles may have gaps closed but status not updated"
                    )
                
                status["workflow_insights"] = workflow_insights
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return {"error": str(e)}

# CLI interface for running the workflow
async def main():
    """Main CLI interface."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Gap Closure Workflow")
    parser.add_argument("--action", choices=["process", "status"], default="status",
                       help="Action to perform")
    parser.add_argument("--collection", default="auto_detect",
                       help="Target collection name")
    parser.add_argument("--merge", action="store_true", default=True,
                       help="Merge with existing collections")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run in dry-run mode (no actual changes)")
    
    args = parser.parse_args()
    
    workflow = GapClosureWorkflow()
    
    if args.action == "process":
        results = await workflow.process_all_open_gaps(
            collection_strategy=args.collection,
            merge_with_existing=args.merge,
            dry_run=args.dry_run
        )
        
        print("\n" + "="*50)
        print("WORKFLOW RESULTS")
        print("="*50)
        print(f"Status: {results['status']}")
        print(f"Gaps found: {results['gaps_found']}")
        print(f"Successfully closed: {results['successful_closures']}")
        print(f"Failed: {results['failed_closures']}")
        
        if results.get('details'):
            print("\nDetailed Results:")
            for detail in results['details']:
                print(f"  - {detail['title_id']}: {detail['status']}")
    
    elif args.action == "status":
        status = await workflow.get_workflow_status()
        
        print("\n" + "="*50)
        print("GAP CLOSURE STATUS")
        print("="*50)
        
        if "total_titles" in status:
            print(f"Total titles: {status['total_titles']}")
            print(f"Gaps closed: {status['gaps_closed_count']}")
            print(f"Gaps open: {status['gaps_open_count']}")
            print(f"NEW status: {status['new_status_count']}")
            
            if "workflow_insights" in status:
                insights = status["workflow_insights"]
                print(f"Completion: {insights['workflow_progress']['completion_percentage']}%")
                
                if insights["recommendations"]:
                    print("\nRecommendations:")
                    for rec in insights["recommendations"]:
                        print(f"  - {rec}")
        else:
            print(f"Error: {status.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())