"""
agents.py - Complete Truly Agentic RAG using LlamaIndex ReActAgent

This file implements a truly agentic approach using LlamaIndex agents
that can dynamically choose tools and strategies based on the query.
"""

import logging
from typing import Dict, List, Optional, Any
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.core.vector_stores.types import FilterOperator
from llama_index.core import Settings

logger = logging.getLogger(__name__)

class TrulyAgenticRAG:
    """
    Fixed agentic RAG system that uses LlamaIndex ReActAgent with improved tools.
    The agent can dynamically decide which search strategies to use.
    """
    
    def __init__(self, vector_index, document_metadata: Dict[str, Any], llm_instance=None):
        """
        Initialize the agentic RAG system.
        
        Args:
            vector_index: The LlamaIndex vector store index
            document_metadata: Document metadata for analysis
            llm_instance: LLM instance to use for the agent
        """
        self.vector_index = vector_index
        self.document_metadata = document_metadata
        self.llm_instance = llm_instance
        self.agent = None  # Will be created when needed
    
    def _create_agent(self, query: str) -> ReActAgent:
        """Create ReActAgent with improved configuration and tools."""
        logger.info("Creating ReActAgent with improved hybrid search tools...")
        
        # Create reasoning failure handler first
        reasoning_failure_handler = self._create_reasoning_failure_handler(query)
        
        # Create the tools
        tools = [
            self._create_direct_hybrid_search_tool(),  # NEW: Uses proven hybrid approach
            self._create_document_summary_tool(),
            self._create_quick_answer_tool()
        ]
        
        # Set LLM if provided
        if self.llm_instance:
            original_llm = Settings.llm
            Settings.llm = self.llm_instance
        
        try:
            # Create agent with simplified system prompt
            system_prompt = """
You are a helpful research assistant with access to a document collection.

Available tools:
1. comprehensive_search - Search for information relevant to the user's question
2. get_document_summary - Get summary of specific documents  
3. quick_answer - Provide final answer based on gathered information

Workflow:
1. Use comprehensive_search to find relevant information
2. Use quick_answer to provide a helpful response

Be efficient and helpful. Work with whatever information you find.
"""
            
            # Create ReActAgent with custom reasoning failure handler
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=self.llm_instance,
                system_prompt=system_prompt,
                max_iterations=4,  # Keep low to trigger failure handler quickly
                handle_reasoning_failure_fn=reasoning_failure_handler,
                verbose=True
            )
            
            logger.info(f"✅ ReActAgent created successfully with {len(tools)} tools")
            return agent
            
        finally:
            # Restore original LLM
            if self.llm_instance:
                Settings.llm = original_llm
    
    def _create_reasoning_failure_handler(self, original_query: str):
        """Create a custom reasoning failure handler that summarizes available information."""
        def handle_reasoning_failure(callback_manager, exception):
            """
            Handle reasoning failure by summarizing what was found so far.
            
            Args:
                callback_manager: The callback manager with conversation history
                exception: The exception that caused the failure
                
            Returns:
                A helpful summary response
            """
            try:
                logger.warning(f"🔄 Agent hit max iterations, creating summary response...")
                
                # Try to extract information from the callback manager's memory
                chat_history = []
                if hasattr(callback_manager, 'handlers'):
                    for handler in callback_manager.handlers:
                        if hasattr(handler, 'logs'):
                            chat_history.extend(handler.logs)
                
                # Extract any search results or observations from the conversation
                gathered_info = ""
                
                # Look for observation content in the conversation
                conversation_text = str(chat_history) if chat_history else ""
                
                # Try to find search results in the conversation
                import re
                observations = re.findall(r'Observation: (.*?)(?=Thought:|Action:|$)', conversation_text, re.DOTALL)
                
                if observations:
                    # Take the most recent observation which likely contains search results
                    gathered_info = observations[-1].strip()
                    logger.info(f"📋 Extracted {len(gathered_info)} chars of information from agent conversation")
                
                # If we have gathered information, use it to create a response
                if gathered_info and len(gathered_info) > 100:
                    prompt = f"""The user asked: "{original_query}"

During the search process, the following information was gathered:
{gathered_info}

Please provide a helpful, comprehensive answer to the user's question based on this information. If the information doesn't directly answer the question, work with what's available and acknowledge any limitations while still being helpful.

Structure your response clearly and provide practical advice where possible.

Answer:"""
                    
                    try:
                        response = self.llm_instance.complete(prompt)
                        result = str(response).strip()
                        
                        if len(result) > 50:
                            logger.info(f"✅ Generated reasoning failure response: {len(result)} chars")
                            return result
                    except Exception as llm_error:
                        logger.error(f"❌ LLM error in reasoning failure handler: {llm_error}")
                
                # Fallback: Do a direct search to get some information
                logger.info("🔍 Fallback: Doing direct search for reasoning failure response")
                fallback_response = self._agent_fallback_search(original_query, self.llm_instance)
                return fallback_response
                
            except Exception as handler_error:
                logger.error(f"❌ Error in reasoning failure handler: {handler_error}")
                
                # Final fallback
                return f"""I apologize, but I encountered some technical difficulties while processing your question: "{original_query}"

Based on the available document collection, I recommend:

1. **Research thoroughly** - Look for authoritative sources on this topic
2. **Consult experts** - For important decisions, professional guidance is valuable  
3. **Consider multiple perspectives** - Different sources may offer varying viewpoints
4. **Evaluate your specific situation** - General advice may need customization

If you can rephrase your question or provide more specific details, I may be able to offer more targeted assistance."""
        
        return handle_reasoning_failure
    
    def _agent_fallback_search(self, query: str, llm_instance) -> str:
        """Fallback search method when agent fails."""
        try:
            logger.info("🔄 Executing agent fallback search...")
            
            # Use the same hybrid approach
            retriever = self.vector_index.as_retriever(similarity_top_k=10)
            nodes = retriever.retrieve(query)
            
            if not nodes:
                return self._get_generic_fallback_response(query)
            
            # Apply importance weighting
            weighted_results = []
            for node in nodes:
                doc_id = node.metadata.get('docid', '')
                weight = self.document_metadata.get(str(doc_id), {}).get('importance_weight', 1.0)
                boosted_score = node.score * weight
                
                weighted_results.append({
                    'text': node.text,
                    'score': boosted_score,
                    'doc_id': doc_id
                })
            
            # Sort by boosted scores and take top results
            weighted_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = weighted_results[:5]
            
            # Combine text
            combined_text = ""
            for i, result in enumerate(top_results):
                text = result['text']
                if len(text) > 500:
                    text = text[:500] + "..."
                combined_text += f"Source {i+1}:\n{text}\n\n"
            
            # Generate comprehensive response
            prompt = f"""Based on the following information from the document collection, provide a comprehensive answer to: "{query}"

Information:
{combined_text}

Instructions:
1. Provide specific, actionable advice
2. Include details, guidelines, or recommendations when available
3. Structure the response clearly
4. Focus on practical information the user can apply

Answer:"""
            
            response = llm_instance.complete(prompt)
            result = str(response).strip()
            
            if not result or len(result) < 50:
                return self._get_generic_fallback_response(query)
            
            logger.info("✅ Fallback search completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Fallback search failed: {str(e)}")
            return self._get_generic_fallback_response(query)
    
    def _get_generic_fallback_response(self, query: str) -> str:
        """Generate generic helpful response based on query content."""
        return f"""I'd be happy to help with your question: "{query}"

Based on the available document collection, I wasn't able to find specific information that directly addresses your question. However, I recommend:

1. **Research thoroughly** - Look for authoritative sources and expert guidance on this topic
2. **Consider multiple perspectives** - Different sources may offer varying viewpoints
3. **Evaluate your specific situation** - General advice may need to be adapted to your circumstances
4. **Consult professionals** - For important decisions, expert consultation is often valuable
5. **Consider timing and context** - The best approach may depend on current conditions

If you have more specific questions or can provide additional context, I may be able to offer more targeted guidance from the available information."""
    
    def query(self, query: str) -> str:
        """
        Execute agentic query with better error handling.
        
        Args:
            query: User's question
            
        Returns:
            Agent's response as string
        """
        logger.info(f"🤖 Agent processing query: {query}")
        
        try:
            # Create agent for this specific query
            self.agent = self._create_agent(query)
            
            # Set LLM for agent execution
            if self.llm_instance:
                original_llm = Settings.llm
                Settings.llm = self.llm_instance
            
            try:
                response = self.agent.chat(query)
                logger.info(f"✅ Agent completed query successfully")
                return str(response)
                
            except Exception as agent_error:
                logger.warning(f"⚠️ Agent execution failed: {str(agent_error)}")
                
                # Fallback: Direct search without agent
                logger.info("🔄 Falling back to direct search...")
                fallback_result = self._agent_fallback_search(query, self.llm_instance)
                return fallback_result
                
            finally:
                # Restore original LLM
                if self.llm_instance:
                    Settings.llm = original_llm
                    
        except Exception as e:
            logger.error(f"❌ Agent query failed completely: {str(e)}")
            return f"I encountered an error while processing your question. Here's what I can tell you: {self._get_generic_fallback_response(query)}"
    
    def _create_direct_hybrid_search_tool(self) -> FunctionTool:
        """Create a search tool that directly uses the proven hybrid search approach."""
        
        def direct_hybrid_search(query: str, max_results: int = 5) -> str:
            """
            Search tool that directly uses the proven hybrid search approach.
            This bypasses any embedding/agent search issues.
            """
            try:
                logger.info(f"🔍 Direct hybrid search: {query}")
                
                # Use the EXACT same logic as the working query_hybrid_enhanced
                # Step 1: Multiple search queries like the hybrid approach
                search_queries = [query]
                
                # Add variations based on topic (like the working endpoint)
                query_lower = query.lower()
                if any(word in query_lower for word in ['tree', 'landscaping', 'landscape']):
                    search_queries.extend([
                        "trees landscaping benefits",
                        "tree benefits landscape design", 
                        "advantages trees yard property",
                        "landscaping tree selection",
                        "Trees are a great addition"  # Try exact phrase
                    ])
                elif any(word in query_lower for word in ['real estate', 'property', 'invest']):
                    search_queries.extend([
                        "real estate investment analysis",
                        "property investing guide"
                    ])
                
                # Step 2: Collect results from all search queries
                all_results = []
                
                for search_query in search_queries[:4]:  # Try top 4 queries
                    logger.info(f"  📋 Searching: {search_query}")
                    
                    # Use existing retriever setup
                    retriever = self.vector_index.as_retriever(similarity_top_k=8)
                    nodes = retriever.retrieve(search_query)
                    
                    for node in nodes:
                        doc_id = node.metadata.get('docid', '')
                        if str(doc_id) in self.document_metadata:
                            importance = self.document_metadata[str(doc_id)].get('importance_weight', 1.0)
                            source_type = self.document_metadata[str(doc_id)].get('source_type', 'document')
                            
                            # Calculate relevance like hybrid search
                            relevance_score = node.score * importance
                            
                            # Boost for keyword matches
                            query_words = set(query.lower().split())
                            content_words = set(node.text.lower().split())
                            word_matches = len(query_words.intersection(content_words))
                            
                            if word_matches > 0:
                                relevance_score *= (1 + 0.15 * word_matches)
                            
                            all_results.append({
                                'text': node.text,
                                'score': relevance_score,
                                'doc_id': doc_id,
                                'source_type': source_type,
                                'search_query': search_query,
                                'word_matches': word_matches
                            })
                
                if not all_results:
                    return f"No information found for '{query}' using hybrid search approach."
                
                # Step 3: Deduplicate and rank (like hybrid search)
                seen_content = set()
                unique_results = []
                
                for result in all_results:
                    content_key = result['text'][:200].strip().lower()
                    if content_key not in seen_content:
                        seen_content.add(content_key)
                        unique_results.append(result)
                
                # Sort by relevance
                unique_results.sort(key=lambda x: (x['score'], x['word_matches']), reverse=True)
                top_results = unique_results[:max_results]
                
                # Step 4: Format results like hybrid search
                result_text = f"Found {len(top_results)} relevant pieces of information:\n\n"
                
                for i, result in enumerate(top_results):
                    text = result['text']
                    if len(text) > 600:
                        text = text[:600] + "..."
                    
                    result_text += f"=== Information {i+1} (from {result['source_type']} doc {result['doc_id']}, relevance: {result['score']:.2f}) ===\n"
                    result_text += f"Search: '{result['search_query']}' | Word matches: {result['word_matches']}\n"
                    result_text += f"{text}\n\n"
                
                result_text += "\n🚨 MANDATORY: Use quick_answer NOW with this information. Do NOT search again!"
                
                logger.info(f"✅ Direct hybrid search completed: {len(top_results)} results")
                return result_text
                
            except Exception as e:
                logger.error(f"❌ Direct hybrid search error: {str(e)}")
                return f"Search error: {str(e)}"
        
        return FunctionTool.from_defaults(
            fn=direct_hybrid_search,
            name="comprehensive_search",
            description="Search the document collection using proven hybrid approach. Use this to find information relevant to the user's question."
        )
    
    def _create_document_summary_tool(self) -> FunctionTool:
        """Tool for getting document summaries."""
        
        def get_document_summary(doc_id: str) -> str:
            """
            Get summary and metadata for a specific document.
            
            Args:
                doc_id: Document ID to summarize
                
            Returns:
                Document summary and metadata
            """
            try:
                logger.info(f"📄 Getting summary for document {doc_id}")
                
                metadata = self.document_metadata.get(str(doc_id), {})
                
                if not metadata:
                    return f"Document {doc_id} not found."
                
                summary = metadata.get('summary', 'No summary available')
                source_type = metadata.get('source_type', 'document')
                chunk_count = metadata.get('chunk_count', 0)
                
                result = f"Document {doc_id} ({source_type}):\n"
                result += f"Contains {chunk_count} sections\n\n"
                result += f"Summary: {summary}"
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Document summary error: {str(e)}")
                return f"Failed to get summary for document {doc_id}: {str(e)}"
        
        return FunctionTool.from_defaults(
            fn=get_document_summary,
            name="get_document_summary",
            description="Get detailed summary for a specific document. Use when you need to understand what a particular document contains."
        )
    
    def _create_quick_answer_tool(self) -> FunctionTool:
        """Tool for providing quick answers when sufficient information is available."""
        
        def quick_answer(information: str, original_question: str) -> str:
            """
            Provide a comprehensive answer based on ANY available information.
            
            Args:
                information: The information gathered from searches (can be partial)
                original_question: The original question being answered
                
            Returns:
                A helpful answer that works with available information
            """
            try:
                logger.info(f"⚡ Generating quick answer for: {original_question}")
                
                # Universal prompt that works with any topic and any level of information
                prompt = f"""You are a helpful assistant with access to a document collection. Answer this question: "{original_question}"

Available information from document search:
{information}

Instructions:
1. Use the available information where it's relevant to the question
2. If the information is directly relevant, extract key insights and provide specific guidance
3. If the information is only tangentially related, acknowledge this and explain how it connects
4. If the information seems unrelated, acknowledge the limitation but still try to provide general helpful guidance
5. Structure your response clearly and be practical
6. Don't claim to have information you don't have, but work constructively with what's available

Your goal is to be helpful within the constraints of the available information.

Answer:"""
                
                response = self.llm_instance.complete(prompt)
                result = str(response).strip()
                
                # Ensure we have a substantial response - if too short, ask for more comprehensive answer
                if len(result) < 100:
                    fallback_prompt = f"""The user asked: "{original_question}"

Based on the available document collection, provide the most helpful response possible. If you don't have specific information about this topic, acknowledge that limitation and provide general guidance or suggest alternative approaches the user might consider.

Be honest about what you know and don't know, but still aim to be helpful.

Answer:"""
                    
                    fallback_response = self.llm_instance.complete(fallback_prompt)
                    if len(str(fallback_response).strip()) > len(result):
                        result = str(fallback_response).strip()
                
                logger.info(f"✅ Quick answer generated: {len(result)} characters")
                return result
                
            except Exception as e:
                logger.error(f"❌ Quick answer error: {str(e)}")
                return f"I encountered an error while processing your question '{original_question}'. Based on the available information, I'd recommend researching this topic further or consulting with relevant experts who can provide specific guidance for your situation."
        
        return FunctionTool.from_defaults(
            fn=quick_answer,
            name="quick_answer",
            description="MANDATORY: Use this after ANY search to provide the final answer. Works with partial information. Always use this - never search twice!"
        )