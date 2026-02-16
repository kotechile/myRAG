# Complete API Endpoints Reference

This document lists all available API endpoints and their input parameters for the RAG Knowledge Gap System.

## 🔍 Core RAG Query Endpoints

### 1. `POST /query_simple`
**Purpose**: Basic similarity search using vector similarity only.

**Request Body (JSON)**:
- `query` (required, string): The search query
- `collection_name` (required, string): Name of the collection to search
- `llm` (optional, string): LLM provider (default: "deepseek")
- `num_results` (optional, int): Number of results to return (default: 5)

### 2. `POST /query_hybrid_enhanced`
**Purpose**: Enhanced hybrid search with importance weighting favoring comprehensive sources.

**Request Body (JSON)**:
- `query` (required, string): The search query
- `collection_name` (required, string): Name of the collection to search
- `llm` (optional, string): LLM provider (default: "deepseek")
- `top_k` (optional, int): Number of top results to consider (default: 10)

### 3. `POST /query_agentic_iterative`
**Purpose**: Multi-round iterative search with AI-driven refinement.

**Request Body (JSON)**:
- `query` (required, string): The search query
- `collection_name` (required, string): Name of the collection to search
- `llm` (optional, string): LLM provider (default: "deepseek")
- `max_iterations` (optional, int): Maximum search rounds (default: 3)

### 4. `POST /query_truly_agentic`
**Purpose**: AI autonomously selects the best search strategy.

**Request Body (JSON)**:
- `query` (required, string): The search query
- `collection_name` (required, string): Name of the collection to search
- `llm` (optional, string): LLM provider (default: "deepseek")

### 5. `POST /query_agentic_fixed`
**Purpose**: Fixed agent with verbosity control and document tool selection.

**Request Body (JSON)**:
- `query` (required, string): The search query
- `collection_name` (required, string): Name of the collection to search
- `llm` (optional, string): LLM provider (default: "deepseek")
- `max_docs` (optional, int): Maximum documents for agent tools (default: 3)
- `verbose_mode` (optional, string): Verbosity level - "minimal", "balanced", "detailed" (default: "balanced")

### 6. `POST /query_async`
**Purpose**: Start an asynchronous query job with progress tracking.

**Request Body (JSON)**:
- `query` (required, string): The search query
- `collection_name` (required, string): Name of the collection to search
- `method` (required, string): Search method - "simple", "hybrid_enhanced", "agentic_iterative", "truly_agentic", "agentic_fixed"
- `llm` (optional, string): LLM provider (default: "deepseek")
- `num_results` (optional, int): For simple method (default: 5)
- `top_k` (optional, int): For hybrid_enhanced method (default: 10)
- `max_iterations` (optional, int): For agentic_iterative method (default: 3)
- `max_docs` (optional, int): For agentic_fixed method (default: 3)
- `verbose_mode` (optional, string): For agentic_fixed method (default: "balanced")

## 📊 Knowledge Gap Management Endpoints

### 7. `GET /gap_filler/analyze_knowledge_gaps`
**Purpose**: Analyze titles with "NEW" status to identify knowledge gaps.

**Query Parameters**:
- `user_id` (optional, string): User ID for tracking and logging

### 8. `POST /gap_filler/enhance_knowledge/<title_id>`
**Purpose**: Enhance knowledge for a single title by researching and adding content.

**URL Parameters**:
- `title_id` (required, string): ID of the title to enhance

**Request Body (JSON)**:
- `collection_name` (optional, string): Collection name for new documents
- `merge_with_existing` (optional, boolean): Merge with existing collections (default: true)
- `user_id` (optional, string): User ID for tracking

### 9. `POST /bulk_enhance_knowledge`
**Purpose**: Bulk enhancement of multiple titles.

**Request Body (JSON)**:
- `title_ids` (required, array): List of title IDs to enhance
- `user_id` (optional, string): User ID for tracking
- `max_concurrent` (optional, int): Max concurrent processing (default: 3)
- `merge_with_existing` (optional, boolean): Merge with existing collections (default: true)
- `collection_name` (optional, string): Collection name for documents

### 10. `GET /gap_filler/status`
**Purpose**: Get knowledge gap filler system status.

**Query Parameters**:
- `user_id` (optional, string): User ID for tracking

### 11. `GET /gap_closure_status`
**Purpose**: Get knowledge gap closure status for titles.

**Query Parameters**:
- `title_id` (optional, string): Specific title ID (if not provided, returns all)
- `user_id` (optional, string): User ID for tracking

### 12. `GET /analyze_knowledge_gaps_open_only`
**Purpose**: Get only titles with open knowledge gaps.

**Query Parameters**:
- `user_id` (optional, string): User ID for tracking

## 🔄 Additional Knowledge Enhancement

### 13. `POST /enhance_additional_knowledge/<title_id>`
**Purpose**: Add additional knowledge to titles with closed gaps.

**URL Parameters**:
- `title_id` (required, string): ID of the title to enhance

**Request Body (JSON)**:
- `user_id` (required, string): User ID for tracking
- `collection_name` (optional, string): Collection name for new documents
- `research_depth` (optional, string): Research depth - "standard", "deep", "comprehensive" (default: "standard")
- `source_types` (optional, array): Source types to focus on (e.g., ["case_studies", "expert_opinions"])
- `exclude_existing_urls` (optional, boolean): Exclude URLs that already exist (default: true)

### 14. `POST /enhance_additional_knowledge_bulk`
**Purpose**: Bulk additional knowledge enhancement for multiple titles.

**Request Body (JSON)**:
- `user_id` (required, string): User ID for tracking
- `title_ids` (required, array): List of title IDs to enhance
- `collection_name` (optional, string): Collection name for documents
- `research_depth` (optional, string): Research depth (default: "standard")
- `source_types` (optional, array): Source types to focus on
- `exclude_existing_urls` (optional, boolean): Exclude existing URLs (default: true)

## 📄 Document Management Endpoints

### 15. `POST /upload`
**Purpose**: Upload and parse document files.

**Request Body (multipart/form-data)**:
- `file` (required, file): Document file to upload
- `collection_name` (optional, string): Collection name for the document
- `source_type` (optional, string): Type of source document

### 16. `POST /chunk`
**Purpose**: Process document chunks and store in vector database.

**Request Body (JSON)**:
- `docid` (required, string): Document ID to process
- `collection_name` (required, string): Collection name for storage
- `source_type` (optional, string): Type of source document

### 17. `POST /generate_summary`
**Purpose**: Generate document summary from chunks.

**Request Body (JSON)**:
- `docid` (required, string): Document ID to summarize
- `collection_name` (required, string): Collection name
- `summary_type` (optional, string): Type of summary to generate

### 18. `POST /process_gap_filler_docs_manual`
**Purpose**: Manual processing for gap filler documents.

**Request Body (JSON)**:
- `user_id` (required, string): User ID for tracking
- `collection_name` (optional, string): Collection name for processing

## 📋 Status and Monitoring Endpoints

### 19. `GET /job_status/<job_id>`
**Purpose**: Get status of an asynchronous job.

**URL Parameters**:
- `job_id` (required, string): Job ID to check

### 20. `GET /embedding_status`
**Purpose**: Get current embedding configuration status.

**No parameters required**

### 21. `GET /test_linkup`
**Purpose**: Test Linkup API connection.

**Query Parameters**:
- `query` (optional, string): Test query (default: "real estate market trends")

### 22. `GET /validate_knowledge_gap_setup`
**Purpose**: Validate knowledge gap system setup.

**No parameters required**

## 🔍 Document Retrieval Endpoints

### 23. `GET /gap_filler_documents/<title_id>`
**Purpose**: Get all gap filler documents for a specific title.

**URL Parameters**:
- `title_id` (required, string): Title ID to query

**Query Parameters**:
- `user_id` (optional, string): User ID for filtering

### 24. `GET /titles_with_gap_fillers`
**Purpose**: Get all title IDs that have gap filler documents.

**Query Parameters**:
- `user_id` (optional, string): User ID for filtering

### 25. `GET /check_gap_filler_vector_status`
**Purpose**: Check vector store status for gap filler documents.

**Query Parameters**:
- `user_id` (optional, string): User ID for filtering

### 26. `GET /enhancement_details`
**Purpose**: Get detailed enhancement information for titles.

**Query Parameters**:
- `user_id` (optional, string): User ID for filtering
- `title_id` (optional, string): Specific title ID (if not provided, returns all)

### 27. `GET /enhancement_summary`
**Purpose**: Get quick summary of enhancements for dashboard.

**Query Parameters**:
- `user_id` (optional, string): User ID for filtering

## 🎯 Manual Action Suggestions

### 28. `POST /manual_suggestions/<title_id>`
**Purpose**: Get manual action suggestions for a title.

**URL Parameters**:
- `title_id` (required, string): Title ID for suggestions

**Request Body (JSON)**:
- `user_id` (required, string): User ID for tracking

### 29. `PATCH /manual_suggestions/<title_id>/<suggestion_id>/status`
**Purpose**: Update status of a manual suggestion.

**URL Parameters**:
- `title_id` (required, string): Title ID
- `suggestion_id` (required, string): Suggestion ID

**Request Body (JSON)**:
- `status` (required, string): New status for the suggestion
- `user_id` (required, string): User ID for security

### 30. `POST /manual_suggestions`
**Purpose**: Get all manual suggestions for a user.

**Request Body (JSON)**:
- `user_id` (required, string): User ID for filtering

### 31. `DELETE /manual_suggestions/<suggestion_id>`
**Purpose**: Delete a manual suggestion.

**URL Parameters**:
- `suggestion_id` (required, string): Suggestion ID to delete

**Request Body (JSON)**:
- `user_id` (required, string): User ID for security

## 🏥 Health Check Endpoints

### 32. `GET /query_hybrid_enhanced/health`
**Purpose**: Health check for knowledge gap system.

**No parameters required**

### 33. `GET /query_hybrid_enhanced/status`
**Purpose**: Status check for knowledge gap system.

**No parameters required**

### 34. `GET /query_hybrid_enhanced/ping`
**Purpose**: Ping check for knowledge gap system.

**No parameters required**

### 35. `GET /query_hybrid_enhanced/`
**Purpose**: Root endpoint for knowledge gap system.

**No parameters required**

### 36. `GET /test_agent`
**Purpose**: Test endpoint to verify agent functionality.

**No parameters required**

## 📝 Legacy Endpoints

### 37. `POST /enhance_knowledge/<title_id>`
**Purpose**: Legacy endpoint for enhancing single title (use `/gap_filler/enhance_knowledge/<title_id>` instead).

**URL Parameters**:
- `title_id` (required, string): Title ID to enhance

**Request Body (JSON)**:
- `user_id` (required, string): User ID for tracking

## 🔧 Common Response Formats

### Success Response
```json
{
  "status": "success",
  "data": { ... },
  "message": "Operation completed successfully"
}
```

### Error Response
```json
{
  "status": "error",
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

### Async Job Response
```json
{
  "job_id": "uuid-string",
  "status": "running|completed|failed",
  "progress": 0-100,
  "phase": "current_phase",
  "current_step": "current_step_description"
}
```

## 📋 Notes

- All endpoints return JSON responses
- Most endpoints support optional `user_id` for user tracking and security
- Collection names are used to organize documents in the vector database
- The system supports multiple LLM providers (deepseek, openai, etc.)
- Async endpoints return job IDs for progress tracking
- Health check endpoints are available for monitoring system status
