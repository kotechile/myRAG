# Knowledge Gap Endpoints

The following endpoints are defined in `knowledge_gap_http_supabase.py` and integrated into the main Flask application.

## 1. `GET /gap_filler/analyze_knowledge_gaps`

*   **Purpose**: This endpoint analyzes all titles with a "NEW" status in the `Titles` table to identify potential knowledge gaps. It does not perform any research or content generation; it only identifies the gaps.
*   **Query Parameters**:
    *   `user_id` (optional): The ID of the user initiating the analysis. This is used for tracking and logging.
*   **Functionality**:
    1.  It retrieves all titles from the `Titles` table where the `status` is "NEW".
    2.  For each title, it uses the `KnowledgeGapAnalyzer` to analyze the title's content, keywords, and other metadata to determine if there is a knowledge gap.
    3.  If a gap is identified, it creates a `KnowledgeGap` object containing details about the gap, such as the data types needed, priority score, and specific requirements.
    4.  It returns a JSON response containing a list of all identified knowledge gaps, sorted by priority.
*   **Response**:
    *   A JSON object with a `knowledge_gaps` field, which is a list of identified gaps. Each gap object includes details like `title_id`, `focus_keyword`, `topic_category`, `priority_score`, and `data_types_needed`.

## 2. `POST /gap_filler/enhance_knowledge/<title_id>`

*   **Purpose**: This endpoint enhances the knowledge for a single title by identifying a knowledge gap, researching it, and adding the research results to the RAG system.
*   **URL Parameters**:
    *   `title_id`: The ID of the title to enhance.
*   **Request Body (JSON)**:
    *   `collection_name` (optional): The name of the collection to add the research documents to. If not provided, a name will be generated automatically.
    *   `merge_with_existing` (optional, boolean): If `True`, the system will try to merge the new documents with an existing collection for the title. Defaults to `True`.
    *   `user_id` (optional): The ID of the user initiating the enhancement.
*   **Functionality**:
    1.  It retrieves the specified title from the `Titles` table.
    2.  It uses the `KnowledgeGapAnalyzer` to identify a knowledge gap for the title.
    3.  If a gap is found, it uses the `MultiSourceResearcher` to research the required data types.
    4.  The research results are then passed to the `EnhancedRAGKnowledgeEnhancer`, which formats them and adds them as new documents to the specified RAG collection.
    5.  The title's status is updated to indicate that its knowledge has been enhanced.
*   **Response**:
    *   A JSON object confirming the success of the enhancement, including the number of documents added, the collection name, and a summary of the research.

## 3. `GET /gap_filler/status`

*   **Purpose**: This endpoint provides a status check of the knowledge gap filler system.
*   **Query Parameters**:
    *   `user_id` (optional): The ID of the user checking the status.
*   **Functionality**:
    *   It returns a JSON object with the operational status of the system, including whether the `linkup` integration is available and if the required environment variables are set.
*   **Response**:
    *   A JSON object with the `status` of the system (e.g., "operational") and details about the integration status.

## 4. `GET /gap_closure_status`

*   **Purpose**: This endpoint retrieves the knowledge gap closure status for a specific title or for all titles.
*   **Query Parameters**:
    *   `title_id` (optional): If provided, the status for only this title will be returned.
    *   `user_id` (optional): The ID of the user requesting the status.
*   **Functionality**:
    *   If a `title_id` is provided, it retrieves the gap closure status for that specific title.
    *   If no `title_id` is provided, it returns a summary of the gap closure status for all titles, including the total number of titles, the number of titles with closed gaps, and the number of titles with open gaps.
*   **Response**:
    *   A JSON object containing the gap closure status.

## 5. `POST /bulk_enhance_knowledge`

*   **Purpose**: This endpoint allows for the bulk enhancement of multiple titles at once.
*   **Request Body (JSON)**:
    *   `title_ids` (required, list): A list of title IDs to enhance.
    *   `user_id` (optional): The ID of the user initiating the bulk enhancement.
    *   `max_concurrent` (optional, int): The maximum number of titles to process concurrently. Defaults to 3.
    *   `merge_with_existing` (optional, boolean): Whether to merge with existing collections. Defaults to `True`.
    *   `collection_name` (optional): The collection to add the documents to.
*   **Functionality**:
    *   It iterates through the provided list of `title_ids` and processes each one sequentially (or concurrently, although the current implementation is sequential).
    *   For each title, it performs the same enhancement process as the `/gap_filler/enhance_knowledge/<title_id>` endpoint.
*   **Response**:
    *   A JSON object summarizing the bulk operation, including the number of successful and failed enhancements.

## 6. `POST /mark_gaps_closed/<title_id>`

*   **Purpose**: This endpoint allows for manually marking the knowledge gaps for a title as closed.
*   **URL Parameters**:
    *   `title_id`: The ID of the title to mark as closed.
*   **Request Body (JSON)**:
    *   `closure_method` (optional): The method used to close the gap (e.g., "manual").
    *   `closure_notes` (optional): Any notes about the closure.
    *   `user_id` (optional): The ID of the user marking the gaps as closed.
*   **Functionality**:
    *   It updates the specified title in the `Titles` table, setting the `knowledge_gaps_closed` field to `True`.
*   **Response**:
    *   A JSON object confirming that the gaps have been marked as closed.

## 7. `GET /additional_knowledge_documents/<title_id>`

*   **Purpose**: This endpoint retrieves all "additional knowledge" documents that have been added for a specific title.
*   **URL Parameters**:
    *   `title_id`: The ID of the title.
*   **Query Parameters**:
    *   `user_id` (optional): The ID of the user.
*   **Functionality**:
    *   It queries the `lindex_documents` table for all documents that are associated with the given `title_id` and have a `source_type` of `additional_gap_filler`.
*   **Response**:
    *   A JSON object containing a list of the additional knowledge documents, grouped by their data source type.

## 8. `POST /enhance_additional_knowledge/<title_id>`

*   **Purpose**: This endpoint is used to find and add *additional* knowledge sources for a title that *already has its initial knowledge gaps closed*. This is for expanding upon the existing knowledge.
*   **URL Parameters**:
    *   `title_id`: The ID of the title to enhance.
*   **Request Body (JSON)**:
    *   `user_id` (required): The ID of the user.
    *   `collection_name` (optional): The name of the collection to add the new documents to.
    *   `research_depth` (optional): The depth of the research to perform ('standard', 'deep', or 'comprehensive').
    *   `source_types` (optional, list): A list of source types to focus on (e.g., 'case_studies', 'expert_opinions').
    *   `exclude_existing_urls` (optional, boolean): If `True`, it will not add documents with URLs that already exist for the title.
*   **Functionality**:
    1.  It first verifies that the title's initial knowledge gaps are closed.
    2.  It then generates a new set of search queries based on the title's focus keyword and the requested `research_depth` and `source_types`.
    3.  It performs a search using the `linkup` client, filtering out any existing URLs.
    4.  The new sources are then added to the RAG system as "additional knowledge" documents.
*   **Response**:
    *   A JSON object summarizing the enhancement, including the number of new documents added.

## 9. `POST /enhance_additional_knowledge_bulk`

*   **Purpose**: This endpoint performs the "additional knowledge enhancement" in bulk for multiple titles.
*   **Request Body (JSON)**:
    *   `user_id` (required): The ID of the user.
    *   `title_ids` (required, list): A list of title IDs to enhance.
    *   Other parameters are the same as `/enhance_additional_knowledge/<title_id>`.
*   **Functionality**:
    *   It filters the provided `title_ids` to only include those that have their initial knowledge gaps closed.
    *   It then iterates through the eligible titles and performs the additional knowledge enhancement for each one.
*   **Response**:
    *   A JSON object summarizing the bulk operation.

## 10. `GET /validate_knowledge_gap_setup`

*   **Purpose**: This endpoint validates the overall setup of the knowledge gap system.
*   **Functionality**:
    *   It checks for the presence of required environment variables (`SUPABASE_URL`, `SUPABASE_KEY`, `LINKUP_API_KEY`, etc.).
    *   It checks for the availability of dependencies like the `linkup` library.
    *   It provides recommendations for improving the setup.
*   **Response**:
    *   A JSON object with the validation results.
