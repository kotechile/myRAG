# Balance Emphasis Guide

## Overview

The RAG system now supports configurable balance emphasis to control the prioritization between comprehensive sources (books, long documents) and recent content (articles, news). This addresses the issue where books were heavily prioritized over smaller articles, which was problematic for queries requiring current information.

## Balance Emphasis Options

### 1. `news_focused`
- **Best for**: Current events, recent developments, breaking news, trending topics
- **Prioritizes**: Articles, news, recent content
- **Document type weights**:
  - `news`: 1.3x
  - `article`: 1.2x
  - `document`: 1.0x
  - `pdf`: 0.9x
  - `book`: 0.8x
  - `email`: 0.7x
- **Size weighting**: Less aggressive (max 1.2x for large documents)
- **Chunk weighting**: Reduced bias toward documents with many chunks

### 2. `balanced`
- **Best for**: General queries, mixed content needs
- **Prioritizes**: Moderate balance between comprehensive and recent content
- **Document type weights**:
  - `news`: 1.1x
  - `article`: 1.1x
  - `document`: 1.0x
  - `pdf`: 1.1x
  - `book`: 1.2x
  - `email`: 0.8x
- **Size weighting**: Moderate (max 1.5x for large documents)
- **Chunk weighting**: Moderate bias toward documents with many chunks

### 3. `comprehensive` (default)
- **Best for**: Detailed analysis, research, comprehensive studies
- **Prioritizes**: Books, long documents, comprehensive sources
- **Document type weights**:
  - `book`: 1.5x
  - `pdf`: 1.3x
  - `document`: 1.0x
  - `article`: 0.9x
  - `news`: 0.85x
  - `email`: 0.8x
- **Size weighting**: Aggressive (max 2.0x for large documents)
- **Chunk weighting**: Strong bias toward documents with many chunks

## Auto-Detection

The system automatically detects the appropriate balance emphasis based on query characteristics:

### News-Focused Indicators
- Keywords: "latest", "recent", "current", "today", "this week", "this month"
- Years: "2024", "2025"
- Terms: "breaking", "news", "update", "trending", "happening now", "just released"

### Comprehensive Indicators
- Keywords: "comprehensive", "detailed", "complete", "thorough", "in-depth"
- Phrases: "everything about", "all about", "full analysis", "complete guide"
- Terms: "comprehensive study", "extensive", "exhaustive"

### Balanced (Default)
- Used when no strong indicators are detected
- Provides moderate balance between all content types

## API Usage

### Query Parameters

All query endpoints now support the `balance_emphasis` parameter:

```json
{
  "query": "What are the latest AI developments?",
  "collection_name": "your_collection",
  "balance_emphasis": "news_focused",
  "num_results": 5
}
```

### Supported Endpoints

- `POST /query_simple`
- `POST /query_hybrid_enhanced`
- `POST /query_agentic_fixed`
- `POST /query_agentic_iterative`
- `POST /query_truly_agentic`
- `POST /query_async`

### Example Requests

#### News-Focused Query
```bash
curl -X POST http://localhost:5000/query_simple \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in quantum computing in 2024?",
    "collection_name": "tech_collection",
    "balance_emphasis": "news_focused",
    "num_results": 5
  }'
```

#### Comprehensive Research Query
```bash
curl -X POST http://localhost:5000/query_hybrid_enhanced \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Give me a comprehensive analysis of machine learning algorithms",
    "collection_name": "ml_collection",
    "balance_emphasis": "comprehensive",
    "top_k": 10
  }'
```

#### Auto-Detection Query
```bash
curl -X POST http://localhost:5000/query_simple \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does neural network training work?",
    "collection_name": "ai_collection",
    "num_results": 5
  }'
```

## Implementation Details

### Weight Calculation

The importance weight is calculated using the formula:
```
weight = size_weight × type_weight × chunk_weight
```

Where:
- `size_weight`: Based on document size (varies by balance emphasis)
- `type_weight`: Based on document type (varies by balance emphasis)
- `chunk_weight`: Based on number of chunks (varies by balance emphasis)

### Document Type Detection

The system recognizes these document types:
- `book`: Books and comprehensive texts
- `pdf`: PDF documents
- `document`: General documents
- `article`: Articles and blog posts
- `news`: News articles and current events
- `email`: Email communications

### Query Type Detection

The system analyzes query text for indicators:
- **News indicators**: "latest", "recent", "current", "2024", "2025", etc.
- **Comprehensive indicators**: "comprehensive", "detailed", "complete", etc.
- **Scoring**: Queries with 2+ indicators trigger the respective balance mode

## Testing

Use the provided test script to verify the balance emphasis functionality:

```bash
python test_balance_emphasis.py
```

Make sure to:
1. Update `COLLECTION_NAME` in the test script
2. Ensure your RAG server is running on `http://localhost:5000`
3. Have some test documents in your collection

## Best Practices

### When to Use Each Balance Mode

1. **News-Focused**: 
   - Current events and recent developments
   - Breaking news and updates
   - Trending topics and latest information

2. **Balanced**:
   - General knowledge questions
   - Mixed content requirements
   - When unsure about content needs

3. **Comprehensive**:
   - Research and analysis
   - Detailed technical topics
   - Academic and scholarly content

### Query Optimization

- Use specific keywords to trigger auto-detection
- Be explicit about your content needs in the query
- Consider using the appropriate balance emphasis for your use case
- Test different balance settings to find the best fit

## Migration Notes

- Existing queries will continue to work with auto-detection
- The default behavior remains `comprehensive` for backward compatibility
- New `balance_emphasis` parameter is optional
- All query methods support the new parameter

## Troubleshooting

### Common Issues

1. **Queries still favor books**: Check if you're using `news_focused` balance
2. **Auto-detection not working**: Ensure query contains clear indicators
3. **Inconsistent results**: Verify document metadata includes correct `source_type`

### Debug Information

The response includes enhancement information:
```json
{
  "enhancement_info": {
    "importance_weighting_applied": true,
    "balance_emphasis": "news_focused",
    "document_scores": {...}
  }
}
```

This helps debug why certain documents were prioritized.
