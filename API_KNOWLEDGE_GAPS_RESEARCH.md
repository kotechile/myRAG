# APIs for Knowledge Gap Identification and Research Recommendations

This document identifies the APIs that detect:
1. **Additional knowledge that is not online** (books, datasets, expert interviews) that needs manual research
2. **Need for deeper research** with advanced tools like Manus

## 📊 Summary

### APIs for Identifying Offline Knowledge Needs

**Primary Endpoint:**
- **`POST /manual_suggestions/<title_id>`** - Retrieves all manual action suggestions for a title

**When Suggestions Are Generated:**
- Automatically during knowledge gap enhancement via:
  - `POST /gap_filler/enhance_knowledge/<title_id>`
  - `POST /bulk_enhance_knowledge`
  - `POST /enhance_additional_knowledge/<title_id>`

**Action Types Identified:**
1. **`textbook`** - Textbooks and educational materials needed
2. **`dataset`** - Datasets and statistical data to acquire
3. **`expert_interview`** - Expert interviews to conduct
4. **`tool_development`** - Tools to develop for content enhancement
5. **`manus_wide_research`** - Advanced research with Manus tools

### APIs for Manus-Wide Research Detection

**Primary Endpoint:**
- **`POST /manual_suggestions/<title_id>`** - Returns `manus_wide_research` suggestions when research quality is insufficient

**Trigger Conditions:**
- Automatically triggered when `research_quality_score < 30`
- Quality score calculated during:
  - `POST /gap_filler/enhance_knowledge/<title_id>`
  - `POST /bulk_enhance_knowledge`

**Assessment Logic:**
- Calculated in `_calculate_research_quality()` method
- Factors:
  - Total sources found (max 40 points)
  - Source diversity (max 30 points)
  - Quality sources (academic, government, industry) (max 30 points)
- Score < 30 triggers Manus-wide research suggestion

## 🔍 Detailed API Endpoints

### 1. Get Manual Action Suggestions for a Title

**Endpoint:** `POST /manual_suggestions/<title_id>`

**Purpose:** Retrieve all manual action suggestions (textbooks, datasets, expert interviews, manus research) for a specific title

**Request:**
```json
{
  "user_id": "required_user_id_string"
}
```

**Response:**
```json
{
  "status": "success",
  "title_id": "title_123",
  "user_id": "user_456",
  "suggestions": [
    {
      "id": "uuid",
      "title_id": "title_123",
      "action_type": "textbook",
      "title": "Read: Real Estate Investment Analysis and Valuation",
      "description": "Comprehensive textbook covering real estate fundamentals",
      "resource_name": "Real Estate Investment Analysis and Valuation by William Poorvu",
      "priority_score": 85.5,
      "priority_level": "critical",
      "estimated_effort_hours": 20,
      "cost_estimate": "$30-80",
      "status": "suggested"
    },
    {
      "id": "uuid",
      "action_type": "dataset",
      "title": "Acquire dataset: Local MLS historical data",
      "priority_score": 75.2,
      ...
    },
    {
      "id": "uuid",
      "action_type": "manus_wide_research",
      "title": "Enable Manus-wide research for career coaching",
      "description": "Standard research yielded insufficient results. Enable Manus-wide research...",
      "research_quality_score": 25,
      "priority_level": "critical",
      ...
    }
  ],
  "total_suggestions": 8
}
```

### 2. Get All Manual Suggestions for a User

**Endpoint:** `POST /manual_suggestions`

**Purpose:** Retrieve all manual suggestions across all titles for a user

**Request:**
```json
{
  "user_id": "required_user_id_string",
  "status": "suggested",  // optional filter
  "action_type": "textbook",  // optional filter
  "priority_level": "critical",  // optional filter
  "limit": 50  // optional, default 50
}
```

**Response:**
```json
{
  "status": "success",
  "user_id": "user_456",
  "suggestions": [...],
  "total_suggestions": 42,
  "filters": {
    "status": "suggested",
    "action_type": "textbook",
    "priority_level": null,
    "limit": 50
  }
}
```

### 3. Update Suggestion Status

**Endpoint:** `PATCH /manual_suggestions/<title_id>/<suggestion_id>/status`

**Purpose:** Update the status of a manual suggestion (suggested → in_progress → completed)

**Request:**
```json
{
  "status": "completed",
  "user_id": "required_user_id_string"
}
```

**Valid Statuses:**
- `suggested` - Initial state
- `in_progress` - Currently working on
- `completed` - Finished
- `rejected` - Decided not to pursue

### 4. Delete a Suggestion

**Endpoint:** `DELETE /manual_suggestions/<suggestion_id>`

**Request:**
```json
{
  "user_id": "required_user_id_string"
}
```

### 5. Knowledge Gap Enhancement (Generates Suggestions)

**Endpoint:** `POST /gap_filler/enhance_knowledge/<title_id>`

**Purpose:** Enhances knowledge for a title and automatically generates manual action suggestions

**Request:**
```json
{
  "collection_name": "optional_collection_name",
  "merge_with_existing": true,
  "user_id": "optional_user_id"
}
```

**What It Does:**
1. Analyzes knowledge gaps
2. Researches online sources
3. Calculates research quality score
4. If quality < 30, generates Manus-wide research suggestion
5. Generates textbook, dataset, expert interview suggestions
6. Saves all suggestions to database

**Response Includes:**
```json
{
  "status": "success",
  "manual_suggestions_generated": 8,
  "research_quality_score": 25,
  "manus_suggestion_needed": true,
  ...
}
```

## 💾 Where Results Are Stored

### Database Table: `manual_action_suggestions`

**Location:** Supabase PostgreSQL database

**Table Structure:**
```sql
CREATE TABLE manual_action_suggestions (
    id UUID PRIMARY KEY,
    title_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    action_type TEXT NOT NULL CHECK (action_type IN (
        'textbook', 
        'dataset', 
        'tool_development', 
        'expert_interview', 
        'course_creation', 
        'template_creation', 
        'partnership', 
        'research_study', 
        'manus_wide_research'
    )),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    resource_name TEXT NOT NULL,
    estimated_effort_hours INTEGER NOT NULL,
    difficulty_level TEXT NOT NULL,
    expected_benefit TEXT NOT NULL,
    cost_estimate TEXT NOT NULL,
    implementation_notes TEXT NOT NULL,
    content_enhancement_potential TEXT NOT NULL,
    impact_score INTEGER NOT NULL,
    feasibility_score INTEGER NOT NULL,
    priority_score DECIMAL(5,2) NOT NULL,
    priority_level TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'suggested',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    research_context JSONB,
    additional_data JSONB,
    notes TEXT
);
```

### Key Fields for Each Type:

**Textbooks:**
- `action_type`: `"textbook"`
- `resource_name`: Book title and author
- `estimated_effort_hours`: Reading time (typically 20+ hours)
- `cost_estimate`: "$30-80" or "$50-150"

**Datasets:**
- `action_type`: `"dataset"`
- `resource_name`: Dataset name/description
- `estimated_effort_hours`: Analysis time (typically 8-12 hours)
- `cost_estimate`: "$0-500" or "$200-1000"

**Expert Interviews:**
- `action_type`: `"expert_interview"`
- `resource_name`: Expert type/name
- `estimated_effort_hours`: Interview and processing time (typically 6 hours)
- `cost_estimate`: "$0-300"
- `additional_data.interview_topics`: Suggested interview questions

**Manus-Wide Research:**
- `action_type`: `"manus_wide_research"`
- `resource_name`: "Manus-wide Research System"
- `description`: Explains why standard research is insufficient
- `research_quality_score`: The quality score that triggered this (< 30)
- `priority_level`: Always `"critical"`
- `additional_data.manus_features`: List of available Manus capabilities

### Querying Suggestions:

**By Title:**
```python
manual_action_suggestions?title_id=eq.{title_id}&user_id=eq.{user_id}&order=priority_score.desc
```

**By Action Type:**
```python
manual_action_suggestions?user_id=eq.{user_id}&action_type=eq.textbook&order=priority_score.desc
```

**By Priority:**
```python
manual_action_suggestions?user_id=eq.{user_id}&priority_level=eq.critical&order=priority_score.desc
```

## 🔄 Automatic Generation Flow

1. **Knowledge Gap Analysis:**
   - `GET /gap_filler/analyze_knowledge_gaps` identifies gaps

2. **Research Execution:**
   - `POST /gap_filler/enhance_knowledge/<title_id>` researches gaps
   - Uses `MultiSourceResearcher` to search online sources
   - Calculates `research_quality_score` in `_calculate_research_quality()`

3. **Quality Assessment:**
   - If `research_quality_score < 30`:
     - Sets `manus_suggestion_needed = True`
     - Generates Manus-wide research suggestion

4. **Suggestion Generation:**
   - `ManualActionSuggestionsGenerator.generate_action_suggestions()`:
     - Checks for Manus needs → generates `manus_wide_research` suggestion
     - Generates textbook suggestions based on topic category
     - Generates dataset suggestions based on topic and audience
     - Generates expert interview suggestions based on topic category
     - Scores and prioritizes all suggestions

5. **Storage:**
   - `save_suggestions_to_database()` saves to `manual_action_suggestions` table
   - Each suggestion includes:
     - `title_id` - Links to the title with the gap
     - `user_id` - User who owns the suggestion
     - `action_type` - Type of action needed
     - `priority_score` - Calculated priority (0-100)
     - `status` - Current state (suggested/in_progress/completed/rejected)

## 📈 Research Quality Score Calculation

**Method:** `MultiSourceResearcher._calculate_research_quality()`

**Scoring:**
- **Source Count Score (0-40):** Based on total sources found
- **Diversity Score (0-30):** Based on number of different source types
- **Quality Score (0-30):** Based on authoritative sources:
  - Academic papers: ×3 points each
  - Industry reports: ×2 points each
  - Government data: ×2 points each
  - Statistical data: ×1 point each

**Thresholds:**
- Score < 30: **Manus-wide research required** (Critical priority)
- Score < 60: **Manus-wide research recommended** (High priority)
- Score < 85: **Additional research suggested** (Medium priority)
- Score ≥ 85: **Research quality sufficient** (Low priority)

## 🎯 Action Type Details

### 1. Textbook Suggestions (`textbook`)

**Generated When:**
- Knowledge gap identified in title
- Academic or foundational knowledge needed
- Topic category matches predefined templates

**Templates Include:**
- **Real Estate:** "Real Estate Investment Analysis and Valuation", "The Millionaire Real Estate Investor"
- **Finance:** "A Random Walk Down Wall Street", "The Intelligent Investor"
- **Technology:** "Clean Code", "System Design Interview"
- **Health:** "Nutrition: Science and Applications", "ACSM's Guidelines"
- **Business:** "Good to Great", "The Lean Startup"

**Custom Suggestions:**
- Academic textbooks when `academic_papers` in `data_types_needed`

### 2. Dataset Suggestions (`dataset`)

**Generated When:**
- Statistical or data-driven content needed
- Market research required
- Industry benchmarks needed

**Common Datasets:**
- **Real Estate:** MLS historical data, property tax records, rental yield data
- **Finance:** Stock market performance, Federal Reserve indicators, credit score data
- **Technology:** GitHub statistics, Stack Overflow surveys, adoption rates
- **Health:** CDC statistics, FDA nutrition database, clinical trial results
- **Business:** SBA statistics, market research reports, consumer behavior data

### 3. Expert Interview Suggestions (`expert_interview`)

**Generated When:**
- Insider insights needed
- Authority quotes desired
- Professional perspectives required

**Expert Types by Category:**
- **Real Estate:** Local real estate agents, property managers, real estate attorneys, investors
- **Finance:** Certified Financial Planners, tax professionals, portfolio managers
- **Technology:** Senior software engineers, DevOps engineers, cybersecurity specialists
- **Health:** Registered dietitians, physical therapists, mental health counselors
- **Business:** Successful entrepreneurs, business consultants, venture capitalists

**Includes Interview Topics:**
- Current trends and challenges
- Common mistakes
- Best practices
- Future outlook
- Tool recommendations

### 4. Manus-Wide Research Suggestions (`manus_wide_research`)

**Generated When:**
- `research_quality_score < 30`
- Standard online research yields insufficient results

**Features Available:**
- AI-powered semantic search across proprietary databases
- Access to expert networks and specialized knowledge bases
- Advanced data synthesis and analysis capabilities
- Real-time market intelligence and trend analysis
- Cross-platform knowledge integration

**Additional Data:**
- `research_quality_score`: The score that triggered this (0-29)
- `manus_features`: List of available capabilities

## 🔧 Configuration

### Manus Research Configuration

**File:** `manus_research_config.py`

**Environment Variables:**
- `MANUS_QUALITY_THRESHOLD_LOW=30` - Threshold for critical suggestion
- `MANUS_QUALITY_THRESHOLD_MEDIUM=60` - Threshold for high priority
- `MANUS_QUALITY_THRESHOLD_HIGH=85` - Threshold for medium priority
- `MANUS_ENABLE_PROPRIETARY=true` - Enable proprietary databases
- `MANUS_ENABLE_EXPERT_NETWORKS=true` - Enable expert networks
- `MANUS_ENABLE_ADVANCED_ANALYSIS=true` - Enable advanced analysis

### Suggestion Templates

**File:** `manual_action_suggestions.py`

**Template Categories:**
- `real_estate`
- `finance`
- `technology`
- `health`
- `business`

Each category contains predefined:
- Textbooks
- Datasets
- Tools to develop
- Expert types to interview

## 📝 Notes

1. **User Isolation:** All suggestions are user-specific via `user_id`
2. **Automatic Prioritization:** Suggestions are automatically scored and sorted by `priority_score`
3. **Status Tracking:** Suggestions can be tracked through workflow (suggested → in_progress → completed)
4. **Research Context:** Original gap and research context stored in `research_context` JSONB field
5. **Additional Data:** Type-specific metadata stored in `additional_data` JSONB field

## 🚀 Usage Examples

### Get All Textbook Suggestions for a Title

```bash
POST /manual_suggestions/title_123
{
  "user_id": "user_456"
}
```

### Get All Manus Research Suggestions

```bash
POST /manual_suggestions
{
  "user_id": "user_456",
  "action_type": "manus_wide_research"
}
```

### Enhance Knowledge and Auto-Generate Suggestions

```bash
POST /gap_filler/enhance_knowledge/title_123
{
  "user_id": "user_456",
  "collection_name": "real_estate_collection"
}
```

This will automatically:
1. Research the knowledge gap
2. Calculate research quality score
3. Generate suggestions if quality < 30 or based on topic category
4. Save all suggestions to database
5. Return suggestion count in response

