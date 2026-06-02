# Frontend Implementation Plan: Gradio-Based Interface

**Date:** May 31, 2026  
**Document Version:** 1.0  
**Status:** Design Specification

---

## Executive Summary

This document provides a comprehensive implementation plan for a Gradio-based frontend interface that exposes all backend functionalities of the Information Retrieval System with Retrieval-Augmented Generation (RAG). The interface adopts a Google-inspired search paradigm with progressive disclosure, allowing users to seamlessly interact with local search, web augmentation, RAG generation, system configuration, and evaluation capabilities.

The design prioritizes user experience through a clean, intuitive layout that minimizes cognitive load while providing advanced configuration options for power users.

---

## 1. Design Philosophy and Core Principles

### 1.1 User-Centric Design Goals

- **Simplicity First:** Primary search interface presents only essential input controls
- **Progressive Disclosure:** Advanced features hidden in dedicated panels, accessible without cluttering the main view
- **Non-Blocking RAG:** Search results display immediately; RAG augmentation loads asynchronously
- **Visual Hierarchy:** Clear distinction between retrieval results, RAG responses, and system controls
- **Accessibility:** Keyboard navigation support, semantic HTML structure, sufficient color contrast

### 1.2 Architectural Principles

- **Modular Tab-Based Structure:** Logical separation of concerns (Search, Configuration, Evaluation, System Status)
- **Real-Time Feedback:** Progress indicators and status messages for long-running operations
- **State Persistence:** Maintain user preferences and recent queries across sessions where feasible
- **Error Resilience:** Graceful error handling with actionable error messages
- **Responsive Design:** Adapt layout for different screen sizes

---

## 2. High-Level Layout Architecture

### 2.1 Multi-Tab Interface Structure

The application employs a tabbed navigation model with four primary sections:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Information Retrieval System                  │
├─────────────────────────────────────────────────────────────────┤
│  [Search]  [Configuration]  [Evaluation]  [System Status]       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  ACTIVE TAB CONTENT                                       │ │
│  │                                                            │ │
│  │  [Tab-specific interface]                                 │ │
│  │                                                            │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Tab | Primary Function | Key Responsibilities |
|-----|------------------|---------------------|
| **Search** | Query execution and result display | Input handling, result rendering, RAG augmentation |
| **Configuration** | System parameter management | RAG settings, crawler parameters, LLM configuration |
| **Evaluation** | Test execution and result analysis | Test design, evaluation execution, metrics visualization |
| **System Status** | Database health and diagnostics | Health checks, document counts, crawler state |

---

## 3. Search Tab - Detailed Specification

### 3.1 Query Input Phase (Initial View)

**Visual Pattern:** Google Search Homepage  
**Layout:** Minimalist centered design emphasizing input focus

#### Components:

1. **Header Section**
   - Application title with branding/logo placeholder
   - Subtitle: "Information Retrieval with Augmented Generation"

2. **Search Input Control**
   - Text input field with placeholder text: "Enter your search query..."
   - Input validation: Minimum 3 characters, maximum 1000 characters
   - Auto-focus on page load
   - **Actions:**
     - Enter key or "Search" button triggers query execution
     - Clear button (X icon) to reset field
     - Keyboard shortcut support (Cmd+K / Ctrl+K for focus)

3. **Search Options Expandable**
   - Initially collapsed section below search input
   - **Options:**
     - Checkbox: "Use Web Search" (default: enabled)
     - Checkbox: "Auto-reload Database" (default: enabled)
     - Slider: "Max Local Results" (range: 1-20, default: 5)
     - Slider: "Max Web Results" (range: 0-20, default: 10)

4. **Quick Action Buttons** (Below input)
   - "Advanced Search" → Opens Configuration tab's Query Parameters section
   - "View System Status" → Switches to System Status tab
   - "Clear Database" → Confirmation dialog before deletion

#### Code Structure (Pseudocode):

```python
def search_tab():
    with gr.Group(elem_classes="search-header"):
        gr.Markdown("# Information Retrieval System")
        gr.Markdown("Search across indexed documents with RAG-augmented answers")
    
    with gr.Group(elem_classes="search-input-group"):
        query_input = gr.Textbox(
            label="Search Query",
            placeholder="Enter your search query...",
            lines=1,
            max_lines=2,
            scale=3
        )
        search_button = gr.Button("🔍 Search", scale=1, variant="primary")
        clear_button = gr.Button("✕", scale=0, elem_classes="clear-btn")
    
    with gr.Group(elem_classes="search-options", visible=False) as options_group:
        options_visible = gr.Checkbox(label="Show Options", value=False)
        enable_web_search = gr.Checkbox(label="Use Web Search", value=True)
        auto_reload = gr.Checkbox(label="Auto-reload Database", value=True)
        max_local = gr.Slider(1, 20, value=5, label="Max Local Results")
        max_web = gr.Slider(0, 20, value=10, label="Max Web Results")
    
    # Results section initially hidden
    results_section = gr.Group(visible=False, elem_classes="results-container")
    
    # Event handlers
    search_button.click(execute_search, ...)
    options_visible.change(toggle_options, ...)
```

### 3.2 Results Display Phase (Post-Query)

**Transition:** Upon successful query execution, interface transitions to results view  
**Animation:** Smooth collapse of input section, expansion of results

#### 3.2.1 Results Layout Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Search Input (Collapsed)  |  [Modify Search]                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RETRIEVAL RESULTS                                              │
│  ───────────────────────────────────────────────────────────   │
│                                                                 │
│  Metadata Bar:                                                  │
│  ├─ "X results found (Y local, Z web) in Ts seconds"           │
│  ├─ Insufficiency Detected: [Reasons list]                     │
│  └─ [Filter/Sort controls]                                     │
│                                                                 │
│  Results List:                                                  │
│  ├─ [Result Card 1]                                            │
│  ├─ [Result Card 2]                                            │
│  └─ [Result Card N]                                            │
│                                                                 │
│  RAG AUGMENTATION PANEL (Async Loading)                        │
│  ───────────────────────────────────────────────────────────   │
│                                                                 │
│  Status: [Loading spinner] or [Generated answer]               │
│                                                                 │
│  [RAG Response Section]                                         │
│  [Citation Details]                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 Search Metadata Section

**Content:**
- Query execution statistics (count, source breakdown, timing)
- Insufficiency detection summary with specific reasons
- Result filtering and sorting controls

**Components:**
```
┌─ Results Metadata ────────────────────────────────────────┐
│                                                           │
│ Found 15 results (8 local + 7 web) in 2.34 seconds       │
│                                                           │
│ ⚠ Insufficiency Detected (3 reasons):                    │
│   • Too few results (8 < 10 minimum)                     │
│   • Low average relevance score (0.42 < 0.60)            │
│   • Web search augmented with high-quality results       │
│                                                           │
│ [📊 View Detailed Metrics] [🔄 Refine Query]            │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

**Expandable Details:**
- Click on insufficiency warning reveals detailed metrics:
  - Result count per source
  - Average relevance score
  - Semantic overlap assessment
  - Timeline of operations (local search, web search, consolidation)

#### 3.2.3 Individual Result Card Design

**Layout:** Horizontal card with clear visual hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ [Source Badge]  Title (Clickable Link)                      │
│ www.example.com | Published: 2024-05-15 | Relevance: 0.87  │
│                                                             │
│ Snippet preview with query terms highlighted... Lorem       │
│ ipsum dolor sit amet consectetur. [... Read more]           │
│                                                             │
│ [Save] [Share] [View Source] [↑ More Relevant] [↓ Less]    │
└─────────────────────────────────────────────────────────────┘
```

**Components per Card:**
- **Source Badge:** Visual indicator (Local/Web, color-coded)
- **Title:** Primary heading, clickable to external link
- **Metadata Line:** URL, publication date, relevance score
- **Snippet:** Text preview (150-200 characters) with matched terms highlighted
- **Actions:** Relevance feedback, save/share, expand details
- **Expand/Collapse:** Toggle full content visibility

**Relevance Score Visualization:**
- Circular progress indicator (0-100%) with color gradient
- Green (0.7-1.0), Yellow (0.4-0.7), Red (0.0-0.4)

#### 3.2.4 Result Filtering and Sorting

**Controls:**
- **Sort Options:** Relevance (default), Date, Source
- **Filter:** By source (Local/Web), By relevance threshold
- **Search Within Results:** Additional text search box for refinement

#### 3.2.5 RAG Augmentation Panel

**Design Principle:** Non-blocking, asynchronously loaded

**Behavior:**
1. Immediately after query execution, results display without waiting for RAG
2. RAG generation starts in parallel, showing loading state
3. Upon completion, augmented answer replaces loading indicator
4. If RAG fails, graceful error message with option to retry

**Layout:**

```
┌─ RAG-Augmented Answer ──────────────────────────────────────┐
│                                                              │
│ Status: [⏳ Generating answer (3.2s so far)...]             │
│                                                              │
│ [When ready:]                                               │
│                                                              │
│ Generated Answer:                                            │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ The comprehensive response generated by the LLM,      │   │
│ │ synthesizing information from the retrieved documents │   │
│ │ in a coherent, natural manner...                     │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                              │
│ 📚 Citations (N documents referenced):                      │
│ ├─ [1] "Title from Document A" (www.source-a.com) - "key   │
│ │      quote illustrating relevance"                        │
│ ├─ [2] "Title from Document B" (www.source-b.com) - "quote" │
│ └─ [N] "Title from Document N" ...                          │
│                                                              │
│ [🔄 Regenerate] [📋 Copy Answer] [💾 Save Response]        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**RAG Response Components:**

1. **Answer Section**
   - Rendered as formatted markdown with proper text wrapping
   - Syntax highlighting for code blocks (if present)
   - LaTeX rendering for mathematical expressions (if applicable)

2. **Citations Section**
   - Numbered list with clickable links
   - Each citation includes:
     - Document title (linked)
     - Source URL
     - Relevant quote (snippet) with proper character truncation
     - Confidence/relevance indicator
   - Collapsible full document view on citation click

3. **Metadata Footer**
   - Generation time in seconds
   - Model used (e.g., "Ollama: llama3.2:latest")
   - Token count (approximate)

4. **Action Buttons**
   - Regenerate: Re-run RAG with same documents
   - Copy Answer: Copy to clipboard
   - Export: Download as PDF/Markdown
   - Save Response: Store in local response history

**Error Handling:**
- If RAG generation fails: Display error message with retry option
- If LLM unreachable: Suggest checking Ollama service status
- If timeout: Allow user to proceed with retrieval results only

### 3.3 Query Refinement and History

**Components:**

1. **Recent Queries**
   - Sidebar or collapsible panel showing last 5-10 queries
   - Click to re-execute with identical parameters
   - Swipe-to-delete gesture support
   - Star/favorite marking for frequently used queries

2. **Query Suggestions**
   - Auto-complete suggestions based on indexed document titles
   - Related query suggestions from evaluation test set
   - Spelling correction feedback

3. **Modify Search Controls** (Within results view)
   - "Refine Query" button opens edit mode
   - Adjust search parameters without full page reload
   - "Start New Search" returns to clean input state

---

## 4. Configuration Tab - Detailed Specification

### 4.1 Tab Organization

Configuration divided into logical subsections with collapsible panels:

```
┌─ Configuration ────────────────────────────────────────────┐
│                                                            │
│ ┌─ Query Parameters ────────────────────────────────────┐ │
│ │ [Settings for search behavior]                        │ │
│ └──────────────────────────────────────────────────────┘ │
│                                                            │
│ ┌─ RAG Configuration ──────────────────────────────────┐ │
│ │ [LLM and generation settings]                        │ │
│ └──────────────────────────────────────────────────────┘ │
│                                                            │
│ ┌─ Crawler Configuration ──────────────────────────────┐ │
│ │ [Data acquisition settings]                          │ │
│ └──────────────────────────────────────────────────────┘ │
│                                                            │
│ ┌─ Database Management ────────────────────────────────┐ │
│ │ [Clear and load operations]                          │ │
│ └──────────────────────────────────────────────────────┘ │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Query Parameters Section

**Editable Parameters:**
- Max local results (1-20)
- Max web results (0-20)
- Enable web search (toggle)
- Auto-reload empty database (toggle)
- Min articles requirement (number input)

**UI Implementation:**
```python
with gr.Group(label="Query Parameters", elem_classes="config-section"):
    max_local = gr.Slider(1, 20, value=5, label="Max Local Results")
    max_web = gr.Slider(0, 20, value=10, label="Max Web Results")
    use_web = gr.Checkbox(value=True, label="Use Web Search")
    auto_reload = gr.Checkbox(value=True, label="Auto-reload Database")
    min_docs = gr.Number(value=500, label="Minimum Database Documents")
    
    save_query_btn = gr.Button("Save Query Settings")
```

### 4.3 RAG Configuration Section

**Editable Parameters** (from `rag/config.py`):
- Ollama model selection (dropdown)
- Ollama base URL (text input)
- RAG template selection (radio buttons: basic, domain_specific, chain_of_thought)
- Retrieval top-k (1-20)
- Temperature (0.0-1.0 slider)
- Max tokens (100-4096)
- Citation threshold (0.0-1.0)
- Max citations (1-20)
- Response character limit (100-5000)

**UI Implementation:**
```python
with gr.Group(label="RAG Configuration", elem_classes="config-section"):
    ollama_model = gr.Dropdown(
        choices=["llama3.2:latest", "llama2", "mistral", "neural-chat"],
        value="llama3.2:latest",
        label="LLM Model",
        allow_custom_value=True
    )
    ollama_url = gr.Textbox(value="http://localhost:11434", label="Ollama Base URL")
    rag_template = gr.Radio(
        choices=["basic", "domain_specific", "chain_of_thought"],
        value="domain_specific",
        label="RAG Template"
    )
    temperature = gr.Slider(0.0, 1.0, value=0.7, label="Temperature")
    max_tokens = gr.Slider(100, 4096, value=1024, step=100, label="Max Tokens")
    max_citations = gr.Slider(1, 20, value=10, label="Max Citations")
    
    with gr.Row():
        save_rag_btn = gr.Button("Save RAG Settings")
        test_connection_btn = gr.Button("Test LLM Connection")
    
    connection_status = gr.Textbox(label="Connection Status", interactive=False)
```

**Test Connection Feature:**
- Attempts to ping Ollama service
- Returns latency and model availability
- Provides troubleshooting suggestions if unreachable

### 4.4 Crawler Configuration Section

**Editable Parameters:**
- Max articles per spider (100-10000)
- Force recrawl toggle
- Crawler timeout (30-300 seconds)
- Specific spider selection (checkboxes for enabling/disabling individual crawlers)

**UI Implementation:**
```python
with gr.Group(label="Crawler Configuration", elem_classes="config-section"):
    max_articles = gr.Slider(100, 10000, value=1000, step=100, label="Max Articles per Spider")
    force_recrawl = gr.Checkbox(value=False, label="Force Recrawl (ignore cache)")
    crawler_timeout = gr.Slider(30, 300, value=120, step=30, label="Crawler Timeout (seconds)")
    
    gr.Markdown("### Enabled Crawlers")
    with gr.Row():
        devto_enabled = gr.Checkbox(value=True, label="DevTo")
        hackernews_enabled = gr.Checkbox(value=True, label="HackerNews")
        realpython_enabled = gr.Checkbox(value=True, label="RealPython")
    with gr.Row():
        lobsters_enabled = gr.Checkbox(value=True, label="Lobsters")
        newstack_enabled = gr.Checkbox(value=True, label="TheNewStack")
        theverge_enabled = gr.Checkbox(value=True, label="TheVerge")
    
    save_crawler_btn = gr.Button("Save Crawler Settings")
```

### 4.5 Database Management Section

**Operations:**

1. **Clear Database**
   - Destructive operation requiring confirmation
   - Removes: VectorStore, ChromaDB collection, cached indices, documents.json
   - Progress indicator during deletion
   - Confirmation dialog: "This action cannot be undone"

2. **Load/Reload Database**
   - Execute full crawler pipeline
   - Progress bar showing:
     - Crawler execution status
     - Document count being indexed
     - ETA based on historical data
   - Cancel button to abort mid-operation

3. **Status Indicators**
   - Current document count (indexed vs. file vs. raw)
   - Last update timestamp
   - Database health status (healthy/degraded/empty)

**UI Implementation:**
```python
with gr.Group(label="Database Management", elem_classes="config-section"):
    gr.Markdown("### Quick Actions")
    
    with gr.Row():
        clear_db_btn = gr.Button("🗑️ Clear Database", variant="stop")
        reload_db_btn = gr.Button("🔄 Load/Reload Database", variant="primary")
        health_check_btn = gr.Button("📋 Check Health")
    
    with gr.Row():
        db_status = gr.Textbox(label="Database Status", interactive=False)
        doc_count = gr.Number(label="Indexed Documents", interactive=False)
    
    operation_progress = gr.Progress(visible=False)
    operation_log = gr.Textbox(
        label="Operation Log",
        lines=5,
        max_lines=10,
        interactive=False,
        visible=False
    )
    
    # Event handlers
    clear_db_btn.click(confirm_clear_database, ...)
    reload_db_btn.click(load_database_with_progress, ...)
```

**Confirmation Dialogs:**
- Use Gradio's modal-like behavior or custom HTML overlay
- Show potential consequences of operation
- Require explicit acknowledgment (e.g., type "CONFIRM")

---

## 5. Evaluation Tab - Detailed Specification

### 5.1 Evaluation Interface Layout

```
┌─ Evaluation ──────────────────────────────────────────────────┐
│                                                               │
│ ┌─ Test Configuration ──────────────────────────────────────┐│
│ │ [Design custom test or run defaults]                      ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Test Designer ───────────────────────────────────────────┐│
│ │ [Interface for creating custom test cases]                ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
│ ┌─ Evaluation Results ──────────────────────────────────────┐│
│ │ [Metrics visualization and detailed breakdown]            ││
│ └───────────────────────────────────────────────────────────┘│
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 5.2 Test Configuration Section

**Options:**

1. **Run Default Test**
   - Loads test_queries.json from data/ directory
   - Button to execute evaluation on stored test set
   - Displays message if no stored tests found

2. **Design Custom Test**
   - Toggle to switch to custom test design mode
   - Hides default test controls
   - Shows test designer interface

**UI Implementation:**
```python
with gr.Group(label="Test Configuration"):
    test_mode = gr.Radio(
        choices=["Run Default Test", "Design Custom Test"],
        value="Run Default Test",
        label="Test Mode"
    )
    
    with gr.Group(visible=True) as default_test_group:
        gr.Markdown("### Run Default Test")
        gr.Markdown("Execute evaluation on stored test queries from `data/test_queries.json`")
        
        run_default_btn = gr.Button("▶️ Run Evaluation", variant="primary")
        default_test_status = gr.Textbox(
            label="Status",
            value="Ready",
            interactive=False
        )
    
    with gr.Group(visible=False) as custom_test_group:
        gr.Markdown("### Design Custom Test")
        custom_test_interface = create_test_designer()
```

### 5.3 Test Designer Interface

**Multi-Row Form for Adding Test Cases:**

```
┌─ Add Test Query ───────────────────────────────────────────┐
│                                                            │
│ Query ID:           [____________________]                │
│ Query Text:         [____________________]                │
│ Relevance Scale:    [○ Binary ○ Graded (0-3)]            │
│                                                            │
│ Relevant Documents: [Comma-separated doc IDs]            │
│                                                            │
│ [If Graded Selected:]                                     │
│ Doc Grade Mappings: [____________________]                │
│ (Format: doc_id:grade, doc_id:grade, ...)               │
│                                                            │
│ [➕ Add Query] [✕ Clear]                                 │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Test Query Table:**
- Sortable/filterable table showing added queries
- Columns: Query ID, Query Text (truncated), # Relevant Docs, Grade Scale
- Row actions: Edit, Delete, Preview
- Bulk operations: Delete all, Export JSON

**UI Implementation:**
```python
with gr.Group(elem_classes="test-designer"):
    gr.Markdown("### Add Test Queries")
    
    with gr.Row():
        query_id = gr.Textbox(label="Query ID", scale=1)
        query_text = gr.Textbox(label="Query Text", scale=4)
    
    with gr.Row():
        grade_scale = gr.Radio(
            choices=["Binary (0/1)", "Graded (0-3)"],
            value="Binary (0/1)",
            scale=2,
            label="Relevance Scale"
        )
        relevant_docs = gr.Textbox(
            label="Relevant Doc IDs (comma-separated)",
            scale=3
        )
    
    grade_mappings = gr.Textbox(
        label="Grade Mappings (if graded)",
        placeholder="doc_id:grade,doc_id:grade",
        visible=False
    )
    
    with gr.Row():
        add_query_btn = gr.Button("➕ Add Query", variant="secondary")
        clear_form_btn = gr.Button("Clear Form")
    
    test_queries_table = gr.Dataframe(
        headers=["Query ID", "Query Text", "# Relevant", "Scale", "Actions"],
        interactive=True,
        label="Test Queries"
    )
    
    with gr.Row():
        export_test_btn = gr.Button("📥 Export as JSON")
        delete_all_btn = gr.Button("🗑️ Delete All", variant="stop")
    
    # Event handlers
    grade_scale.change(
        lambda x: gr.update(visible="Graded" in x),
        inputs=grade_scale,
        outputs=grade_mappings
    )
```

### 5.4 Evaluation Results Section

**Aggregate Metrics Display:**

```
┌─ Evaluation Summary ───────────────────────────────────────┐
│                                                            │
│ Test Set: 10 queries | Execution Time: 45.3 seconds       │
│                                                            │
│ ┌─ Overall Metrics ──────────────────────────────────────┐│
│ │ ┌──────────┬──────────┬──────────┬──────────┐          ││
│ │ │   MAP    │   MRR    │   P@1    │  NDCG@5  │          ││
│ │ │  0.652   │  0.847   │  0.700   │  0.715   │          ││
│ │ └──────────┴──────────┴──────────┴──────────┘          ││
│ │                                                         ││
│ │ ┌──────────┬──────────┬──────────┬──────────┐          ││
│ │ │   P@5    │   R@5    │  Mean AP │  Mean RR │          ││
│ │ │  0.560   │  0.432   │  0.652   │  0.847   │          ││
│ │ └──────────┴──────────┴──────────┴──────────┘          ││
│ │                                                         ││
│ └─────────────────────────────────────────────────────────┘│
│                                                            │
│ [📊 View Charts] [📋 Export Results] [🔄 Run Again]      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Components:**

1. **Aggregate Metrics Panel**
   - Display key metrics in card layout:
     - MAP (Mean Average Precision)
     - MRR (Mean Reciprocal Rank)
     - Precision @ K (1, 3, 5, 10)
     - Recall @ K (1, 3, 5, 10)
     - NDCG @ K (1, 3, 5, 10)
     - Mean AP and RR
   - Color-coded performance indicators (green/yellow/red)

2. **Visualization Charts**
   - Line chart: Metrics across different K values
   - Bar chart: Comparison of metric values
   - Distribution plot: Per-query performance spread

3. **Per-Query Breakdown Table**
   ```
   ┌─────┬──────────────┬───────┬────┬────┬──────┬──────┐
   │ ID  │ Query        │ #Rel  │AP  │RR  │P@5  │NDCG  │
   ├─────┼──────────────┼───────┼────┼────┼──────┼──────┤
   │ q1  │ machine...   │ 5     │.85 │1.0 │.80  │.895  │
   │ q2  │ deep...      │ 3     │.62 │.67 │.40  │.578  │
   │ ... │ ...          │ ...   │... │... │...  │ ...  │
   └─────┴──────────────┴───────┴────┴────┴──────┴──────┘
   ```
   - Click row to expand and view retrieved results ranking
   - Export individual query details

4. **Export Options**
   - Download results as JSON
   - Export as CSV
   - Generate PDF report with visualizations

**UI Implementation:**
```python
def create_evaluation_results_section():
    with gr.Group(label="Evaluation Results", elem_classes="eval-results"):
        
        # Summary header
        with gr.Row():
            eval_summary = gr.Textbox(
                label="Evaluation Summary",
                interactive=False,
                max_lines=2
            )
            eval_time = gr.Textbox(
                label="Execution Time (s)",
                interactive=False,
                scale=1
            )
        
        # Aggregate metrics
        with gr.Group(label="Overall Metrics"):
            metrics_display = create_metrics_cards()
        
        # Charts
        with gr.Row():
            metrics_chart = gr.Plot(label="Metrics by K Value")
            performance_chart = gr.BarPlot(label="Performance Distribution")
        
        # Per-query results
        with gr.Group(label="Per-Query Results"):
            results_table = gr.Dataframe(
                headers=["Query ID", "Query", "#Relevant", "AP", "RR", "P@5", "R@5", "NDCG@5"],
                interactive=False,
                label="Query Results"
            )
        
        # Export buttons
        with gr.Row():
            export_json_btn = gr.Button("📥 Export JSON")
            export_csv_btn = gr.Button("📥 Export CSV")
            export_pdf_btn = gr.Button("📥 Export PDF Report")
        
        return metrics_display, metrics_chart, performance_chart, results_table

def create_metrics_cards():
    """Create visual metric cards for key metrics"""
    with gr.Row():
        map_card = gr.Number(label="MAP", interactive=False)
        mrr_card = gr.Number(label="MRR", interactive=False)
        p_at_1_card = gr.Number(label="P@1", interactive=False)
        ndcg_at_5_card = gr.Number(label="NDCG@5", interactive=False)
    
    with gr.Row():
        p_at_5_card = gr.Number(label="P@5", interactive=False)
        r_at_5_card = gr.Number(label="R@5", interactive=False)
        mean_ap_card = gr.Number(label="Mean AP", interactive=False)
        mean_rr_card = gr.Number(label="Mean RR", interactive=False)
    
    return [map_card, mrr_card, p_at_1_card, ndcg_at_5_card,
            p_at_5_card, r_at_5_card, mean_ap_card, mean_rr_card]
```

### 5.5 Evaluation Execution Flow

1. User selects test mode (default or custom)
2. For custom tests: Design test cases using designer interface
3. Click "Run Evaluation" button
4. Progress indicator shows:
   - Current query being evaluated (N/Total)
   - Time elapsed
   - ETA to completion
5. Upon completion:
   - Results section populates with metrics
   - Charts render automatically
   - Per-query table displays
   - Export options become available
6. Option to refine test and re-run

---

## 6. System Status Tab - Detailed Specification

### 6.1 Overview Dashboard

```
┌─ System Status ────────────────────────────────────────────┐
│                                                            │
│ ┌─ Health Summary ──────────────────────────────────────┐│
│ │ Status: [🟢 Healthy / 🟡 Degraded / 🔴 Critical]   ││
│ │ Last Updated: [timestamp]                             ││
│ └───────────────────────────────────────────────────────┘│
│                                                            │
│ ┌─ Database Status ─────────────────────────────────────┐│
│ │ Indexed Documents:     [N] (Min required: 500)        ││
│ │ File Documents:        [N] (documents.json)           ││
│ │ Raw Documents:         [N] (data/raw/)                ││
│ │ ChromaDB Available:    [Yes/No]                        ││
│ └───────────────────────────────────────────────────────┘│
│                                                            │
│ ┌─ Crawler State ───────────────────────────────────────┐│
│ │ DevTo:        [Last update: X hours ago] [N docs]    ││
│ │ HackerNews:   [Last update: X hours ago] [N docs]    ││
│ │ RealPython:   [Last update: X hours ago] [N docs]    ││
│ │ Lobsters:     [Last update: X hours ago] [N docs]    ││
│ │ TheNewStack:  [Last update: X hours ago] [N docs]    ││
│ │ TheVerge:     [Last update: X hours ago] [N docs]    ││
│ └───────────────────────────────────────────────────────┘│
│                                                            │
│ ┌─ LLM Service ─────────────────────────────────────────┐│
│ │ Status:      [🟢 Connected / 🔴 Unreachable]         ││
│ │ Model:       [llama3.2:latest]                        ││
│ │ Latency:     [X ms]                                   ││
│ │ Last Check:  [timestamp]                              ││
│ └───────────────────────────────────────────────────────┘│
│                                                            │
│ [🔄 Refresh All] [📥 Export Diagnostics]                │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 6.2 Health Summary Section

**Content:**
- Overall system status indicator with status code
- Last health check timestamp
- Quick action buttons for common diagnostics
- Color-coded status badge:
  - Green (🟢): All systems operational, sufficient documents
  - Yellow (🟡): Degraded - below minimum documents or partial failures
  - Red (🔴): Critical - database empty, LLM unavailable, or major failures

### 6.3 Database Status Section

**Displays:**
- Count of indexed documents (from VectorStore.count())
- Count of documents in documents.json file
- Count of raw documents in data/raw/
- ChromaDB availability status
- Storage usage (optional, if feasible)
- Last index timestamp

**Visual Elements:**
- Progress bars showing document counts relative to minimum threshold
- Status indicator for each count source
- Quick links to configuration for parameter adjustment

### 6.4 Crawler State Section

**Per-Spider Information:**
- Spider name
- Last successful crawl timestamp
- Document count from that spider
- Status badge (Success/Pending/Error)
- Manual refresh button per spider
- Historical document trends (optional)

**UI Implementation:**
```python
with gr.Group(label="Crawler State"):
    crawlers = ["DevTo", "HackerNews", "RealPython", "Lobsters", "TheNewStack", "TheVerge"]
    
    crawler_rows = []
    for crawler_name in crawlers:
        with gr.Row():
            crawler_label = gr.Textbox(
                value=f"{crawler_name}",
                interactive=False,
                scale=1
            )
            last_update = gr.Textbox(
                label="Last Update",
                interactive=False,
                scale=2
            )
            doc_count = gr.Number(
                label="Documents",
                interactive=False,
                scale=1
            )
            status = gr.Textbox(
                label="Status",
                interactive=False,
                scale=1
            )
            refresh_btn = gr.Button("🔄", scale=0)
        
        crawler_rows.append({
            'name': crawler_name,
            'last_update': last_update,
            'doc_count': doc_count,
            'status': status,
            'refresh_btn': refresh_btn
        })
```

### 6.5 LLM Service Status Section

**Displays:**
- Connection status (Connected/Unreachable)
- Active model name
- Latency (in milliseconds)
- Last connection check timestamp
- Error messages if applicable

**LLM Health Check:**
- Attempts simple inference to verify service operability
- Shows response time
- Provides troubleshooting suggestions if connection fails

### 6.6 Refresh and Export Functions

**Refresh All Button:**
- Triggers health check across all system components
- Updates all status displays
- Shows progress indicator during refresh

**Export Diagnostics:**
- Generates comprehensive diagnostics file (JSON/PDF)
- Includes:
  - System version information
  - Timestamp
  - All status metrics
  - Recent error logs
  - Configuration snapshot (sanitized)

**Auto-Refresh (Optional):**
- Periodic background refresh (e.g., every 30 seconds)
- User-configurable refresh interval
- Toast notifications for status changes

---

## 7. Cross-Cutting Concerns

### 7.1 Responsive Design and Layout

**Grid System:**
- Implement responsive columns that adapt to viewport width
- Desktop (>1200px): Full multi-column layouts
- Tablet (768-1200px): Reduced columns, optimized spacing
- Mobile (<768px): Single-column stacked layout

**Theme Support:**
- Light theme (default)
- Dark theme with proper contrast ratios
- System preference detection (prefers-color-scheme)

### 7.2 Error Handling and User Feedback

**Error Display Strategy:**
- Non-blocking error notifications (toast/banner style)
- Contextual error messages with actionable suggestions
- Error logging for debugging (console/file)

**Toast Notifications:**
- Success: Green badge with checkmark
- Error: Red badge with X icon
- Warning: Yellow badge with exclamation mark
- Info: Blue badge with info icon
- Auto-dismiss after 5 seconds (or manual close)

**Loading States:**
- Spinner animations during async operations
- Progress bars for long operations (>2 seconds)
- Skeleton screens as placeholders for content

### 7.3 State Management

**Session State Persistence:**
- Store user configuration preferences in browser localStorage
- Maintain search history
- Cache RAG responses for identical queries
- Preserve tab state between page refreshes

**Backend State:**
- Query execution logging
- Performance metrics collection
- Error tracking and reporting

### 7.4 Keyboard Navigation and Accessibility

**Keyboard Shortcuts:**
- Cmd+K / Ctrl+K: Focus search input
- Cmd+Enter / Ctrl+Enter: Execute search
- Escape: Clear input or close dialogs
- Tab: Navigate between form fields
- Enter: Submit forms

**Accessibility Features:**
- ARIA labels for interactive elements
- Semantic HTML structure
- Color-blind safe palette
- Text alternatives for icons
- Focus indicators for keyboard navigation

### 7.5 Performance Optimization

**Frontend Optimization:**
- Lazy loading of tab content
- Virtual scrolling for large result lists
- Debounced search input (300ms)
- Memoization of expensive computations
- CSS-in-JS or compiled CSS for minimal bundle size

**Backend Communication:**
- Asynchronous queries using Gradio's event system
- Streaming responses (if supported by backend)
- Request cancellation for user-initiated aborts
- Caching of frequently accessed data

### 7.6 Data Validation and Input Sanitization

**Client-Side Validation:**
- Query text: Non-empty, maximum 1000 characters
- Numeric inputs: Type checking and range validation
- Configuration parameters: Whitelist of allowed values
- Test queries: Required fields, format validation

**Error Messages:**
- Clear, non-technical language
- Specific guidance on correcting input
- Example formats when applicable

---

## 8. Backend Integration Points

### 8.1 API Mapping to MainOrchestrator

| Frontend Feature | Backend Method | Parameters | Returns |
|------------------|-----------------|-----------|---------|
| Search query execution | `query()` | query, max_local_results, enable_web_search, auto_reload | RAGResponse |
| Retrieve documents | `retrieve_documents()` | question, max_local_results, enable_web_search, auto_reload | Dict with documents & metadata |
| Clear database | `clear_all_indices()` | - | Dict with success status |
| Load/reload database | `load_documents_from_crawlers()` | max_articles, force_recrawl | Dict with load statistics |
| Database health | `check_database_health()` | - | Dict with health metrics |
| System status | `get_status()` | - | Dict with complete status |
| Run evaluation | `evaluate_test()` | test_spec (optional) | Dict with evaluation results |
| Augment response (RAG) | `augment_response()` | question, documents | RAGResponse |
| LLM connection test | Custom helper | model, url | Dict with latency & status |

### 8.2 Asynchronous Processing Pattern

**Non-Blocking RAG Flow:**
```
[Query Execution]
      ↓
[Immediate: Retrieve Documents + Display Results]
      ↓
[Parallel: Generate RAG Response]
      ↓
[Update: Display RAG Answer When Ready]
```

**Implementation Approach:**
```python
def execute_search(query, max_local, use_web, auto_reload):
    # Step 1: Retrieve documents
    retrieval_result = orchestrator.retrieve_documents(
        question=query,
        max_local_results=max_local,
        enable_web_search=use_web,
        auto_reload=auto_reload
    )
    
    documents = retrieval_result['documents']
    metadata = retrieval_result['metadata']
    
    # Step 2: Return retrieval results immediately
    yield render_results(documents, metadata)
    
    # Step 3: Asynchronously generate RAG response
    rag_response = orchestrator.augment_response(query, documents)
    
    # Step 4: Update with RAG response
    yield render_results_with_rag(documents, metadata, rag_response)
```

### 8.3 Error Recovery and Fallbacks

**Query Execution:**
- If RAG generation fails: Display results without augmentation
- If web search fails: Continue with local results
- If local search returns no results: Trigger database reload

**Database Operations:**
- If clear fails: Provide detailed error and retry option
- If load fails: Suggest checking crawler logs
- Graceful degradation when partial operations succeed

---

## 9. Technology Stack

### 9.1 Frontend Framework

- **Primary:** Gradio (Python-based UI framework for ML/AI applications)
- **Styling:** Custom CSS with CSS variables for theming
- **Charting:** Plotly/Matplotlib for evaluation metrics visualization
- **Icons:** Unicode or Font Awesome for visual elements

### 9.2 Dependencies

```
gradio>=4.0.0
plotly>=5.14.0  # For evaluation result visualizations
pandas>=1.5.0   # For data table handling and export
pydantic>=2.0.0  # For data validation
```

### 9.3 Application Structure

```
src/
├── frontend/
│   ├── app.py              # Main Gradio app initialization
│   ├── tabs/
│   │   ├── search.py       # Search tab components
│   │   ├── configuration.py # Configuration tab
│   │   ├── evaluation.py   # Evaluation tab
│   │   └── status.py       # System Status tab
│   ├── components/
│   │   ├── result_cards.py    # Reusable result display
│   │   ├── charts.py          # Evaluation visualizations
│   │   ├── metrics.py         # Metric display components
│   │   └── dialogs.py         # Modal dialogs
│   ├── utils/
│   │   ├── formatting.py      # Text formatting helpers
│   │   ├── state.py           # Session state management
│   │   └── validators.py      # Input validation
│   └── styles/
│       ├── theme.css          # Main stylesheet
│       └── variables.css      # CSS variables for theming
```

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
- Set up Gradio application structure
- Implement core tab navigation
- Create Search tab with basic query input
- Implement MainOrchestrator integration stub

### Phase 2: Search Functionality (Week 2-3)
- Implement retrieval results display
- Add result card rendering
- Integrate asynchronous RAG response generation
- Add result filtering and sorting

### Phase 3: Configuration & Management (Week 3-4)
- Implement Configuration tab
- Add database management operations (clear, load/reload)
- Create RAG settings editor with LLM connection testing
- Add crawler configuration panel

### Phase 4: Evaluation System (Week 4-5)
- Implement test designer interface
- Create evaluation execution flow
- Build results visualization with charts
- Add export functionality (JSON, CSV, PDF)

### Phase 5: System Status & Monitoring (Week 5)
- Implement System Status tab
- Add health check and diagnostics
- Create crawler state monitoring
- Add LLM service status display

### Phase 6: Polish & Optimization (Week 6)
- Responsive design refinement
- Performance optimization
- Error handling improvements
- User documentation and help tooltips
- Theme support (light/dark)

---

## 11. Deployment Considerations

### 11.1 Container Integration

The frontend will be deployed as part of the Docker Compose stack:

```yaml
services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "7860:7860"  # Gradio default port
    environment:
      - ORCHESTRATOR_HOST=http://backend:8000
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama
      - backend
```

### 11.2 Configuration Management

- Environment variables for Ollama URL, API endpoints
- ConfigMap support for Kubernetes deployments
- Secret management for sensitive parameters

---

## 12. Testing and Quality Assurance

### 12.1 Frontend Testing Strategy

**Unit Tests:**
- Component rendering with various data inputs
- Input validation and error handling
- State management and persistence

**Integration Tests:**
- Tab navigation and state transitions
- Backend API integration
- Async operation handling

**E2E Tests:**
- Complete user workflows (search, configuration, evaluation)
- Cross-browser compatibility
- Responsive design across device sizes

### 12.2 Performance Benchmarks

- Search result rendering: <1 second for 20 results
- Tab switching: <500ms transition time
- Configuration save: <2 seconds
- Evaluation execution: Display results within 5 seconds of completion

---

## 13. Future Enhancements

- **Advanced Query Syntax:** Support for boolean operators, filters
- **Saved Searches & Alerts:** Store and monitor specific queries
- **Collaborative Features:** Share evaluations and results with team members
- **Custom Metrics:** User-defined evaluation metrics beyond standard IR metrics
- **Multi-Language Support:** Internationalization for UI text
- **Browser Extension:** Quick search from external websites
- **Mobile App:** Native mobile application with offline support
- **Analytics Dashboard:** Historical performance tracking and trend analysis

---

## 14. Documentation and User Guides

### 14.1 In-App Help

- Contextual tooltips for all major controls
- Help icons with expandable documentation
- Quick start guide on initial load
- Video tutorials for complex operations

### 14.2 External Documentation

- User manual with screenshots
- Configuration reference guide
- Troubleshooting FAQ
- API documentation for backend integration

---

## Appendix A: Design Tokens and Styling Guidelines

### Color Palette

| Purpose | Light Theme | Dark Theme |
|---------|-------------|-----------|
| Primary | #1F2937 (Slate-900) | #F3F4F6 (Slate-100) |
| Success | #10B981 (Emerald-500) | #10B981 |
| Error | #EF4444 (Red-500) | #EF4444 |
| Warning | #F59E0B (Amber-500) | #F59E0B |
| Info | #3B82F6 (Blue-500) | #3B82F6 |
| Background | #FFFFFF | #111827 (Slate-900) |
| Surface | #F9FAFB (Slate-50) | #1F2937 (Slate-800) |
| Border | #E5E7EB (Slate-200) | #374151 (Slate-700) |

### Typography

- **Heading 1:** 2.5rem, bold, line-height 1.2
- **Heading 2:** 2rem, bold, line-height 1.3
- **Heading 3:** 1.5rem, semibold, line-height 1.4
- **Body:** 1rem, normal, line-height 1.6
- **Small:** 0.875rem, normal, line-height 1.5
- **Code:** Monospace, 0.875rem

### Spacing

- **XS:** 0.25rem (4px)
- **SM:** 0.5rem (8px)
- **MD:** 1rem (16px)
- **LG:** 1.5rem (24px)
- **XL:** 2rem (32px)
- **2XL:** 3rem (48px)

---

## Appendix B: Wireframes and Visual References

### B.1 Search Tab - Query Phase
```
┌────────────────────────────────────────────────────────┐
│           Information Retrieval System                 │
├────────────────────────────────────────────────────────┤
│  [Search] [Config] [Evaluation] [Status]              │
├────────────────────────────────────────────────────────┤
│                                                        │
│         ┌───────────────────────────────────┐        │
│         │ Search Query                      │  [🔍]  │
│         │ Enter your search query...        │        │
│         └───────────────────────────────────┘        │
│                                                        │
│         [⚙️ Show Options] [?] [Status] [Clear DB]    │
│                                                        │
│         ┌─ Options (Hidden) ─────────────────────┐   │
│         │ ☑ Use Web Search                       │   │
│         │ ☑ Auto-reload Database                 │   │
│         │ Max Local Results: [————•————] 5        │   │
│         │ Max Web Results:   [————•————] 10       │   │
│         └────────────────────────────────────────┘   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### B.2 Search Tab - Results Phase
```
┌────────────────────────────────────────────────────────┐
│ Search: [machine learning] [🔍]  [⌄ Modify Search]   │
├────────────────────────────────────────────────────────┤
│                                                        │
│ Found 15 results (8 local + 7 web) in 2.34s           │
│ ⚠ Insufficiency Detected: Low avg score              │
│ [View Details] [Filter] [Sort: ⌄Relevance]           │
│                                                        │
│ ┌──────────────────────────────────────────────────┐ │
│ │[Local] Title of First Result                     │ │
│ │example.com | 2024-05-15 | Relevance: 0.87 ⭐   │ │
│ │Snippet: Lorem ipsum dolor sit amet consectetur  │ │
│ │[Save] [Share] [More] [Feedback]                  │ │
│ └──────────────────────────────────────────────────┘ │
│                                                        │
│ ┌──────────────────────────────────────────────────┐ │
│ │[Web] Title of Second Result                      │ │
│ │www.source.com | 2024-04-20 | Relevance: 0.72 ⭐│ │
│ │Snippet: Consectetur adipiscing elit sed do...   │ │
│ │[Save] [Share] [More] [Feedback]                  │ │
│ └──────────────────────────────────────────────────┘ │
│                                                        │
│ ┌─ RAG-Augmented Answer ────────────────────────────┐ │
│ │ ⏳ Generating answer (3.2s so far)...              │ │
│ └────────────────────────────────────────────────────┘ │
│                                                        │
│ [More Results]  [Refine Search]  [New Search]        │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## Appendix C: Configuration Parameter Reference

### Query Parameters
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| max_local_results | 1-20 | 5 | Maximum local search results to use |
| max_web_results | 0-20 | 10 | Maximum web search results |
| enable_web_search | boolean | true | Enable web augmentation |
| auto_reload | boolean | true | Auto-load data if DB empty |

### RAG Configuration
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| ollama_model | string | llama3.2:latest | Model identifier |
| ollama_base_url | URL | http://localhost:11434 | Ollama service URL |
| rag_template | {basic, domain_specific, chain_of_thought} | domain_specific | Prompt template |
| temperature | 0.0-1.0 | 0.7 | Response randomness |
| max_tokens | 100-4096 | 1024 | Maximum response length |
| max_citations | 1-20 | 10 | Maximum citations in response |

### Crawler Configuration
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| max_articles | 100-10000 | 1000 | Articles per spider |
| force_recrawl | boolean | false | Ignore cache and re-crawl |
| crawler_timeout | 30-300 | 120 | Timeout per crawler (seconds) |

---

**Document Status:** Ready for Review  
**Approval Required By:** Project Lead  
**Next Steps:** Implementation begins upon approval
