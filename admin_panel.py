"""
RCE-LLM Admin Panel - Streamlit Version
Adapted for the new RCE-LLM prototype backend

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time
from datetime import datetime
import importlib

# Add RCE-LLM to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# NUCLEAR OPTION: Clear ALL Python cache
import shutil
for root, dirs, files in os.walk(current_dir):
    if '__pycache__' in dirs:
        pycache_path = os.path.join(root, '__pycache__')
        try:
            shutil.rmtree(pycache_path)
        except:
            pass

# Force reload modules to avoid caching issues
if 'rce_llm' in sys.modules:
    # Remove from sys.modules to force reimport
    modules_to_reload = [m for m in sys.modules if m.startswith('rce_llm')]
    for module in modules_to_reload:
        del sys.modules[module]

# Now import fresh
try:
    import rce_llm.core.renderer
except:
    pass

# Page config
st.set_page_config(
    page_title="RCE-LLM Admin Panel",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stage-header {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-weight: 600;
    }
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Import RCE-LLM components
try:
    from rce_llm import RCEEngine
    from rce_llm.core.types import Context, ContextIntent
    RCE_AVAILABLE = True
except ImportError as e:
    st.error(f"RCE-LLM import error: {e}")
    st.error("Please ensure RCE-LLM is installed: pip install -e .")
    RCE_AVAILABLE = False

# Header
st.markdown('<div class="main-header">🔍 RCE-LLM Admin Panel</div>', unsafe_allow_html=True)
st.markdown("**Relational Coherence Engine** - Advanced Semantic Processing with Coherence Optimization")
st.markdown(f"*Based on: DOI 10.5281/zenodo.17360372 | Author: Ismail Sialyen*")

# Sidebar for settings
with st.sidebar:
    # Authentication disabled - direct access
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = True

    # Version tracking
    st.info("**Version Info**")
    st.text("Commit: [DEBUG]")
    st.text("Build: 2025-11-12")
    st.text(f"Loaded: {datetime.now().strftime('%H:%M:%S')}")

    # Show renderer source location to verify it's loading the right file
    if RCE_AVAILABLE:
        try:
            import rce_llm.core.renderer as renderer_module
            import os
            renderer_file = renderer_module.__file__
            st.caption(f"File: {renderer_file}")

            # Check file modification time
            if renderer_file and os.path.exists(renderer_file):
                mtime = os.path.getmtime(renderer_file)
                from datetime import datetime
                mod_time = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
                st.caption(f"Modified: {mod_time}")

            # Check if fixes are in the source
            import inspect
            source = inspect.getsource(renderer_module.AnswerRenderer._generate_text)
            has_original_query = "original_query" in source and "source_text_preview" in source
            has_test_mode = "TEST MODE ACTIVE" in source
            st.caption(f"✓ ML fix: {'APPLIED' if has_original_query else 'MISSING'}")
            st.caption(f"✓ Test mode: {'ACTIVE' if has_test_mode else 'OFF'}")
        except Exception as e:
            st.caption(f"Error: {e}")
    else:
        st.caption("RCE not available")

    st.markdown("---")
    st.header("⚙️ Settings")

    # Optimization strategy
    optimization_strategy = st.selectbox(
        "Optimization Strategy",
        ["auto", "beam", "ilp"],
        index=0,
        help="auto: Automatic selection based on graph size\nbeam: Beam search (fast)\nilp: Integer Linear Programming (exact)"
    )

    # Beam size (for beam search)
    beam_size = st.slider(
        "Beam Size",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Number of candidates to keep in beam search"
    )

    # Display mode
    show_detailed_stats = st.checkbox(
        "Show Detailed Statistics",
        value=True,
        help="Display detailed computational metrics"
    )

    show_reasoning_trace = st.checkbox(
        "Show Reasoning Trace",
        value=True,
        help="Display step-by-step reasoning"
    )

    st.markdown("---")
    st.markdown("**📊 System Info**")
    st.metric("Status", "🟢 Online" if RCE_AVAILABLE else "🔴 Offline")
    st.metric("Pipeline", "RCE-LLM v1.0")
    st.metric("Paper DOI", "10.5281/zenodo.17360372")

# Main content area
if st.session_state.authenticated:

    if not RCE_AVAILABLE:
        st.error("❌ RCE-LLM engine not available. Please check installation.")
        st.stop()

    # Initialize engine (with caching)
    @st.cache_resource
    def get_rce_engine(strategy, beam_size):
        """Initialize RCE engine with caching."""
        config = {
            "optimization": {
                "strategy": strategy,
                "beam_size": beam_size,
            }
        }
        return RCEEngine(config)

    # Query input
    st.header("💬 Test Query")

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_area(
            "Enter your query:",
            height=100,
            placeholder="Example: A car travels 60 km/h for 30 minutes. How far in meters?"
        )

    with col2:
        st.write("")
        st.write("")
        process_btn = st.button("🚀 Process Query", type="primary", use_container_width=True)

    # Example queries
    with st.expander("📝 Example Queries"):
        st.markdown("""
        **Unit Conversion (F1 - Tests μ_units):**
        - A car travels 60 km/h for 30 minutes. How far in meters?
        - Convert 100 pounds to kilograms.

        **Temporal Reasoning (F2 - Tests μ_time):**
        - If a meeting starts at 2:30 PM and lasts 45 minutes, when does it end?
        - What time is it 3 hours before 11:00 AM?

        **Arithmetic (F3 - Tests μ_arith):**
        - If Alice has 5 apples and Bob has 3 more than Alice, how many does Bob have?
        - Calculate 25% of 80.

        **General Questions:**
        - What is machine learning?
        - Who is the president of the United States?
        """)

    if process_btn and query:

        with st.spinner("Processing query through RCE-LLM pipeline..."):
            try:
                # Get engine
                engine = get_rce_engine(optimization_strategy, beam_size)

                # Process query
                start_time = time.time()
                answer = engine.process(query)
                total_time = (time.time() - start_time) * 1000  # ms

                # Extract metrics
                num_entities = len(answer.subgraph.entities)
                num_relations = len(answer.subgraph.relations)
                graph_density = answer.subgraph.density

                # Token metrics
                input_tokens = len(query.split())
                output_tokens = len(answer.text.split())
                total_tokens = input_tokens + output_tokens

                # Display results
                st.markdown("---")
                st.header("📈 Results")

                # Answer
                st.subheader("💡 Generated Answer")
                st.markdown(f'<div class="success-box"><strong>{answer.text}</strong></div>', unsafe_allow_html=True)

                # Confidence and caveats
                if answer.caveats:
                    st.markdown(f'<div class="warning-box"><strong>Caveats:</strong><br>{"<br>".join(["• " + c for c in answer.caveats])}</div>', unsafe_allow_html=True)

                # Key Metrics
                st.subheader("📊 Key Metrics")

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Coherence", f"{answer.coherence.overall:.3f}")

                with col2:
                    st.metric("Confidence", f"{answer.confidence:.3f}")

                with col3:
                    st.metric("Entities", num_entities)

                with col4:
                    st.metric("Relations", num_relations)

                with col5:
                    st.metric("Density ρ", f"{graph_density:.4f}")

                # Token Analysis
                st.subheader("🔢 Token Analysis")

                token_col1, token_col2, token_col3, token_col4 = st.columns(4)

                with token_col1:
                    st.metric("Input Tokens", input_tokens)

                with token_col2:
                    st.metric("Output Tokens", output_tokens)

                with token_col3:
                    st.metric("Total Tokens", total_tokens)

                with token_col4:
                    compression = output_tokens / input_tokens if input_tokens > 0 else 0
                    st.metric("I/O Ratio", f"{compression:.2f}x")

                # Pipeline Execution Steps
                st.subheader("🔄 5-Stage RCE Pipeline")

                with st.expander("📥 Stage 1: Graph Construction (Graphizer)", expanded=True):
                    st.write(f"**Implementation:** G: X → G (Eq. 6)")
                    st.write(f"**Method:** SpaCy NER + Dependency Parsing")
                    st.write(f"**Entities Extracted:** {num_entities}")
                    st.write(f"**Relations Extracted:** {num_relations}")
                    st.write(f"**Graph Density:** ρ = |R|/|V|² = {graph_density:.4f}")
                    graphization_time = answer.computation_stats.get('graphization_time_ms', 0)
                    st.write(f"**Time:** {graphization_time:.1f}ms")

                    # Show entity types
                    entity_types = {}
                    for entity in answer.subgraph.entities.values():
                        entity_types[entity.semantic_type] = entity_types.get(entity.semantic_type, 0) + 1
                    st.write(f"**Entity Types:** {dict(entity_types)}")

                    # Embedding information
                    st.write("")
                    st.write("**Entity Embeddings:**")
                    embedding_dim = 96  # spaCy en_core_web_sm
                    st.write(f"• Model: spaCy en_core_web_sm")
                    st.write(f"• Dimension: {embedding_dim}D")
                    st.write(f"• Matrix: {num_entities} × {embedding_dim} = {num_entities * embedding_dim} elements")
                    st.write(f"• Adjacency Matrix: {num_entities} × {num_entities} = {num_entities * num_entities} elements")

                with st.expander("🎯 Stage 2: Context Extraction", expanded=True):
                    st.write(f"**Implementation:** E: X → C (Eq. 7)")

                    # We need to reconstruct context info from the answer
                    st.write(f"**Intent Detection:** Pattern-based classification")
                    st.write(f"**Domain Detection:** Keyword and entity-type analysis")
                    st.write(f"**Constraints:** Modal and temporal expression extraction")

                with st.expander("🔧 Stage 2.5: Entity Embedding Harmonization", expanded=True):
                    st.write(f"**Method:** L2 Normalization of Entity Embeddings")
                    st.write(f"**Purpose:** Standardize entity representations for coherence evaluation")
                    st.write("")
                    st.write(f"**Normalization Formula:**")
                    st.code("v̂ = v / ||v||₂")
                    st.write("")
                    embedding_dim = 96
                    st.write(f"**Entities Harmonized:** {num_entities}")
                    st.write(f"**Input:** {num_entities} × {embedding_dim} raw embeddings")
                    st.write(f"**Output:** {num_entities} × {embedding_dim} normalized embeddings")
                    st.write(f"**Properties:** Unit vectors (||v̂||₂ = 1)")

                with st.expander("📊 Stage 3: Coherence Evaluation", expanded=True):
                    st.write(f"**Implementation:** μ(Ω | C) = Σ w_k·μ_k(Ω | C) (Eq. 8)")
                    st.write(f"**Overall Coherence:** {answer.coherence.overall:.3f}")

                    coherence_time = answer.computation_stats.get('coherence_time_ms', 0)
                    st.write(f"**Evaluation Time:** {coherence_time:.1f}ms")

                    # Module scores with weights
                    st.write("**Module Scores (5 Coherence Dimensions):**")
                    for module_name, score in answer.coherence.module_scores.items():
                        weight = answer.coherence.module_weights.get(module_name, 0.2)

                        # Determine if module found violations
                        relevant_violations = [v for v in answer.coherence.violations
                                             if module_name.lower() in v.lower()]

                        if relevant_violations:
                            # Module found issues
                            st.progress(score, text=f"⚠️ {module_name}: {score:.3f} (weight: {weight:.2f}) - {len(relevant_violations)} violations")
                        else:
                            # Module passed
                            st.progress(score, text=f"✓ {module_name}: {score:.3f} (weight: {weight:.2f})")

                    # Show violations if any
                    if answer.coherence.violations:
                        st.write("**Detected Violations:**")
                        for violation in answer.coherence.violations[:5]:
                            st.write(f"- {violation}")
                        if len(answer.coherence.violations) > 5:
                            st.write(f"... and {len(answer.coherence.violations) - 5} more")

                with st.expander("⚡ Stage 4: Actualization Optimization", expanded=True):
                    st.write(f"**Implementation:** Ω* = arg max μ(Ω | C) s.t. Φ(Ω) (Eq. 14)")

                    # Get optimization info from computation stats
                    opt_time = answer.computation_stats.get('optimization_time_ms', 0)

                    st.write(f"**Strategy:** {optimization_strategy}")
                    st.write(f"**Original Graph:** {len(answer.subgraph.metadata.get('parent_graph', ''))} entities (full)")
                    st.write(f"**Optimized Subgraph:** {num_entities} entities, {num_relations} relations")
                    st.write(f"**Final Coherence:** {answer.coherence.overall:.3f}")
                    st.write(f"**Optimization Time:** {opt_time:.1f}ms")

                    if optimization_strategy == "beam":
                        st.write(f"**Beam Size:** {beam_size}")
                        st.write(f"**Complexity:** O(B·|R|·log|R|) = O({beam_size}·{num_relations}·{int(__import__('math').log2(num_relations) if num_relations > 0 else 0)})")
                    elif optimization_strategy == "ilp":
                        st.write(f"**Complexity:** O(2^|R|) = O(2^{num_relations})")

                with st.expander("📤 Stage 5: Answer Rendering", expanded=True):
                    st.write(f"**Implementation:** (y, c) = R(Ω*, C) (Eq. 15)")
                    st.write(f"**Answer Length:** {len(answer.text)} characters")
                    st.write(f"**Output Tokens:** {output_tokens}")
                    st.write(f"**Confidence:** {answer.confidence:.3f}")
                    st.write(f"**Evidence Items:** {len(answer.evidence_map)}")

                # Detailed Statistics
                if show_detailed_stats:
                    st.subheader("📈 Detailed Computational Statistics")

                    stat_col1, stat_col2, stat_col3 = st.columns(3)

                    with stat_col1:
                        st.markdown("**Timing Breakdown**")
                        graphization_time = answer.computation_stats.get('graphization_time_ms', 0)
                        coherence_time = answer.computation_stats.get('coherence_time_ms', 0)
                        optimization_time = answer.computation_stats.get('optimization_time_ms', 0)
                        total_time = answer.computation_stats.get('total_time_ms', 0)

                        st.write(f"Graphization: {graphization_time:.1f}ms")
                        st.write(f"Coherence: {coherence_time:.1f}ms")
                        st.write(f"Optimization: {optimization_time:.1f}ms")
                        st.write(f"**Total: {total_time:.1f}ms**")

                    with stat_col2:
                        st.markdown("**Complexity Analysis**")
                        st.write(f"Graphizer: O(n) where n={input_tokens}")
                        st.write(f"Coherence: O(|R|·K·d)")
                        st.write(f"  |R| = {num_relations} relations")
                        st.write(f"  K = 5 modules")
                        st.write(f"  d = feature dimension")

                    with stat_col3:
                        st.markdown("**Graph Statistics**")
                        st.write(f"Vertices |V|: {num_entities}")
                        st.write(f"Edges |R|: {num_relations}")
                        st.write(f"Density: {graph_density:.4f}")
                        st.write(f"Sparsity: {1 - graph_density:.4f}")

                # Reasoning Trace
                if show_reasoning_trace and answer.reasoning_trace:
                    st.subheader("🧠 Reasoning Trace")

                    for i, step in enumerate(answer.reasoning_trace, 1):
                        with st.expander(f"Step {i}: {step.get('step', 'Unknown')}"):
                            for key, value in step.items():
                                if key != 'step':
                                    st.write(f"**{key}:** {value}")

                # Knowledge Graph Visualization
                st.subheader("🕸️ Knowledge Graph Structure")

                graph_col1, graph_col2 = st.columns(2)

                with graph_col1:
                    st.markdown("**Entities (Sample)**")
                    entity_list = list(answer.subgraph.entities.values())[:10]
                    for entity in entity_list:
                        st.write(f"• {entity.text} ({entity.semantic_type}) - conf: {entity.confidence:.2f}")
                    if num_entities > 10:
                        st.write(f"... and {num_entities - 10} more")

                with graph_col2:
                    st.markdown("**Relations (Sample)**")
                    for rel in answer.subgraph.relations[:10]:
                        subject_entity = answer.subgraph.entities.get(rel.subject)
                        object_entity = answer.subgraph.entities.get(rel.object)
                        subject_text = subject_entity.text if subject_entity else rel.subject
                        object_text = object_entity.text if object_entity else rel.object
                        st.write(f"• {subject_text} →[{rel.predicate}]→ {object_text}")
                    if num_relations > 10:
                        st.write(f"... and {num_relations - 10} more")

                # Evidence Map
                if answer.evidence_map:
                    st.subheader("📚 Evidence Mapping")

                    with st.expander("View Evidence Map"):
                        for claim, evidence_list in list(answer.evidence_map.items())[:5]:
                            st.write(f"**Claim:** {claim}")
                            for evidence in evidence_list:
                                st.write(f"  - {evidence}")
                            st.write("")

                        if len(answer.evidence_map) > 5:
                            st.write(f"... and {len(answer.evidence_map) - 5} more evidence entries")

                # Mathematical Foundations
                st.subheader("🔬 Mathematical Foundations")

                with st.expander("View RCE-LLM Equations"):
                    st.markdown("""
                    **Core Optimization Objective (Eq. 2):**
                    ```
                    max μ(Ω | C)  subject to Φ(Ω)
                    ```

                    **Modular Coherence Functional (Eq. 8):**
                    ```
                    μ(Ω | C) = Σ_{k=1}^5 w_k(C)·μ_k(Ω | C)
                    ```

                    **Five Coherence Modules:**
                    - μ_units: Dimensional consistency (Eq. 9)
                    - μ_time: Temporal ordering (Eq. 10)
                    - μ_arith: Arithmetic validity (Eq. 11)
                    - μ_coref: Coreference resolution (Eq. 12)
                    - μ_entail: Factual entailment (Eq. 13)

                    **Complexity Advantage:**
                    - Transformer Attention: O(n²·d)
                    - RCE-LLM: O(|R|·K·d) where |R| ≪ n²
                    """)

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.exception(e)

    elif process_btn:
        st.warning("Please enter a query to process")

    # Footer
    st.markdown("---")
    st.markdown(
        "🤖 Powered by **RCE-LLM: Relational Coherence Engine** | "
        "Paper: DOI [10.5281/zenodo.17360372](https://doi.org/10.5281/zenodo.17360372) | "
        "Author: Ismail Sialyen | "
        "[GitHub](https://github.com/IsmaIkami/RCE-LLM-prototype)"
    )
