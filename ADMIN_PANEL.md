# RCE-LLM Admin Panel

**Production-ready Streamlit admin interface for the RCE-LLM engine**

Author: Ismail Sialyen
Based on: DOI 10.5281/zenodo.17360372

---

## Overview

The RCE-LLM Admin Panel provides a comprehensive web interface for testing and analyzing the Relational Coherence Engine. It features:

- ✅ **Interactive query testing** with real-time processing
- ✅ **5-stage pipeline visualization** (Algorithm 1 from paper)
- ✅ **Detailed coherence analysis** with all 5 modules
- ✅ **Graph structure inspection** (entities and relations)
- ✅ **Performance metrics** and timing analysis
- ✅ **Evidence mapping** for explainability
- ✅ **Mathematical foundations** reference

---

## Quick Start

### Option 1: Automated Launch (Recommended)

```bash
cd ~/RCE-LLM-prototype
./run_admin.sh
```

This script will:
1. Create/activate virtual environment
2. Install all dependencies
3. Download spaCy model if needed
4. Install RCE-LLM package
5. Launch the admin panel

### Option 2: Manual Launch

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements_admin.txt
python -m spacy download en_core_web_sm

# Install RCE-LLM
pip install -e .

# Launch admin panel
streamlit run admin_panel.py
```

The admin panel will open automatically in your browser at `http://localhost:8501`

---

## Features

### 1. Authentication
- **Username:** `admin`
- **Password:** `rce2024`
- Simple authentication for demo/testing purposes
- Can be extended with proper auth for production

### 2. Query Processing

Enter any natural language query to test the RCE engine:

**Example Queries:**

**Unit Conversion (Tests μ_units):**
```
A car travels 60 km/h for 30 minutes. How far in meters?
Convert 100 pounds to kilograms.
```

**Temporal Reasoning (Tests μ_time):**
```
If a meeting starts at 2:30 PM and lasts 45 minutes, when does it end?
What time is it 3 hours before 11:00 AM?
```

**Arithmetic (Tests μ_arith):**
```
If Alice has 5 apples and Bob has 3 more than Alice, how many does Bob have?
Calculate 25% of 80.
```

**General Questions:**
```
What is machine learning?
Who is the president of the United States?
```

### 3. Pipeline Visualization

The admin panel shows all 5 stages of the RCE pipeline:

#### Stage 1: Graph Construction (Graphizer)
- Implementation: `G: X → G` (Eq. 6)
- Shows entities extracted, relations, graph density
- Entity type distribution
- Timing metrics

#### Stage 2: Context Extraction
- Implementation: `E: X → C` (Eq. 7)
- Intent classification
- Domain detection
- Constraint extraction

#### Stage 3: Coherence Evaluation
- Implementation: `μ(Ω | C) = Σ w_k·μ_k` (Eq. 8)
- All 5 module scores with weights:
  - μ_units (dimensional analysis)
  - μ_time (temporal ordering)
  - μ_arith (arithmetic validity)
  - μ_coref (coreference resolution)
  - μ_entail (factual entailment)
- Violation detection and reporting

#### Stage 4: Actualization Optimization
- Implementation: `Ω* = arg max μ(Ω | C)` (Eq. 14)
- Strategy used (auto/beam/ilp)
- Subgraph selection
- Optimization time
- Complexity analysis

#### Stage 5: Answer Rendering
- Implementation: `(y, c) = R(Ω*, C)` (Eq. 15)
- Natural language answer
- Confidence score
- Evidence mapping
- Caveats and warnings

### 4. Metrics Dashboard

**Key Metrics:**
- Coherence score (0-1)
- Confidence score (0-1)
- Number of entities
- Number of relations
- Graph density ρ

**Token Analysis:**
- Input tokens
- Output tokens
- Total tokens
- Input/output ratio

**Performance:**
- Graphization time
- Coherence evaluation time
- Optimization time
- Total processing time

**Complexity:**
- Graph size |V|, |R|
- Density and sparsity
- Computational complexity (O notation)

### 5. Configuration Options

**Optimization Strategy:**
- `auto`: Automatic selection based on graph size
- `beam`: Beam search (fast, approximate)
- `ilp`: Integer Linear Programming (exact, slow)

**Beam Size:** 5-50 (for beam search)

**Display Options:**
- Show detailed statistics
- Show reasoning trace
- Toggle various visualization elements

### 6. Graph Visualization

**Entity Display:**
- Entity text
- Semantic type
- Confidence score
- Sample of top entities

**Relation Display:**
- Subject → Predicate → Object format
- Relation types
- Sample of top relations

**Evidence Map:**
- Claim → Evidence mapping
- Source attribution
- Traceability for each statement

---

## Architecture Alignment

The admin panel directly reflects the paper's architecture:

| Paper Component | Implementation | Admin Panel Display |
|----------------|----------------|---------------------|
| Eq. 6: G = (V,R,τ,σ) | `Graphizer` | Stage 1: Graph Construction |
| Eq. 7: C = (intent, domain, ...) | `ContextExtractor` | Stage 2: Context Extraction |
| Eq. 8: μ(Ω\|C) = Σ w_k·μ_k | `CoherenceAggregator` | Stage 3: Coherence Evaluation |
| Eq. 9-13: Five modules | `*CoherenceModule` | Module scores with weights |
| Eq. 14: Ω* = arg max | `ActualizationOptimizer` | Stage 4: Optimization |
| Eq. 15: (y,c) = R(Ω*,C) | `AnswerRenderer` | Stage 5: Answer Rendering |

---

## API Integration

The admin panel can be easily integrated with your existing systems:

### Programmatic Access

```python
from rce_llm import RCEEngine

# Initialize engine
engine = RCEEngine({
    "optimization": {
        "strategy": "beam",
        "beam_size": 10,
    }
})

# Process query
answer = engine.process("Your query here")

# Access results
print(f"Answer: {answer.text}")
print(f"Confidence: {answer.confidence}")
print(f"Coherence: {answer.coherence.overall}")
print(f"Entities: {len(answer.subgraph.entities)}")
```

### REST API (Future)

A REST API can be easily added using FastAPI:

```python
from fastapi import FastAPI
from rce_llm import RCEEngine

app = FastAPI()
engine = RCEEngine()

@app.post("/process")
async def process_query(query: str):
    answer = engine.process(query)
    return answer.to_dict()
```

---

## Customization

### Styling

The admin panel uses custom CSS that can be modified in `admin_panel.py`:

```python
st.markdown("""
<style>
    .main-header { ... }
    .success-box { ... }
    # Add your custom styles
</style>
""", unsafe_allow_html=True)
```

### Authentication

For production, replace the simple authentication with:
- OAuth2 integration
- Database-backed user management
- Role-based access control
- Session management

### Branding

Customize the header, footer, and branding elements:

```python
st.markdown('<div class="main-header">Your Company Name</div>', ...)
st.markdown("Footer with your branding...")
```

---

## Performance Optimization

### Caching

The admin panel uses Streamlit's caching for engine initialization:

```python
@st.cache_resource
def get_rce_engine(strategy, beam_size):
    return RCEEngine(config)
```

This ensures the engine is only initialized once per configuration.

### Memory Management

For production with many concurrent users:
- Use a separate engine instance per session
- Implement request queuing
- Add resource limits
- Monitor memory usage

---

## Deployment Options

### 1. Local Development

```bash
./run_admin.sh
```

### 2. Streamlit Community Cloud

1. Push to GitHub
2. Connect to Streamlit Cloud
3. Deploy from repository
4. Set secrets in dashboard

### 3. Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt -r requirements_admin.txt
RUN python -m spacy download en_core_web_sm
RUN pip install -e .

EXPOSE 8501

CMD ["streamlit", "run", "admin_panel.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 4. Production Server

Use Gunicorn or similar:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker admin_api:app
```

---

## Troubleshooting

### Import Errors

```bash
# Ensure RCE-LLM is installed
pip install -e .

# Check Python path
python -c "import rce_llm; print(rce_llm.__file__)"
```

### SpaCy Model Missing

```bash
python -m spacy download en_core_web_sm
```

### Port Already in Use

```bash
# Use different port
streamlit run admin_panel.py --server.port 8502
```

### Slow Processing

- Reduce beam size (5-10)
- Use "beam" strategy instead of "ilp"
- Enable caching
- Process shorter queries

---

## Screenshots

**Main Interface:**
- Query input area
- Process button
- Example queries expandable

**Results Display:**
- Generated answer with confidence
- Key metrics dashboard
- Token analysis
- 5-stage pipeline details

**Coherence Analysis:**
- Module scores with visual progress bars
- Violations highlighted
- Evidence mapping
- Graph structure

---

## Comparison with Original

This admin panel is adapted from your original but enhanced with:

✅ **RCE-LLM integration** instead of old RCE engine
✅ **5 coherence modules** instead of generic scoring
✅ **Mathematical equations** from the paper
✅ **Evidence mapping** for explainability
✅ **Reasoning traces** for transparency
✅ **Optimization strategies** (beam/ilp/auto)
✅ **Production-ready code** (no mocks)

---

## Future Enhancements

Planned features:

1. **Benchmark Integration**
   - F1-F5 task generators
   - Automated testing
   - Performance comparisons

2. **Visualization**
   - Interactive graph rendering
   - 3D entity embeddings
   - Coherence heatmaps

3. **Export Options**
   - PDF reports
   - JSON API responses
   - CSV metrics export

4. **Advanced Analytics**
   - Historical query tracking
   - Performance trends
   - A/B testing framework

---

## Support

For issues or questions:
- **Author:** Ismail Sialyen
- **Email:** is.sialyen@gmail.com
- **GitHub:** https://github.com/IsmaIkami/RCE-LLM-prototype
- **Paper:** DOI 10.5281/zenodo.17360372

---

## License

MIT License - See LICENSE file

---

**Built with academic rigor. Ready for production. Designed for transparency.**
