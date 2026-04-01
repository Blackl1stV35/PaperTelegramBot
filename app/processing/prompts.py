"""
Prompt templates — few-shot JSON prompts for "What/How/To Whom"
and for figure description.

All prompts return strict JSON to simplify downstream parsing.
"""

# ─────────────────────────────────────────────────────────────────
#  TEXT SUMMARY:  What / How / To Whom
# ─────────────────────────────────────────────────────────────────
TEXT_SUMMARY_SYSTEM = """\
You are a senior research analyst. Given the full text of an academic paper, \
produce a concise structured summary in **strict JSON** (no markdown, no preamble).

Output exactly this schema:
{
  "what": "<1-2 sentences: the core contribution or discovery>",
  "how": "<1-2 sentences: the methodology, architecture, or approach>",
  "to_whom": "<1 sentence: who benefits — researchers, practitioners, industry, patients, etc.>",
  "domain": "<research domain from the list provided>",
  "confidence": <int 0-100: how confident you are in the summary>
}
"""

TEXT_SUMMARY_FEWSHOT = """\
### Example 1
Paper excerpt: "We introduce ReAct, a framework that synergizes reasoning and acting in large language models. ReAct prompts LLMs to generate verbal reasoning traces and task-specific actions in an interleaved manner, allowing for dynamic reasoning to create, maintain, and adjust plans for acting."

Output:
{
  "what": "ReAct — a framework that interleaves verbal reasoning traces with task actions inside LLMs, enabling dynamic plan creation and adjustment.",
  "how": "Prompting LLMs to alternate between chain-of-thought reasoning steps and concrete tool/API actions, evaluated on question-answering and decision-making benchmarks.",
  "to_whom": "AI researchers building autonomous agents and practitioners deploying LLM-based systems that must reason and act in real time.",
  "domain": "Agentic LLM Research",
  "confidence": 92
}

### Example 2
Paper excerpt: "We present AlphaFold 3, which achieves unprecedented accuracy in predicting 3D structures of protein-ligand complexes without multiple sequence alignments. The model uses a diffusion-based architecture trained on the Protein Data Bank."

Output:
{
  "what": "AlphaFold 3 predicts 3D structures of protein-ligand complexes with record accuracy, eliminating the need for multiple sequence alignments.",
  "how": "A diffusion-based neural architecture trained end-to-end on the Protein Data Bank, directly outputting atomic coordinates.",
  "to_whom": "Structural biologists, drug discovery teams, and computational chemists seeking faster and more accurate molecular modeling.",
  "domain": "Biology",
  "confidence": 95
}

### Example 3
Paper excerpt: "We demonstrate a 48-qubit trapped-ion quantum computer that executes fault-tolerant circuits. Using real-time decoding of surface codes, logical error rates drop below the physical error rate threshold for the first time."

Output:
{
  "what": "First demonstration of fault-tolerant quantum circuits on a 48-qubit trapped-ion machine where logical error rates fall below the physical error threshold.",
  "how": "Real-time decoding of surface codes on trapped-ion qubits, enabling mid-circuit error correction at scale.",
  "to_whom": "Quantum computing researchers, quantum hardware engineers, and organisations planning quantum advantage roadmaps.",
  "domain": "Quantum Research",
  "confidence": 88
}
"""

TEXT_SUMMARY_USER_TEMPLATE = """\
Paper domain: {domain}
Paper title: {title}

Full text (truncated to fit context):
{text}

Produce the JSON summary now. Output ONLY valid JSON, nothing else.
"""


# ─────────────────────────────────────────────────────────────────
#  FIGURE DESCRIPTION
# ─────────────────────────────────────────────────────────────────
FIGURE_DESCRIPTION_SYSTEM = """\
You are a scientific figure analyst. Given an image of a figure, diagram, or \
visualization from an academic paper, produce a strict JSON response:

{
  "description": "<one clear sentence describing what the figure shows>",
  "relevance": "<one sentence explaining how this figure relates to the paper's core discovery or methodology>"
}

Be precise and technical. Do NOT speculate beyond what is visible.
"""

FIGURE_DESCRIPTION_FEWSHOT = """\
### Example 1
[Image: A bar chart comparing F1 scores of 5 models on 3 benchmarks]
Output:
{
  "description": "Bar chart comparing F1 scores of five models (GPT-4, Claude, Llama-2, Falcon, Mistral) across three QA benchmarks (SQuAD, TriviaQA, NaturalQuestions).",
  "relevance": "Demonstrates the proposed model's competitive advantage on open-domain question answering, supporting the paper's claim of state-of-the-art performance."
}

### Example 2
[Image: A block diagram showing encoder-decoder architecture with attention layers]
Output:
{
  "description": "Block diagram of the proposed encoder-decoder architecture with multi-head cross-attention layers connecting the vision encoder to the language decoder.",
  "relevance": "Illustrates the core architectural contribution — the novel cross-attention fusion mechanism that enables vision-language grounding."
}

### Example 3
[Image: A scatter plot of galaxy redshifts vs luminosity]
Output:
{
  "description": "Scatter plot showing photometric redshift estimates versus spectroscopic ground truth for 10,000 galaxies, color-coded by luminosity class.",
  "relevance": "Validates the neural network's redshift prediction accuracy, central to the paper's claim of replacing traditional template-fitting methods."
}
"""

FIGURE_DESCRIPTION_USER = """\
This figure is from the paper: "{title}"
Paper domain: {domain}

Analyse the image and produce the JSON response. Output ONLY valid JSON.
"""


# ─────────────────────────────────────────────────────────────────
#  WEEKLY META-SYNTHESIS
# ─────────────────────────────────────────────────────────────────
WEEKLY_SYNTHESIS_SYSTEM = """\
You are a chief research strategist. Given summaries of approved papers across \
multiple scientific domains from the past week, produce a cross-domain \
meta-synthesis report in Markdown.

Structure:
1. **Macro Trends** — 3-5 overarching trends you see across all domains.
2. **Cross-Domain Links** — unexpected connections between papers in different fields.
3. **Emerging Signals** — weak signals that could become important in 3-6 months.
4. **Recommended Deep Dives** — 2-3 papers deserving closer reading, with reasons.

Be concrete — cite specific papers by title. Keep the report under 1500 words.
"""

WEEKLY_SYNTHESIS_USER = """\
Here are the approved paper summaries from {start_date} to {end_date}:

{paper_summaries}

Produce the cross-domain meta-synthesis report now.
"""
