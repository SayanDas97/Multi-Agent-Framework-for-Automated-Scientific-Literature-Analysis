from agents.gemini_client import generate_text

AGENTS = [
    ("Research Discovery Agent",
     "Identify the most relevant papers and evidence passages for the research question. Summarise the scope of the supplied evidence. Do not invent papers."),
    ("Paper Analysis Agent",
     "Analyse the retrieved passages. Extract each paper's apparent objective, methods, dataset or study context, findings and limitations only when supported. Mark missing details as not stated in the supplied passages."),
    ("Comparative Analysis Agent",
     "Compare approaches and findings across the supplied sources. Identify similarities, differences, trade-offs and contradictory findings. Cite source labels and page numbers."),
    ("Research Gap Identification Agent",
     "Suggest possible research gaps that are cautiously inferable from the supplied evidence. Distinguish an evidence-based gap from a hypothesis requiring further literature review."),
    ("Fact-Checking Agent",
     "Check the main claims against the retrieved passages. For each claim, label Supported, Contradicted, or Insufficient evidence. Give source and page references. Do not claim external verification."),
    ("Report Generation Agent",
     "Create a coherent academic-style research synthesis with an executive summary, scope, methods/themes, comparison, possible research gaps, limitations, conclusion and numbered evidence references.")
]

def format_evidence(evidence):
    blocks = []
    for i, item in enumerate(evidence, 1):
        blocks.append(
            f"[E{i}] Source: {item['source']} | Page: {item['page']} | Retrieval similarity: {item['score']:.3f}\n"
            f"{item['text']}"
        )
    return "\n\n".join(blocks)

def run_research_workflow(question, retriever, api_key, model_name="gemini-2.5-flash", top_k=6):
    evidence = retriever.search(question, top_k=top_k)
    if not evidence:
        raise ValueError("No evidence was retrieved for this question.")
    evidence_text = format_evidence(evidence)
    previous_outputs = []
    agent_outputs = {}

    for agent_name, instruction in AGENTS:
        prior = "\n\n".join(
            f"### {name}\n{content}" for name, content in previous_outputs
        )
        prompt = f"""
You are the {agent_name} in a multi-agent academic research assistant.
Research question: {question}

Your task:
{instruction}

Strict evidence rules:
- Use ONLY the evidence passages and prior agent notes supplied below.
- Never fabricate authors, titles, dates, datasets, numerical results, citations or quotations.
- Cite evidence using [E1], [E2], etc. and preserve the source filename and page when possible.
- If evidence is insufficient, say so clearly.
- This is research assistance, not a substitute for reading the complete papers.

RETRIEVED EVIDENCE:
{evidence_text}

PRIOR AGENT NOTES:
{prior if prior else "No prior notes. You are the first agent."}
"""
        output = generate_text(api_key, model_name, prompt)
        agent_outputs[agent_name] = output
        previous_outputs.append((agent_name, output))

    final_report = agent_outputs["Report Generation Agent"]
    return {
        "final_report": final_report,
        "agent_outputs": agent_outputs,
        "evidence": evidence
    }
