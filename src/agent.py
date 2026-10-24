"""Agentic RAG engine using Qwen3 native tool calling and ReAct loop."""
import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

from src.config import LLM_MODEL, OLLAMA_URL
from src.retrieval import hybrid_search
from src.generation import format_context_blocks, extract_citations


AGENT_SYSTEM_PROMPT = """You are an expert Multimodal RAG Assistant.
You have access to tools that can search the local knowledge base and inspect available documents.

Guidelines:
1. If the user asks a question about documents, policies, or specific facts, call the `search_knowledge_base` tool to retrieve relevant evidence before answering.
2. If the user asks a conversational question or greeting, answer directly without calling unnecessary tools.
3. If the retrieved evidence does not contain enough information, you may re-query with a refined search phrase, or state that the documents do not contain the answer.
4. Always cite specific sources and pages in your final answer, e.g. [Source: filename, Page: X].
5. Never invent facts not supported by the retrieved context.
"""


def run_agentic_rag(
    query: str,
    db: Chroma,
    all_documents: List[Document] = None,
    model: str = LLM_MODEL,
    temperature: float = 0.1,
    max_turns: int = 3,
    top_k: int = 4,
) -> Dict[str, Any]:
    """Execute autonomous agentic loop: Thought -> Action -> Observation -> Final Answer."""
    retrieved_evidence: List[Document] = []
    tool_steps: List[Dict[str, Any]] = []

    # 1. Define tools dynamically closed over active db and document index
    @tool
    def search_knowledge_base(search_query: str) -> str:
        """Search the indexed documents and visual diagrams for relevant passages and figures."""
        docs = hybrid_search(search_query, db, all_documents or [], k=top_k)
        retrieved_evidence.extend(docs)
        if not docs:
            return "No matching documents or passages found in the knowledge base."
        return format_context_blocks(docs)

    @tool
    def inspect_indexed_documents() -> str:
        """List all indexed documents, sources, and chunk statistics in the knowledge base."""
        if not all_documents:
            return "No documents are currently indexed in the knowledge base."
        sources = list(dict.fromkeys(d.metadata.get("source", "Unknown") for d in all_documents))
        return f"Currently indexed documents ({len(all_documents)} chunks total): {', '.join(sources)}"

    tools_map = {
        "search_knowledge_base": search_knowledge_base,
        "inspect_indexed_documents": inspect_indexed_documents,
    }

    # 2. Bind tools to Ollama LLM
    llm = ChatOllama(model=model, temperature=temperature, base_url=OLLAMA_URL)
    llm_with_tools = llm.bind_tools(list(tools_map.values()))

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    # 3. Agentic Loop (Thought -> Tool Action -> Observation)
    for turn in range(max_turns):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # Check if the agent called any tools
        if not ai_msg.tool_calls:
            # No further tool calls, agent has formulated final answer
            break

        # Process each tool call
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            call_id = tool_call.get("id", f"call_{turn}")

            tool_fn = tools_map.get(tool_name)
            if tool_fn:
                try:
                    tool_output = tool_fn.invoke(tool_args)
                except Exception as e:
                    tool_output = f"Tool execution error: {str(e)}"
            else:
                tool_output = f"Unknown tool: {tool_name}"

            tool_steps.append({
                "turn": turn + 1,
                "tool": tool_name,
                "args": tool_args,
                "output_snippet": str(tool_output)[:300],
            })

            messages.append(ToolMessage(content=str(tool_output), tool_call_id=call_id))

    # Deduplicate retrieved evidence
    unique_evidence = []
    seen_contents = set()
    for doc in retrieved_evidence:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_evidence.append(doc)

    final_content = messages[-1].content if messages else "No response generated."
    if isinstance(final_content, list):
        final_content = "\n".join(str(c) for c in final_content)

    return {
        "answer": final_content.strip(),
        "tool_steps": tool_steps,
        "citations": extract_citations(unique_evidence),
        "evidence": unique_evidence,
    }
