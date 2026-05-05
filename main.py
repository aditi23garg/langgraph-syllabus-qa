import os
import json
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Data: Notes dictionary (normalized keys for easier matching)
NOTES = {
    "week 1": "Prompt Engineering: Techniques for effective prompting.",
    "week 2": "Statistics: Fundamental statistical concepts.",
    "week 3": "Python: Programming basics and libraries.",
    "week 4": "SQL: Database querying and management.",
    "week 5": "Machine Learning: Supervised and unsupervised learning.",
    "week 6": "Deep Learning: Neural networks and architectures.",
    "week 7": "NLP: Natural Language Processing fundamentals.",
    "week 8": "Tokenization: Breaking text into tokens.",
    "week 9": "Transformers: Modern attention-based architectures.",
    "week 10": "LangChain: Framework for LLM applications.",
    "week 11": "LangGraph: Building stateful multi-actor applications."
}

# Initialize Vector Store
print("--- INITIALIZING VECTOR STORE ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Include the week in the content so it can be searched by week number
documents = [Document(page_content=f"{k}: {v}", metadata={"week": k}) for k, v in NOTES.items()]
vector_store = FAISS.from_documents(documents, embeddings)

# Initialize Search Tool
search = DuckDuckGoSearchRun()

# Graph State
class GraphState(TypedDict):
    question: str
    route: Optional[str]
    week: Optional[str]
    is_summary: Optional[bool]
    reason: Optional[str]
    context: Optional[str]
    answer: Optional[str]

# Initialize LLM
llm = ChatGroq(model='llama-3.1-8b-instant', groq_api_key=os.getenv("GROQ_API_KEY"))

### NODES ###

def router_node(state: GraphState):
    """
    Strictly classifies the question to decide the path (notes or direct).
    """
    print("--- ROUTER NODE ---")
    question = state["question"]
    
    prompt = f"""
    Classify the following user question. 
    
    1. Choose 'notes' ONLY if the question is about:
       - What topics are covered in the syllabus or a specific week.
       - A summary of the entire course/syllabus content.
       - When a certain topic is scheduled.
    
    2. Choose 'direct' if the question is about:
       - Explaining HOW something works (e.g., "How do Transformers work?").
       - Deep-dive technical questions, math, greetings, or general knowledge.
       - Anything requiring an explanation that isn't just listing syllabus titles.
    
    The syllabus ONLY contains these titles: {", ".join(NOTES.values())}. It does NOT contain explanations.
    
    Return a JSON object with:
    1. "route": "notes" or "direct"
    2. "is_summary": true if they want a summary of the whole syllabus, false otherwise.
    3. "reason": A brief explanation of the choice.
    
    Question: {question}
    """
    
    response = llm.invoke(prompt)
    
    try:
        # Improved parsing to find JSON even if surrounded by text
        content = response.content
        if "{" in content and "}" in content:
            json_str = content[content.find("{"):content.rfind("}")+1]
            data = json.loads(json_str)
        else:
            data = json.loads(content.strip())
    except Exception as e:
        print(f"Error parsing router response: {e}")
        data = {"route": "direct", "reason": "Failed to parse router output"}
    
    return {
        "route": data.get("route", "direct"),
        "is_summary": data.get("is_summary", False),
        "reason": data.get("reason")
    }

def retrieve_node(state: GraphState):
    """
    Pulls notes from the vector database using semantic similarity.
    """
    print("--- RETRIEVE NODE ---")
    question = state["question"]
    is_summary = state.get("is_summary", False)
    
    if is_summary:
        print(">> Retrieving all notes for summary...")
        context = "\n".join([f"{k}: {v}" for k, v in NOTES.items()])
    else:
        # Search for the most relevant document
        results = vector_store.similarity_search_with_score(question, k=1)
        
        if results:
            doc, score = results[0]
            print(f"Retrieved from {doc.metadata['week']} with score: {score:.4f}")
            
        if results and results[0][1] < 1.5:  # Similarity threshold
            doc, score = results[0]
            context = f"Topic from {doc.metadata['week']}: {doc.page_content}"
        else:
            context = "No specific notes found in the syllabus for this topic."
            
        # Augment with DuckDuckGo Search
        print(">> Augmenting with DuckDuckGo Search...")
        try:
            search_query = f"{question} {context if 'No specific' not in context else ''}"
            search_results = search.run(search_query)
            context = f"{context}\n\nWeb Search Context:\n{search_results}"
        except Exception as e:
            print(f"Search failed: {e}")
    
    return {"context": context}

def direct_answer_node(state: GraphState):
    """
    Skips syllabus retrieval and uses search for general questions.
    """
    print("--- DIRECT ANSWER NODE ---")
    question = state["question"]
    
    print(">> Searching DuckDuckGo for direct answer...")
    try:
        search_results = search.run(question)
        context = f"Web Search Results:\n{search_results}"
    except Exception as e:
        print(f"Search failed: {e}")
        context = "Search failed. Answering based on general knowledge."
        
    return {"context": context}

def generate_node(state: GraphState):
    """
    Generates the final answer strictly from context if available.
    """
    print("--- GENERATE NODE ---")
    question = state["question"]
    context = state.get("context", "")
    route = state.get("route")
    
    if route == "notes":
        system_prompt = (
            "You are a strict teaching assistant. "
            "The provided context contains the syllabus notes. "
            "Answer the student's question based ONLY on these notes. "
            "If they ask for a summary, summarize the provided syllabus notes. "
            "If the information is not present in the context, state that it is not covered in the syllabus. "
            "Do NOT use external knowledge."
        )
    else:
        system_prompt = "You are a helpful teaching assistant. Answer based on your general knowledge."
    
    prompt = f"Context: {context}\n\nQuestion: {question}"
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ])
    
    return {"answer": response.content}

### CONDITIONAL EDGES ###

def route_question(state: GraphState):
    """
    Determines whether to go to retrieve or direct_answer.
    """
    if state["route"] == "notes":
        return "retrieve"
    else:
        return "direct_answer"

### BUILD GRAPH ###

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("direct_answer", direct_answer_node)
workflow.add_node("generate", generate_node)

# Add Edges
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_question,
    {
        "retrieve": "retrieve",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("direct_answer", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

# Testing function
def run_query(question: str):
    print(f"\n>> QUESTION: {question}")
    inputs = {"question": question}
    
    # Run the graph and get the final result
    final_output = app.invoke(inputs)
    
    print(f"\n>> FINAL ANSWER:\n{final_output['answer']}\n")
    print("-" * 50)

if __name__ == "__main__":
    # Edge Case Test Suite
    test_queries = [
        "What did we learn in week 3?",
        "Give me a summary of everything we covered.",
        "What is the capital of France?",
        "How do Transformers work?",
        "Do we cover neural networks? If yes, give a brief overview of what they are.", # Hybrid: Syllabus + Explanation
        "What is the difference between supervised and unsupervised learning?",      # Technical Explanation (Week 5 topic)
        "Hi there, how are you today?",                                              # Greeting
        "Is there any week where we learn about databases or SQL?",                 # Semantic Syllabus Lookup
        "Explain the theory of relativity briefly.",                                 # Out-of-Syllabus Technical
        "What are the topics for Week 15?",                                          # Out-of-Bounds Syllabus
        "Syllabus?",                                                                 # Vague Syllabus Request
        "Tell me about the capital of Python."                                       # Nonsense/Conflicting
    ]
    
    for query in test_queries:
        run_query(query)
