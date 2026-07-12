import json
import streamlit as st
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load API keys from .env
load_dotenv()

# Streamlit Page Setup
st.set_page_config(page_title="Agentic Workflow Quiz", page_icon="🎓", layout="wide")

st.title("🎓 Agentic Workflow & Systems Quiz Agent")
st.caption("Test your knowledge on Agent Pipelines, Tools, Evaluation, and Single-Agent Architectures.")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)

# ---------------------------------------------------------
# 1. STATE DEFINITION & TOOLS
# ---------------------------------------------------------
class QuizAgentState(TypedDict):
    question: str
    user_answer: str
    score: int
    feedback: str
    retries: int

evaluator_schema = [{
    "type": "function",
    "function": {
        "name": "grade_answer",
        "description": "Evaluates a quiz answer and assigns a score from 0 to 100 with actionable feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "Score out of 100 based on accuracy and technical completeness."
                },
                "feedback": {
                    "type": "string",
                    "description": "Detailed explanation of what was correct or missing."
                }
            },
            "required": ["score", "feedback"]
        }
    }
}]

llm_evaluator = llm.bind_tools(evaluator_schema)

# ---------------------------------------------------------
# 2. GRAPH NODES & RETRY EDGE
# ---------------------------------------------------------
def evaluator_node(state: QuizAgentState) -> QuizAgentState:
    """Evaluator Role: Assesses user's answer against core agent concepts."""
    prompt = f"""
    You are an expert evaluator in AI Agent Architectures.
    
    Quiz Question: {state['question']}
    User's Submitted Answer: {state['user_answer']}
    
    Evaluate the response for accuracy, technical depth, and completeness.
    Use the `grade_answer` tool to output the score and feedback.
    """
    
    res = llm_evaluator.invoke(prompt)
    
    if res.tool_calls:
        args = res.tool_calls[0]["args"]
        state["score"] = args.get("score", 0)
        state["feedback"] = args.get("feedback", "No feedback generated.")
    else:
        state["score"] = 50
        state["feedback"] = "Answer processed, but formal structured grading failed."
        state["retries"] += 1

    return state

def retry_logic(state: QuizAgentState) -> str:
    if state["score"] == 0 and state["retries"] < 2:
        return "retry"
    return "proceed"

@st.cache_resource
def build_quiz_pipeline():
    workflow = StateGraph(QuizAgentState)
    
    workflow.add_node("evaluator", evaluator_node)
    workflow.set_entry_point("evaluator")
    
    workflow.add_conditional_edges(
        "evaluator",
        retry_logic,
        {
            "retry": "evaluator",
            "proceed": END
        }
    )
    return workflow.compile()

app_graph = build_quiz_pipeline()

# ---------------------------------------------------------
# 3. QUIZ QUESTIONS DATASET
# ---------------------------------------------------------
QUIZ_QUESTIONS = [
    "Explain the concept of a stateful directed graph in agent pipelines. How does it differ from a simple linear pipeline?",
    "Describe the role of nodes and edges in an agent workflow. Give an example of each.",
    "What is conditional routing in an agent system? Design a simple rule-based routing logic for three different query types.",
    "Why are cycles (loops) important in agent pipelines? Provide a use case where a retry loop is necessary.",
    "Explain how a single-agent system can simulate multi-agent behavior internally.",
    "What are JSON schema tools? How do they help in structuring tool inputs and outputs?",
    "Compare sequential tool calls and parallel tool calls. When would you prefer one over the other?",
    "How would you implement error handling in a tool-using agent? Provide at least two strategies.",
    "What is trajectory evaluation in agent systems? Why is it important beyond just checking final output?",
    "Define task completion rate and cost metrics. How would you measure and optimize them in a real-world system?"
]

# ---------------------------------------------------------
# 4. STREAMLIT UI & INTERACTION LOGIC
# ---------------------------------------------------------

# Question selection sidebar
st.sidebar.title("📌 Quiz Navigation")
selected_q_idx = st.sidebar.radio(
    "Select Question:", 
    options=range(len(QUIZ_QUESTIONS)),
    format_func=lambda i: f"Q{i+1}: {QUIZ_QUESTIONS[i][:30]}..."
)

current_question = QUIZ_QUESTIONS[selected_q_idx]

st.subheader(f"Question {selected_q_idx + 1}")
st.write(f"**{current_question}**")

# Input field for answer
user_answer = st.text_area("Write your answer here:", height=150, key=f"q_{selected_q_idx}")

if st.button("Submit & Evaluate", type="primary"):
    if not user_answer.strip():
        st.warning("Please type an answer before submitting!")
    else:
        st.divider()
        
        # Step A: Run Graph Evaluation (Evaluator Node)
        with st.spinner("🤖 Agent Graph processing and scoring your answer..."):
            init_state = {
                "question": current_question,
                "user_answer": user_answer,
                "score": 0,
                "feedback": "",
                "retries": 0
            }
            final_state = app_graph.invoke(init_state)

        # Step B: Display Score Badge
        score = final_state["score"]
        if score >= 80:
            st.success(f"### 🏆 Score: {score}/100")
        elif score >= 50:
            st.warning(f"### ⚠️ Score: {score}/100")
        else:
            st.error(f"### ❌ Score: {score}/100")

        # Step C: Stream Synthesizer Explanation via Streamlit API
        st.markdown("### 📝 Detailed Explanation & Coaching")
        
        def stream_explanation():
            prompt = f"""
            The student scored {score}/100 on the question: '{current_question}'
            User Answer: '{user_answer}'
            Evaluator Notes: '{final_state['feedback']}'
            
            Synthesize a friendly, clear feedback response detailing strengths and missing aspects.
            """
            for chunk in llm.stream(prompt):
                yield chunk.content

        # Stream real-time tokens to UI
        st.write_stream(stream_explanation())




