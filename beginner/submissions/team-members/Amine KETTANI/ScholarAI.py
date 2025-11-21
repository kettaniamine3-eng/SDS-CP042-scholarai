import os
import asyncio
import logging

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ==========================
# Setup
# ==========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Logging (shows in terminal where you run `streamlit run`)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==========================
# 1. TOPIC SPLITTING
# ==========================
async def split_into_subtopics(query: str) -> list[str]:
    logger.info("Splitting topic...")
    prompt = f"""
    Break the following question into 3 clear research subtopics:
    "{query}"
    Return ONLY a Python list, example: ["sub1", "sub2", "sub3"].
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.choices[0].message.content.strip()
    # In a quick prototype we use eval; in production use ast.literal_eval
    return eval(text)


# ==========================
# 2. RESEARCH FUNCTION
# ==========================
async def research_subtopic(subtopic: str) -> str:
    logger.info(f"Researching: {subtopic}")

    prompt = f"""
    Research this subtopic and write a concise summary:
    "{subtopic}"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


# ==========================
# 3. SYNTHESIS
# ==========================
async def synthesize_report(query: str, findings: list[str]) -> str:
    logger.info("Synthesizing report...")

    prompt = f"""
    Create a final, well-structured report for the query:
    "{query}"

    Here are the research findings:
    {findings}

    Return a clean narrative summary with bullet points where needed.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


# ==========================
# 4. OPTIMIZER
# ==========================
async def needs_more_research(findings: list[str]) -> bool:
    logger.info("Checking if more research is needed...")

    prompt = f"""
    Evaluate whether the following findings appear thorough enough:

    {findings}

    Answer with ONLY: "yes" or "no".
    "yes" = more research is needed.
    "no" = sufficient.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip().lower()
    return answer == "yes"


# ==========================
# MAIN PIPELINE (async)
# ==========================
async def research_pipeline(user_query: str):
    subtopics = await split_into_subtopics(user_query)

    # Show subtopics in logs
    logger.info(f"Subtopics: {subtopics}")

    # First research pass
    research_tasks = [research_subtopic(s) for s in subtopics]
    findings = await asyncio.gather(*research_tasks)

    # Optional extra passes if needed
    for _ in range(2):  # max 2 extra loops
        if await needs_more_research(findings):
            logger.info("Optimizer says more research needed. Doing another pass.")
            research_tasks = [research_subtopic(s) for s in subtopics]
            findings = await asyncio.gather(*research_tasks)
        else:
            break

    final_report = await synthesize_report(user_query, findings)
    return subtopics, findings, final_report


# Small sync wrapper for Streamlit
def run_research_sync(user_query: str):
    return asyncio.run(research_pipeline(user_query))


# ==========================
# STREAMLIT UI
# ==========================
def main():
    st.title("📚 AI Research Assistant")
    st.write("Ask a question and I’ll run a multi-step research workflow for you.")

    # User input
    user_query = st.text_area(
        "Your research question:",
        value="Should I buy, hold, or sell Tesla stock in current market conditions?",
        height=120
    )

    if st.button("Run research"):
        if not user_query.strip():
            st.warning("Please enter a question first.")
            return

        with st.spinner("Running research pipeline..."):
            subtopics, findings, final_report = run_research_sync(user_query)

        st.success("Done! 🎯")

        # Show subtopics
        st.subheader("🔹 Identified subtopics")
        for i, s in enumerate(subtopics, start=1):
            st.markdown(f"**{i}. {s}**")

        # Show findings for each subtopic
        st.subheader("📑 Research findings by subtopic")
        for i, (s, f) in enumerate(zip(subtopics, findings), start=1):
            with st.expander(f"Subtopic {i}: {s}"):
                st.write(f)

        # Show final report
        st.subheader("📝 Final synthesized report")
        st.write(final_report)


if __name__ == "__main__":
    main()
