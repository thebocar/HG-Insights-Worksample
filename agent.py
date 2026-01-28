import asyncio
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from openai import OpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# ====== SECRETS (local) ======
import os
def get_secret(name: str) -> str:
    val = None
    try:
        val = st.secrets.get(name)
    except Exception:
        val = None
    if not val:
        val = os.getenv(name)
    return val

OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
PHOENIX_MCP_URL = get_secret("PHOENIX_MCP_URL")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY missing")
if not PHOENIX_MCP_URL:
    raise RuntimeError("PHOENIX_MCP_URL missing")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"  # tu peux changer
# =============================

llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


def run_async(coro):
    def _runner():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_runner).result()


async def mcp_call(tool_name: str, params: dict):
    async with streamable_http_client(PHOENIX_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_name, params)


def clean_domain(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("https://", "").replace("http://", "")
    s = s.split("/")[0].strip()
    return s


def extract_payload(result):
    """
    Retourne:
    - text: texte concaténé (pour le prompt LLM)
    - raw: contenu brut (debug)
    """
    items = getattr(result, "content", []) or []
    texts = []
    raw = []

    for c in items:
        raw.append(str(c))
        t = getattr(c, "text", None)
        if t:
            texts.append(t)

    return {
        "text": "\n".join(texts).strip(),
        "raw": raw
    }


def make_brief_with_llm(domain: str, firmo: dict, techno: dict) -> str:
    prompt = f"""
You are a B2B sales assistant helping a sales rep prepare for a first interaction.

Goal:
Before a first interaction, identify the most relevant tech-based value angles and risks for a target account, so the sales rep can personalize both outbound messaging and the first-call pitch.

Company (domain): {domain}

Firmographic data (source: HG Insights MCP):
{json.dumps(firmo, ensure_ascii=False)}

Technographic data (source: HG Insights MCP):
{json.dumps(techno, ensure_ascii=False)}


Rules:
- Do NOT invent information.
- Use ONLY the data provided above.
- If a data point is missing or unclear, explicitly write "N/A".
- Base every value angle and risk on a specific technology signal from the technographic data.
- These are NOT confirmed buying signals.
- These are hypotheses based on common market behavior for similar stacks.
- If no clear theme can be inferred, write "N/A".
- Keep the output concise, practical, and sales-ready.
- Output must be in Markdown.

Return EXACTLY the following sections and nothing else:

## 1) Company snapshot
- Company name
- Employee size
- Industry
- Headquarters location
- Revenue (if available, otherwise N/A)

## 2) Tech stack highlights
- 6–12 key technologies max
- Group by category when possible (Cloud, CRM, Data, Security, Dev, etc.)
- Only list technologies explicitly present in the data

## 3) Relevant intent themes
Based on the detected technology stack and spending patterns, infer 2–4 likely buying or evaluation themes.

For each theme:
- Theme name
- Why it is relevant given the current tech stack.

Important:
- These are NOT confirmed buying signals.
- These are hypotheses based on common market behavior for similar stacks.
- If no clear theme can be inferred, write "N/A".

## 4) Risks & watchouts
- 3–6 concise bullets
- Each risk must be directly tied to a technology or intent signal
- Examples: stack complexity, migration risk, vendor lock-in, security exposure, tooling overlap

## 5) Sales talking points
Provide exactly 3 talking points.
For each talking point:
- Angle (what to focus on)
- Why now (based on tech or intent signal)
- Question to open the conversation

## 6) Outbound email 
Write a concise and formal outbound email:
- Max 6 lines
- Personalized using the company’s tech stack or intent
- Professional, neutral sales tone
- No hype, no buzzwords

## 7) Data confidence
Briefly state:
- Which key data points were missing or incomplete
- How this impacts confidence in the analysis
- Company-level buying intent signals are not available in this environment; intent themes are inferred from HG Insights taxonomy.
"""
    resp = llm.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# -------- UI --------
st.set_page_config(page_title="First-Touch Sales Assistant", layout="wide")
st.session_state.setdefault("history", [])
st.title("First-Touch Sales Assistant")
st.caption("Know what to say and where to push. Personalized angles and risks for your first sales conversation.")

with st.form("gen"):
    domain_in = st.text_input("Company domain", placeholder="e.g. apple.com")

    categories = st.text_input(
        "Categories (comma)",
        placeholder="Cloud, Data, Security",
        help="Optional. Filters technographics by category (fuzzy match). Leave empty for all categories."
    )

    submitted = st.form_submit_button("Generate first-touch pack", type="primary")

if submitted:
    domain = clean_domain(domain_in)

    if not domain:
        st.error("Please enter a company domain (e.g. apple.com).")
    else:
        try:
            with st.spinner("MCP: firmographics..."):
                firmo_res = run_async(mcp_call("company_firmographic", {"companyDomain": domain}))
                firmo = extract_payload(firmo_res)

            with st.spinner("MCP: technographics..."):
                techno_res = run_async(mcp_call("company_technographic", {
                    "companyDomain": domain,
                    "categories": [c.strip() for c in categories.split(",") if c.strip()]
                }))
                techno = extract_payload(techno_res)

            with st.spinner("LLM: generating pack..."):
                md = make_brief_with_llm(domain, firmo, techno)

            st.session_state.setdefault("history", [])
            st.session_state["history"].append({"domain": domain, "markdown": md})

            st.success("Done")

            sections = split_pack_sections(md)

            tab_pack, tab_talk, tab_email, tab_conf, tab_export, tab_debug = st.tabs(
                ["Pack", "Talking points", "Outbound email", "Data confidence", "Export", "Debug"]
            )

            with tab_pack:
                for k in ["1", "2", "3", "4"]:
                    if sections.get(k):
                        st.markdown(sections[k])
                        st.divider()
                    else:
                        st.info(f"Section {k} missing (N/A)")

            with tab_talk:
                st.markdown(sections.get("5", "N/A"))

            with tab_email:
                st.markdown(sections.get("6", "N/A"))

            with tab_conf:
                st.markdown(sections.get("7", "N/A"))

            with tab_export:
                st.download_button(
                    "Download Markdown",
                    data=md,
                    file_name=f"first_touch_{domain}.md",
                    mime="text/markdown",
                )
                st.text_area("Copy", md, height=400)

            with tab_debug:
                st.subheader("Firmographics (payload)")
                st.json(firmo)
                st.subheader("Technographics (payload)")
                st.json(techno)


        except Exception as e:
            st.error("Error:")
            st.exception(e)

async def list_tools():
    async with streamable_http_client(PHOENIX_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return tools

if st.button("Debug: list MCP tools"):
    tools = run_async(list_tools())
    st.json(tools)

def split_pack_sections(md: str) -> dict:
    """
    Split le markdown du pack en sections:
    ## 1) ... jusqu'à ## 7) ...
    """
    if not md:
        return {}

    pattern = r"(?ms)^##\s*([1-7])\)\s*(.*?)\n(.*?)(?=^##\s*[1-7]\)|\Z)"
    sections = {str(i): "" for i in range(1, 8)}

    for m in re.finditer(pattern, md):
        num = m.group(1)
        title = m.group(2).strip()
        body = m.group(3).strip()
        sections[num] = f"## {num}) {title}\n{body}".strip()

    if all(v == "" for v in sections.values()):
        sections["1"] = md

    return sections