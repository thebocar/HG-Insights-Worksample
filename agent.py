import asyncio
import json
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
    Retourne un dict:
    - text: texte concaténé
    - json: dict/list si le texte est du JSON parseable
    - raw: contenu brut (debug)
    """
    items = getattr(result, "content", []) or []
    texts = []
    raw = []

    for c in items:
        raw.append(str(c))
        t = getattr(c, "text", None)
        if t is not None:
            texts.append(t)

    text = "\n".join(texts).strip()

    parsed = None
    if text:
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None

    return {"text": text, "json": parsed, "raw": raw}

def normalize_firmo(firmo_payload: dict) -> dict:
    data = firmo_payload.get("json")

    # Si pas de JSON, on ne peut pas extraire proprement
    if not isinstance(data, (dict,)):
        return {
            "company_name": "N/A",
            "employee_size": "N/A",
            "revenue": "N/A",
            "hq": "N/A",
            "industry": "N/A",
        }

    def pick(*keys, default="N/A"):
        for k in keys:
            v = data.get(k)
            if v not in (None, "", [], {}):
                return v
        return default

    # HQ parfois split en city/state/country
    hq = pick("headquarters", "hq", "headquarters_location", "location", default=None)
    if hq is None:
        city = pick("headquartersCity", "city", default="")
        state = pick("headquartersState", "state", default="")
        country = pick("headquartersCountry", "country", default="")
        parts = [p for p in [city, state, country] if p and p != "N/A"]
        hq = ", ".join(parts) if parts else "N/A"

    return {
        "company_name": pick("companyName", "name", "company_name"),
        "employee_size": pick("employeeSize", "employees", "employee_count", "employee_size"),
        "revenue": pick("revenue", "annualRevenue", "annual_revenue"),
        "hq": hq,
        "industry": pick("industry", "industryName", "primaryIndustry"),
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
            with st.status("Generating first-touch pack...", expanded=False) as status:
                status.write("MCP: firmographics")
                firmo_res = run_async(mcp_call("company_firmographic", {"companyDomain": domain}))
                firmo = extract_payload(firmo_res)

                status.write("MCP: technographics")
                techno_res = run_async(mcp_call("company_technographic", {
                    "companyDomain": domain,
                    "categories": [c.strip() for c in categories.split(",") if c.strip()]
                }))
                techno = extract_payload(techno_res)

                status.write("LLM: generating pack")
                md = make_brief_with_llm(domain, firmo, techno)
    

                st.session_state.setdefault("history", [])
                st.session_state["history"].append({
                    "domain": domain,
                    "markdown": md
                })

                status.update(label="Done", state="complete")

            tab1, tab2, tab3, tab4 = st.tabs(["Pack", "Data", "Export", "Debug"])
            with tab1:
                st.subheader("Company snapshot")

                c1, c2, c3, c4 = st.columns(4)
                firmo_clean = normalize_firmo(firmo)
                c1.metric("Employees", firmo_clean["employee_size"])
                c2.metric("Revenue", firmo_clean["revenue"])
                c3.metric("HQ", firmo_clean["hq"])
                c4.metric("Industry", firmo_clean["industry"])
                st.divider()

            with tab2:
                st.subheader("Firmographics (raw)")
                st.json(firmo)

                st.subheader("Technographics (raw)")
                st.json(techno)

            with tab3:
                st.download_button(
                    "Download Markdown",
                    data=md,
                    file_name=f"first_touch_{domain}.md",
                    mime="text/markdown",
                )
                st.text_area("Copy", md, height=400)
            with tab4:
                st.json({"firmo": firmo, "techno": techno})
                st.write("Firmo json parsed?", firmo["json"] is not None)
                st.text_area("Firmo text", firmo["text"][:3000], height=200)

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
