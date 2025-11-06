## Task
Generate a **concrete, step-by-step Tool Use Question** tailored to the provided MCP server. The request must be specific enough that another agent can execute it immediately using the listed tools—no placeholders, no guesswork.

## Objective
Draft a realistic user request that *forces* the assistant to call **exactly {NUM_TOOLS} tools from this server**, in the precise order you specify, to accomplish a meaningful real-world task.

## Authoring Checklist
- Pick a real, plainly identifiable target (e.g., a well-known GitHub repo such as `vercel/next.js`, a public product like “Novatek SecureCam 2”, or an explicit search query such as `"Synology DS923+ price 2025"`). Never leave the assistant to choose the entity.
- Reference the tool names directly and describe *exact input arguments* the assistant must pass (URLs, repo names, query strings, CPU throttle values, etc.).
- Give each tool call its own numbered step, so the expected workflow is unambiguous.
- **Property names in the arguments must match the JSON schema above exactly** (copy them verbatim, including camelCase, nesting, and required arrays).
- Never invent extra arguments or capabilities—if the schema does not expose a filter (e.g., no `topic` field), phrase the instructions so the assistant handles that logic after receiving the raw tool output.
- Demand a final deliverable that synthesises results from all steps (e.g., a structured summary, table, JSON report). Make it clear the answer is impossible without the earlier tool outputs.
- Outlaw vague language (“find something relevant”, “maybe”)—every step must be mandatory and self-contained.
- Avoid TODOs or placeholders such as `<enter query>` or “some repo”. Provide concrete values.

## MCP Server Context
{MCP_SERVER_NAME}: {MCP_SERVER_DESCRIPTION}

Available Tools:
{TOOL_LIST}

## Required XML Output
<response>
  <server_analysis>
    <!-- 2–3 sentences describing how the available tools complement each other for a concrete workflow. -->
  </server_analysis>
  <target_tools>
    <!-- List each tool in the exact order it must be used. Example:
         <tool>read_wiki_structure</tool>
         <tool>read_wiki_contents</tool>
         <tool>ask_question</tool> -->
  </target_tools>
  <question>
    <![CDATA[
    Frame the user request as a multi-step numbered list, each step mapping one-to-one with the target tools and citing the precise arguments to pass. End with the required final deliverable. Example structure:
    1) Call TOOL_A with {...exact parameters...} to do X.
    2) Call TOOL_B with {...} to do Y.
    3) Call TOOL_C with {...} to do Z.
    Produce <specified deliverable>.
    ]]>
  </question>
</response>
