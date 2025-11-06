## Task
Generate a **state-changing Tool Use Question** tailored to the provided MCP server. The instructions must require the assistant to perform real write/update actions with the listed tools (no hypotheticals) and to verify the resulting state.

## Objective
Draft a realistic workflow that *forces* the assistant to call **exactly {NUM_TOOLS} tools from this server**, in the precise order you specify, to create, modify, or otherwise mutate persistent data. Each step must both issue the concrete command and confirm the effect with the next step or in the final deliverable.

## Stateful Scenario Guidance
Follow these guardrails when crafting the task:

{STATEFUL_SCENARIO}

## Authoring Checklist
- Use the tool names verbatim and provide *fully-specified JSON arguments* required by each tool—no placeholders like `<timestamp>` or `someRepo`. If uniqueness is needed, explicitly instruct the assistant to embed the current UTC timestamp (e.g., `ops-sandbox-2025-11-01T15-30Z`) inside the argument value.
- Ensure every numbered step triggers a **persistent change** (creation, update, append, navigation with side effects, etc.) using the exact tool for that step.
- Include any prerequisite read operations (e.g., `get_memory`) only if they are part of the mandated tool list, and make their purpose explicit.
- Require the final deliverable to return a structured JSON summary capturing identifiers returned by the tools (IDs, URLs, confirmation messages) so the mutations can be audited.
- Prohibit vague language (“maybe”, “if possible”)—each step must be mandatory and executable immediately with no additional clarification.

## MCP Server Context
{MCP_SERVER_NAME}: {MCP_SERVER_DESCRIPTION}

Available Tools:
{TOOL_LIST}

## Required XML Output
<response>
  <server_analysis>
    <!-- 2–3 sentences describing how these tools work together to perform irreversible or auditable changes. -->
  </server_analysis>
  <target_tools>
    <!-- List each tool in the exact order it must be used. -->
  </target_tools>
  <question>
    <![CDATA[
    Frame the user request as a numbered checklist where step *i* invokes TOOL_i with fully-specified JSON arguments (matching the schema exactly) that carry out the mutation described in the scenario guidance. End with a requirement that the assistant return a JSON object summarising:
    - Each operation performed (tool name, key arguments, resulting IDs/URLs).
    - Evidence that the state now reflects the requested changes (e.g., latest memory summary, issue URL, DOM confirmation).
    ]]>
  </question>
</response>
