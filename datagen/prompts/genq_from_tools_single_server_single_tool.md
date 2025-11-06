## Task
Generate a **fully specified Tool Use Question** for the given MCP server so that the single available tool must be invoked exactly once with concrete arguments.

## Objective
Author a realistic user request that cannot be satisfied without calling the tool described below. The task must be executable immediately—no placeholders, no ambiguity.

## Authoring Checklist
- Choose a real, recognizable target (URL, file path, search query, product name, etc.). Never leave the choice to the assistant.
- Refer to the tool by name and spell out the exact arguments / options it must receive.
- **Property names in the argument object must match the JSON schema shown above exactly**—copy them verbatim.
- Do not invent unsupported arguments or behaviour; rely strictly on what the schema exposes and let the assistant post-process the raw result if necessary.
- Present the request as a numbered mini-checklist (even if a single step) so the workflow is explicit.
- Demand a tangible deliverable (summary, extracted fields, JSON snippet, log output, etc.) that clearly depends on the tool output.
- Ban vague phrasing or “if possible” clauses. Everything must be mandatory and self-contained.

## MCP Server Context
{MCP_SERVER_NAME}: {MCP_SERVER_DESCRIPTION}

Available Tool:
{TOOL_LIST}

## Required XML Output
<response>
  <server_analysis>
    <!-- 2–3 sentences summarising what this tool does and the precise scenario it can unlock. -->
  </server_analysis>
  <target_tools>
    <!-- Include one <tool> element naming the tool exactly. -->
  </target_tools>
  <question>
    <![CDATA[
    Write the user prompt as a numbered checklist that states the exact tool call (with parameters) and the required final deliverable. Example:
    1) Call TOOL_X with {...arguments...} to retrieve ...
    2) Provide <deliverable> summarising ...
    ]]>
  </question>
</response>
