"""
mcp/tool_executor.py — ReAct tool-calling loop executor.

"""
import logging
from typing import List, Dict, Any, Tuple

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from mcp.tool_definitions import TOOL_MAP
from config.settings import MAX_TOOL_ITERATIONS

logger = logging.getLogger(__name__)


def execute_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Execute a single tool and return string result."""
    tool_fn = TOOL_MAP.get(tool_name)
    if tool_fn is None:
        msg = f"Tool '{tool_name}' not found. Available: {', '.join(TOOL_MAP.keys())}"
        logger.error(f"[ToolExecutor] {msg}")
        return msg
    try:
        logger.info(f"[ToolExecutor] → {tool_name}({tool_args})")
        result = tool_fn.invoke(tool_args)
        result_str = str(result) if not isinstance(result, str) else result
        logger.info(f"[ToolExecutor] ← {tool_name}: {len(result_str)} chars")
        return result_str
    except Exception as e:
        msg = f"Tool '{tool_name}' error: {type(e).__name__}: {e}"
        logger.error(f"[ToolExecutor] {msg}")
        return msg


def run_tool_call_loop(
    llm_with_tools,
    messages: List[BaseMessage],
    max_iterations: int = MAX_TOOL_ITERATIONS,
) -> Tuple[str, List[Dict[str, Any]], List[BaseMessage]]:
    """
    Run the full ReAct tool-calling loop.

    Returns:
        final_text   — LLM's final answer string
        tool_trace   — list of {iteration, tool, args, result, result_len}
        full_messages— complete message history
    """
    tool_trace: List[Dict[str, Any]] = []
    current_messages = list(messages)

    for iteration in range(1, max_iterations + 1):
        logger.info(f"[ToolExecutor] ReAct iteration {iteration}/{max_iterations}")

        try:
            response: AIMessage = llm_with_tools.invoke(current_messages)
        except Exception as e:
            logger.error(f"[ToolExecutor] LLM call failed: {e}")
            return f"LLM call failed: {e}", tool_trace, current_messages

        current_messages.append(response)
        tool_calls = getattr(response, "tool_calls", []) or []

        if not tool_calls:
            # No more tool calls — LLM has a final answer
            logger.info(f"[ToolExecutor] Done after {iteration} iteration(s), {len(tool_trace)} tool calls")
            return response.content or "", tool_trace, current_messages

        # Execute each tool call in this iteration
        for tc in tool_calls:
            tool_name    = tc.get("name", "")
            tool_args    = tc.get("args", {})
            tool_call_id = tc.get("id", f"call_{iteration}_{tool_name}")

            tool_result = execute_tool_call(tool_name, tool_args)

            tool_trace.append({
                "iteration":  iteration,
                "tool":       tool_name,
                "args":       tool_args,
                "result":     tool_result[:500],
                "result_len": len(tool_result),
            })

            current_messages.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call_id,
                name=tool_name,
            ))

    # Safety: max iterations reached — force a final answer
    logger.warning(f"[ToolExecutor] Max iterations ({max_iterations}) reached, forcing final answer")
    try:
        final = llm_with_tools.invoke(current_messages + [{
            "role": "user",
            "content": "Please provide your final answer now based on all the information gathered."
        }])
        return final.content or "", tool_trace, current_messages
    except Exception as e:
        return f"Max iterations reached. Final call failed: {e}", tool_trace, current_messages


def format_tool_trace_for_display(tool_trace: List[Dict[str, Any]]) -> str:
    """Format tool call trace as readable string for logging."""
    if not tool_trace:
        return "No tools called."
    return "\n".join(
        f"  Step {tc['iteration']}: [{tc['tool']}] → {tc['result_len']} chars"
        for tc in tool_trace
    )