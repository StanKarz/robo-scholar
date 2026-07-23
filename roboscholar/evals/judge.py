"""LLM-as-judge: Sonnet scores answers vs references (faithfulness + correctness),
returning a Pydantic score object via structured outputs. Include 2-3 few-shot
scored examples in the judge prompt; watch for position/verbosity bias.
"""
