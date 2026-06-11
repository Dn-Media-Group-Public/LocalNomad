# LLM Task Capability Matrix

## Performance Key

- **Good** — reliable, production-viable
- **Inconsistent** — works sometimes, fails on edge cases
- **Fails** — not viable for this task

## Matrix

| Task | TinyLlama 1.1B | Phi-3-mini 3.8B | Mistral 7B | Llama 3 8B | Llama 3 70B | GPT-4o / Claude 3.5 |
|---|---|---|---|---|---|---|
| Simple chitchat | OK | Good | Good | Good | Good | Good |
| Basic Q&A (factual) | Inconsistent | Good | Good | Good | Good | Good |
| Summarization (short) | Inconsistent | Good | Good | Good | Good | Good |
| Summarization (long doc) | Fails | Inconsistent | Good | Good | Good | Good |
| Simple instruction follow | Inconsistent | Good | Good | Good | Good | Good |
| Complex instruction follow | Fails | Inconsistent | Good | Good | Good | Good |
| Basic code (snippets) | Inconsistent | Good | Good | Good | Good | Good |
| Code debugging | Fails | Inconsistent | Good | Good | Good | Good |
| Complex code / architecture | Fails | Fails | Inconsistent | Inconsistent | Good | Good |
| Basic math | Fails | Good | Good | Good | Good | Good |
| Multi-step reasoning / logic | Fails | Inconsistent | Inconsistent | Good | Good | Good |
| Structured output (JSON) | Fails | Inconsistent | Good | Good | Good | Good |
| Tool use / function calling | Fails | Fails | Inconsistent | Good | Good | Good |
| Translation (common langs) | Inconsistent | Good | Good | Good | Good | Good |
| Creative writing | Inconsistent | Inconsistent | Good | Good | Good | Good |
| Long context (>4K tokens) | Fails | Fails | Inconsistent | Good | Good | Good |

## Notes

**TinyLlama 1.1B** — useful only for simple conversational filler or demos. 1.1B params is below the threshold where reliable instruction-following emerges. Expect hallucination, topic drift, format failures frequently.

**Phi-3-mini 3.8B** — Microsoft trained it specifically to punch above weight on reasoning/code via high-quality synthetic data. Genuinely competitive with Mistral 7B on reasoning and basic coding despite smaller size. Main weaknesses: context limit (4K), complex agentic tasks, anything requiring broad world knowledge (smaller model = more knowledge gaps).

**Mistral 7B** — solid mid-tier. Good general baseline. Weaker than Llama 3 8B on reasoning.

**Llama 3 8B** — Meta's training quality jump makes this noticeably better than 7B-class predecessors. Reasonable for most single-turn tasks.

**Llama 3 70B / GPT-4o / Claude 3.5** — reliable across nearly all tasks. Differences show on hardest reasoning, long context, and nuanced instruction following.

## Bottom Line

- Use Phi-3-mini for **on-device/browser** tasks where you need more than toy output — basic coding assist, short summarization, simple chat.
- TinyLlama is essentially a **demo/prototype** model — not production-viable for any serious text task.
- For anything requiring reliability at scale, 7B+ server-side models are the practical floor.
