# CN2VI development rules

Before editing, prefer codebase-memory-mcp: index an unindexed checkout, get_architecture,
search_graph and trace_path; read the exact files and use detect_changes after edits.

Local is for code and static inspection. Use CI for automated tests/builds and rented GPU
cloud for environment installation, media benchmarks and model inference. Never download
model weights or run inference on the local development machine.

During an active experimental video run, do not change implementation, model selection,
prompts, thresholds, preprocessing, postprocessing or configuration in response to a discovered
error. Record it and continue whenever technically possible. Quality errors are not fatal.
Produce the most complete playable video and run-report.json/run-report.md. After the run,
explain failures and obtain explicit user approval before corrective changes and a new run.

One task, one model. No model fallback. Every run freezes code/config/model/input provenance,
measures stage and total wall time, load time where practical, RTF, resources and estimated cost.
Unknown API cost or unmeasured quality must remain unknown. Preserve Ngọc Huyền as the fixed
voice until the user explicitly requests a change.

Use Antigravity Gemini 3.8 Flash High only for necessary, bounded delegated work. Keep the
main implementation and verification with the primary agent. Never automatically retry an
agent that returned an empty or failed report.
