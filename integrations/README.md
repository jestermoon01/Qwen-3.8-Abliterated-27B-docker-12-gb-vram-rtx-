# Connect your agents

This repository runs the model backend, chat UI, compression, and memory.
Install Hermes, ARTEX, or PentAGI separately; their workers and tools are not included.

For an agent running on the host, set:

- OpenAI-compatible base URL: `http://localhost:8001/v1`
- Model: `huihui-qwen3.8-27b`
- API key: no authentication required; use `local-not-required` if a client requires a value.

For another Docker container, attach it to the `qwen-local_default` network
and use `http://gateway:8000/v1`. `localhost` inside an agent container refers
to that container, not the gateway. A different Compose project name changes the network name.

Send `X-User-ID` and `X-Agent-ID` to select persistent memory namespaces.
The gateway also provides `/agents/artex/v1` and `/agents/pentagi/v1` routes
that assign fixed namespaces for clients that cannot configure those headers.
These namespaces organize data; they do not provide access control.

Multiple agents share the same model. The default runs one inference request
at a time; queued work competes for that slot. `profiles/balanced-iq3.env`
allows two active slots with 16K context per request. This does not connect
agent control loops or provide pause/resume/steering of agent sessions.
