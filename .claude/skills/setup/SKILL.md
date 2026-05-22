---
name: setup
description: Run initial NanoClaw setup. Use when user wants to install NanoClaw, configure it, or go through first-time setup. Triggers on "setup", "install", "configure nanoclaw", or first-time setup requests.
---

# NanoClaw Setup

Tell the user to run `bash nanoclaw.sh` in their terminal. That script handles the full end-to-end setup — dependencies, container image, OneCLI vault, Anthropic credential, service, first agent, and optional channel wiring.

If they hit an error partway through, it will offer Claude-assisted recovery inline — no need to come back here.

## Google Vertex AI (Claude via GCP)

Vertex AI is a third auth method alongside the Claude subscription and Anthropic API key. It does **not** use OneCLI — credentials are read from `.env` and injected by NanoClaw's built-in credential proxy, so OneCLI secret setup is not needed for this mode.

To use it, add to `.env` in the project root:

```
CLAUDE_CODE_USE_VERTEX=1
CLOUD_ML_REGION=<your-gcp-region>
ANTHROPIC_VERTEX_PROJECT_ID=<your-gcp-project-id>
```

Then ensure a GCP credentials file exists at `~/.config/gcloud/application_default_credentials.json` (run `gcloud auth application-default login` if not). For a non-default path, add `GOOGLE_APPLICATION_CREDENTIALS=<path>` to `.env`. Do **not** set `ANTHROPIC_MODEL`.

On restart, the host starts the credential proxy and routes Vertex requests through it; GCP credentials never enter containers. See [docs/SECURITY.md](../../../docs/SECURITY.md) and [docs/SPEC.md](../../../docs/SPEC.md) for details.
