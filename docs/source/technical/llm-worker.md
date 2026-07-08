# LLM Screening Worker

The LLM screening worker is a long-lived background process that scans ASReview
projects for queued dispatch records and sends their full-text PDFs to an LLM
(Claude by Anthropic) for automated screening. The LLM reads each PDF against
the project's screening criteria and returns a structured pre-fill payload that
populates the screening form before a human reviewer sees the record.

## How it works

1. **Top-up**: Each project maintains a dispatch buffer. When a reviewer
   requests a new record, the server tops up the `llm_dispatch` table with the
   next highest-ranked, not-yet-labeled records (up to the configured
   `buffer_size`).

2. **Claim**: The worker scans all projects under `ASREVIEW_PATH`, opens each
   project database, and atomically claims the oldest `queued` dispatch rows
   (up to the global `max_concurrent` cap).

3. **Resolve**: For each claimed record, the worker uses the project's
   `PdfResolver` to locate or download the PDF. Records without a resolvable
   PDF are marked `missing_pdf`.

4. **Screen**: The PDF is sent as a base64-encoded document to the Anthropic API
   along with the project's current screening criteria, tags, and lists as the
   system prompt. The LLM responds with a JSON payload describing inclusion
   recommendations, relevant tag values, and list selections.

5. **Store**: On success, the JSON payload is stored in `llm_results` keyed by
   `(record_id, prompt_hash)`, and the dispatch row is marked `ready`. On
   transient failure (rate limit, server error), the worker retries with
   exponential backoff (up to 5 attempts). On permanent failure, the row is
   marked `failed` with the error message.

6. **Pre-fill**: When a reviewer next loads that record, the server checks
   `llm_results` for a cached payload matching the current prompt hash. If
   found, the screening form is pre-populated — the human reviewer still makes
   the final decision.

## Prerequisites

- **Anthropic API key**: Set `ANTHROPIC_API_KEY` in the environment. The worker
  uses the default Anthropic client which reads this variable automatically.
- **ASReview projects**: Projects must exist under `ASREVIEW_PATH` and have
  records with resolvable PDFs (DOIs, URLs, or local attachments).

## Running the worker

### Single project (foreground)

```bash
asreview-llm-worker /path/to/project.asreview
```

Options:
- `--model claude-opus-4-8` — model to use (default from `ASREVIEW_LLM_MODEL`
  env var, or `claude-opus-4-8`)
- `--max-concurrent 3` — max parallel LLM calls within this worker (default
  from `ASREVIEW_LLM_MAX_CONCURRENT` env var, or 3)
- `--poll-interval 5.0` — seconds to sleep when no queued records are found
- `-v` — enable DEBUG logging

### All projects (service mode, foreground)

```bash
asreview-llm-worker
```

Scans all project directories under `ASREVIEW_PATH` and drains their queues
with a single global `ThreadPoolExecutor`. The `max_concurrent` cap applies
across *all* projects combined.

### As a systemd service (NixOS example)

```nix
systemd.services.asreview-llm-worker = {
  description = "ASReview LLM screening worker";
  after = [ "network-online.target" ];
  wants = [ "network-online.target" ];
  environment = {
    ASREVIEW_PATH = "/var/lib/asreview";
    ASREVIEW_LLM_MAX_CONCURRENT = "3";
  };
  serviceConfig = {
    ExecStart = "${pkg}/bin/asreview-llm-worker";
    EnvironmentFile = config.age.secrets.anthropic.path;
    Restart = "always";
    User = "asreview";
  };
};
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `ASREVIEW_PATH` | `~/.asreview` | Root directory containing project folders |
| `ASREVIEW_LLM_MODEL` | `claude-opus-4-8` | Model ID passed to the Anthropic API |
| `ASREVIEW_LLM_MAX_CONCURRENT` | `3` | Global cap on parallel LLM calls |

## What to expect

When the worker starts, it logs:

```
ASReview LLM worker starting (model=claude-opus-4-8, max_concurrent=3)
```

During normal operation, each completed record produces a log line:

```
llm screen record_id=42 status=ready latency=8.35s
```

When no records are queued across any project, the worker sleeps for
`poll_interval` seconds and checks again. This makes the worker safe to run
continuously — it idles harmlessly when there is no work.

### Status lifecycle

Each record in `llm_dispatch` moves through these statuses:

```
queued → in_flight → ready       (success)
                   → failed       (error after retries)
                   → missing_pdf  (no PDF resolvable)
```

- **queued**: Waiting to be picked up by a worker.
- **in_flight**: A worker has claimed this record and is currently processing.
- **ready**: LLM result stored and available for pre-fill.
- **failed**: All retry attempts exhausted. Use the "Re-screen" button in the
  UI to re-queue for another attempt.
- **missing_pdf**: No PDF could be found or downloaded. Use the "Recheck PDF"
  button to try again.

### Prompt hash and auto-reprocessing

Each LLM result is stored under a `prompt_hash` — a hash of the full system
prompt (tags + lists + screening criteria). When an admin edits the screening
criteria text, tags, or lists, the prompt hash changes. All unlabeled dispatch
rows with the old hash are automatically re-queued so the LLM re-screens them
under the new prompt on the next worker cycle.

## Monitoring

- **UI**: The LLM result card on each record's review page shows live status
  (polling every 4 seconds while non-terminal). The criteria panel (lightbulb
  icon in the header) shows the current screening criteria text.
- **Logs**: The worker logs to stdout. Set `-v` for DEBUG-level detail
  including full API request/response tracing.
- **Database**: Query `llm_dispatch` and `llm_results` tables directly to
  inspect queue depth and result history.

## Troubleshooting

**Worker starts but nothing happens:**
Check that projects exist under `ASREVIEW_PATH` and have records. The worker
only processes records that have been dispatched (i.e., reviewers are actively
requesting records in those projects).

**All records show `missing_pdf`:**
Verify that records have DOIs, URLs, or file attachments that the `PdfResolver`
can use. Check Zotero credentials if using Zotero-attached PDFs.

**All records show `failed`:**
Check the `last_error` column in `llm_dispatch` for the specific error. Common
causes: invalid API key, rate limiting (the worker retries automatically),
insufficient API quota, or malformed system prompts.

**Worker is too slow:**
Increase `ASREVIEW_LLM_MAX_CONCURRENT` to allow more parallel API calls. Note
that this increases API costs proportionally. The Anthropic API also has its
own rate limits.
