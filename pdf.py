#!/usr/bin/env python3
"""
Proof of concept: send a PDF to Claude, get back the strict JSON your ASReview
screening blueprint will require. Nothing ASReview-specific here yet -- this
isolates the one new link in the chain (PDF -> Claude -> structured verdict) so
you can validate it before touching the fork.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install anthropic
    python poc_pdf_screen.py paper.pdf

Optional:
    python poc_pdf_screen.py paper.pdf --prompt my_screening_prompt.txt
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import anthropic

# ---- The JSON contract -------------------------------------------------------
# In the real blueprint this is BUILT from the project's tags.json / lists.json.
# Here it's hard-coded so you can eyeball whether Claude fills it sensibly.
# Adjust the fields to match one real tag group + one real list from your project.
SYSTEM_PROMPT = """You are screening a full-text medical article for a systematic review.
Read the attached PDF and respond with ONLY a JSON object (no prose, no markdown
fences) in exactly this shape (example response below, although in this example, 
both exclusion_reasons and inclusion_criteria are checked, which doesn't make sense 
-- in a real response, either of those groups should be done, or in other word, 
if not all inclusion criteria are met, then the article should be excluded and the 
exclusion_reasons group should be filled out.
The "issues" field is for any notes you want to leave for the reviewer, e.g. if
you think the article is borderline, or if you think the article is relevant but
the PDF is unreadable, etc. If you have no issues to report, leave it as an empty list.
The most important field is the "case_identifiers" list, which should contain the 
identifiers of all cases in the article that meet the inclusion criteria. 
If no cases meet the inclusion criteria, this list should be empty and the article 
should be excluded with an exclusion criterion. Don't overthink this -- if the article
fulfills the inclusion criteria, then the case identifiers for all cases that mention 
a fatal outcome should be listed, regardless of other details. 

{ 
  "labels": [
    {
      "group": "exclusion_reasons",
      "values": [
        {
          "checked": "no_fatality"
        },
        {
          "checked": "no_infection"
        },
        {
          "checked": "other",
          "text": "Lorem ipsum (max. 50 characters)"
        }
      ]
    },
    {
      "group": "study_type",
      "values": [
        {
          "checked": "case_report",
        }
      ]
    },
    {
      "group": "inclusion_criteria",
      "checked": [
        {
          "checked": "infectious_agent",
        },
        {
          "checked": "mutation",
        },
        {
          "checked": "matchable",
        },
        {
          "checked": "causal_link",
        }
      ]
    }
  ],
  "case_identifiers": [ "Patient 1", "Case A"],
  "issues": [ "Patient 1 died of an infection, but probably not related to the infectious agents mentioned in the article" ],
}



Below are all valid options for the various label groups. You should only check the ones 
that apply to the article you are screening. If the PDF is unreadable or clearly not a 
research article, choose "no_pdf".

[
  {
    "export": "exclusion_reasons",
    "id": 0,
    "label": "Exclusion reasons",
    "require_all": false,
    "required_irrelevant": true,
    "required_relevant": false,
    "single_select": false,
    "values": [
      {
        "export": "no_pdf",
        "id": 0,
        "label": "no pdf"
      },
      {
        "export": "no_fatality",
        "id": 1,
        "label": "no fatality"
      },
      {
        "export": "no_infectious_agent",
        "id": 2,
        "label": "no infectious agent"
      },
      {
        "export": "died_after_immunosuppresive_therapy_eg_hsct",
        "id": 3,
        "label": "died after immunosuppresive therapy (eg. HSCT)"
      },
      {
        "export": "not_humans",
        "id": 4,
        "label": "not humans"
      },
      {
        "export": "theoretical_review",
        "id": 5,
        "label": "theoretical review"
      },
      {
        "export": "lab_study",
        "free_text": false,
        "id": 6,
        "label": "lab study"
      },
      {
        "export": "not_iei",
        "id": 7,
        "label": "not IEI"
      },
      {
        "export": "hiv_g6pd_cf",
        "free_text": true,
        "id": 8,
        "label": "HIV, G6PD, CF"
      },
      {
        "export": "other",
        "id": 9,
        "label": "other"
      }
    ]
  },

  {
    "export": "study_type",
    "id": 2,
    "label": "Study type",
    "require_all": false,
    "required_irrelevant": true,
    "required_relevant": true,
    "single_select": true,
    "values": [
      {
        "export": "case_report",
        "id": 0,
        "label": "Case report"
      },
      {
        "export": "case_report_with_literature_review",
        "id": 1,
        "label": "Case report with literature review"
      },
      {
        "export": "case_series",
        "id": 2,
        "label": "Case series"
      },
      {
        "export": "case_series_with_literature_review",
        "id": 3,
        "label": "Case series with literature review"
      },
      {
        "export": "retrospect_cohort_study",
        "id": 4,
        "label": "Retrospect cohort study"
      },
      {
        "export": "prospective_cohort_study",
        "id": 5,
        "label": "Prospective cohort study"
      },
      {
        "export": "population_study",
        "free_text": true,
        "id": 6,
        "label": "Population study"
      },
      {
        "label": "Other",
        "export": "other",
        "id": 7
      }
    ]
  },
  {
    "export": "inclusion_criteria",
    "id": 3,
    "label": "Inclusion criteria",
    "require_all": true,
    "required_irrelevant": false,
    "required_relevant": true,
    "single_select": false,
    "values": [
      {
        "export": "infectious_agent",
        "id": 0,
        "label": "This article mentions at least one infectious agent."
      },
      {
        "export": "mutation",
        "id": 1,
        "label": "At least one of those cases has a mutation in a gene related to immunity."
      },
      {
        "export": "matchable",
        "id": 2,
        "label": "The data in those cases can be matched to an individual case."
      },
      {
        "export": "causal_link",
        "id": 3,
        "label": "A causal link be found from infection to death in at least one case."
      }
    ]
  }
]




"""


def pdf_to_block(path: Path) -> dict:
    """Turn a PDF file into an Anthropic base64 `document` content block."""
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": data,
        },
        # cache_control lets you re-query the SAME pdf cheaply while iterating on
        # the prompt -- the model caches the (large) document, not the instructions.
        "cache_control": {"type": "ephemeral"},
    }


def screen(pdf_path: Path, system_prompt: str, model: str) -> dict:
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                pdf_to_block(pdf_path),
                {"type": "text", "text": "Screen this article. Respond with JSON only."},
            ],
        }],
    )

    # Claude may return several content blocks; take the text ones.
    text = "".join(b.text for b in resp.content if b.type == "text").strip()

    # Be forgiving about accidental ```json fences, then parse strictly.
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError as e:
        print("!! Claude did not return valid JSON. Raw output:\n", text, file=sys.stderr)
        raise SystemExit(f"JSON parse failed: {e}")

    # Report token usage -- tells you the real per-PDF cost before you batch 1000s.
    print(f"[tokens] input={resp.usage.input_tokens} "
          f"output={resp.usage.output_tokens} "
          f"(cache_read={getattr(resp.usage, 'cache_read_input_tokens', 0)})",
          file=sys.stderr)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--prompt", type=Path, help="file with a custom system prompt")
    ap.add_argument("--model", default="claude-opus-4-6")
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"No such file: {args.pdf}")

    system_prompt = args.prompt.read_text() if args.prompt else SYSTEM_PROMPT
    print(args.pdf)
    #verdict = screen(args.pdf, system_prompt, args.model)
    #print(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()