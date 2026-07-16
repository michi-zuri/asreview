# Copyright 2019-2025 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import textwrap

from asreview.webapp._api.llm_worker import run_worker_service

description = """\
Launches the ASReview LLM screening worker: one long-lived process that
scans all projects under ASREVIEW_PATH and drains their LLM queues, with a
global concurrency cap. Each project's Anthropic API key and screening
criteria are read from its stored LLM settings; the ANTHROPIC_API_KEY
environment variable is used as a fallback when no per-project key is set.

The worker is a polling daemon — it never accepts inbound requests as part
of its screening function. When --port is given, a lightweight health check
HTTP server listens on that port (GET / returns 200) so process monitors
can verify the worker is alive.

Example NixOS systemd service:
  systemd.services.asreview-llm-worker = {
    description = "ASReview LLM screening worker";
    after = [ "network-online.target" ];
    wants = [ "network-online.target" ];
    wantedBy = [ "multi-user.target" ];
    environment = {
      ASREVIEW_PATH = "/var/lib/asreview";
      ASREVIEW_LLM_MAX_CONCURRENT = "3";
    };
    serviceConfig = {
      ExecStart = "${pkg}/bin/asreview-llm-worker --port 5603";
      EnvironmentFile = config.age.secrets.anthropic.path; # fallback ANTHROPIC_API_KEY
      Restart = "always";
      User = "asreview";
    };
  };
"""


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(description).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for DEBUG logging.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--port", type=int, default=None,
                        help="Port for the health check HTTP server. "
                             "No server is started if omitted.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    run_worker_service(
        model=args.model,
        max_concurrent=args.max_concurrent,
        poll_interval=args.poll_interval,
        port=args.port,
    )
