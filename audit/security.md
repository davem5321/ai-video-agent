# Security Audit Report — AI Video Agent

**Date**: 2026-03-21  
**Auditor**: Security Auditor (Microsoft SDL)  
**Project**: `ai-video-agent` — Automated horoscope video generation pipeline  
**Audit Type**: Manual static analysis (CodeQL CLI not available)  
**Scope**: Full codebase, credentials, dependencies, CI/CD configuration  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Attack Surface Summary](#2-attack-surface-summary)
3. [Findings](#3-findings)
   - [Critical](#critical-findings)
   - [High](#high-findings)
   - [Medium](#medium-findings)
   - [Low](#low-findings)
4. [Positive Security Controls](#4-positive-security-controls)
5. [SDL Checklist Status](#5-sdl-checklist-status)
6. [Recommended Next Steps](#6-recommended-next-steps)

---

## 1. Executive Summary

The `ai-video-agent` project is a Python-based automation pipeline that generates daily horoscope videos by orchestrating OpenAI GPT models (for horoscope text) and Google Vertex AI Veo (for video generation), then burns subtitles onto the output videos using FFmpeg.

### Findings by Severity

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High | 3 |
| Medium | 7 |
| Low | 5 |
| **Total** | **16** |

### Overall Risk Posture

**MEDIUM-HIGH.** The codebase is well-structured and follows several good security practices (no `eval`/`exec`, safe subprocess invocation, use of `.env` for secrets). However, several issues require prompt attention:

- A `.env` file containing live API credentials for both OpenAI and Google Cloud is present on the filesystem. While not committed to git, this represents a credential exposure risk that should be addressed immediately.
- Google API keys are passed as URL query parameters, exposing them in server logs and network captures.
- All seven dependencies are completely unpinned, creating meaningful supply-chain risk.
- No CI/CD pipeline exists, leaving the project with no automated secret scanning, dependency auditing, or linting checks.

A threat model has not been separately documented; this report functions as the primary security analysis artifact.

---

## 2. Attack Surface Summary

| Component | Role | Exposed Surface |
|-----------|------|-----------------|
| `src/write/horoscope_writer.py` | OpenAI API client — generates horoscope text | OpenAI API key; API response errors printed to stdout |
| `src/render/veo_horoscope_pipeline.py` | Google Vertex AI Veo client + pipeline orchestrator | Google API key in URL query params; video download URL from API; CLI `--out` path; error response logging |
| `src/render/subtitle_utils.py` | FFmpeg integration for subtitle burning | Subprocess invocation; temp file creation; ASS format injection surface |
| `reference/app.py` | Streamlit web UI (reference/development use) | User-uploaded video files; user-supplied caption text; font file path input; FFmpeg stderr in UI |
| `.env` | Runtime credential store | OpenAI API key; Google API key; Google Project ID |
| `requirements.txt` / `pyproject.toml` | Dependency manifest | Unpinned supply-chain dependencies |
| `.github/workflows/` | CI/CD configuration | **Empty** — no automated security controls |
| `out_real/`, `data/horoscopes/` | Committed output files | Prompt templates and operational data exposed in git history |

---

## 3. Findings

---

### Critical Findings

---

#### [CRITICAL] — Live API Credentials Present on Local Filesystem

- **SDL Phase**: Implementation / Release
- **File**: `.env` (local filesystem, not committed to git)
- **STRIDE Category**: Information Disclosure / Elevation of Privilege
- **Description**:  
  The `.env` file on the local filesystem contains live, production-grade API credentials:
  - **OpenAI API Key**: `sk-proj-pc78yDu...` (full key visible, ~170 chars)
  - **Google API Key**: `AQ.Ab8RN...` (full key visible)
  - **Google Cloud Project ID**: `veo-3-videos-482006`

  While the `.gitignore` correctly excludes `.env` from git tracking, the file is present on disk in plaintext. This means:
  1. Any process running under the same user account can read the file.
  2. Backup systems, cloud sync (OneDrive, Dropbox), IDE telemetry, crash dump utilities, or container image snapshots may inadvertently capture and transmit it.
  3. If the developer shares their machine, accidental `git add .` after a `.gitignore` modification, or repository export, the keys would be immediately exposed.
  4. The keys appear to be real and active, not test/sandbox credentials.

- **Attack Scenario**:  
  A malicious process or compromised IDE extension reads `.env` from the project directory and exfiltrates both API keys. With the OpenAI key, an attacker can run AI workloads billed to the developer. With the Google API key and Project ID, they can generate Vertex AI videos and incur significant cloud compute charges, or access other Google Cloud APIs enabled on project `veo-3-videos-482006`.

- **Recommendation**:
  1. **Immediately rotate** both the OpenAI API key and the Google API key — treat them as compromised.
  2. Store credentials in an OS-level secrets store (Windows Credential Manager, macOS Keychain) or a dedicated secrets manager (Azure Key Vault, HashiCorp Vault, Google Secret Manager) rather than a plaintext file.
  3. For development, use short-lived credentials or restrict API key permissions to only the specific services and operations needed (principle of least privilege).
  4. Add a pre-commit hook using `detect-secrets` or `gitleaks` to prevent accidental future commits of secrets.
  5. Add a CI/CD secret-scanning step (e.g., GitHub Advanced Security, `truffleHog`) to scan on every push.
  6. Consider adding `.env` to the repository's `.gitignore` with a template enforcement check to ensure `.env` is never staged.

- **Source**: Manual Review

---

### High Findings

---

#### [HIGH] — Google API Key Exposed in HTTP URL Query Parameters

- **SDL Phase**: Implementation
- **File**: `src/render/veo_horoscope_pipeline.py` (lines 133–136, 201–204)
- **STRIDE Category**: Information Disclosure
- **Description**:  
  The `VeoClient` constructs Vertex AI API URLs with the API key appended as a query parameter:
  ```python
  url = (f"https://{self.region}-aiplatform.googleapis.com/v1/"
         f"projects/{self.project_id}/locations/{self.region}/"
         f"publishers/google/models/{self.model_id}:predictLongRunning"
         f"?key={self.api_key}")
  ```
  This pattern repeats for the `fetchPredictOperation` endpoint (lines 201–204). Placing credentials in the URL is insecure because:
  - The full URL is recorded in web server access logs, proxy logs, load balancer logs, and CDN logs.
  - The URL may appear in `Referer` headers for subsequent requests.
  - HTTP client libraries (including `requests`) may log URLs during debugging or exception tracebacks.
  - The URL appears in network packet captures and browser history if accessed via a web context.

- **Attack Scenario**:  
  An infrastructure operator with access to network proxy logs, or any log aggregation system receiving stdout from the application, could extract the Google API key from logged request URLs and use it to submit unauthorized Vertex AI video generation jobs on the developer's account.

- **Recommendation**:
  1. Pass the API key in the `Authorization` header instead:
     ```python
     headers = {
         "Content-Type": "application/json",
         "Authorization": f"Bearer {self.api_key}"
     }
     # Remove ?key= from URL
     url = (f"https://{self.region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.region}/"
            f"publishers/google/models/{self.model_id}:predictLongRunning")
     r = requests.post(url, json=payload, headers=headers, timeout=60)
     ```
  2. For Google Cloud APIs, prefer [Application Default Credentials (ADC)](https://cloud.google.com/docs/authentication/application-default-credentials) or a service account with scoped IAM permissions rather than an API key. This avoids key management entirely.
  3. If API key usage is retained, restrict the Google API key in the Google Cloud Console to only the Vertex AI API and to specific IP addresses or referrers where feasible.

- **Source**: Manual Review

---

#### [HIGH] — Unpinned Dependencies — Supply Chain Risk

- **SDL Phase**: Requirements / Verification
- **File**: `requirements.txt` (all 7 lines), `pyproject.toml` (`[project].dependencies`)
- **STRIDE Category**: Tampering / Elevation of Privilege
- **Description**:  
  Every dependency in both `requirements.txt` and `pyproject.toml` is completely unpinned (no version specifier at all):
  ```
  moviepy
  python-dotenv
  google-api-python-client
  google-auth-httplib2
  google-auth-oauthlib
  openai
  requests
  ```
  This means:
  1. `pip install -r requirements.txt` will always pull the **latest available** version of each package.
  2. If any package is compromised (typosquatting, maintainer account takeover, dependency confusion), or if a future version of a dependency introduces a vulnerability or breaking change, it will be silently installed.
  3. There is no lockfile (`pip freeze` output or `poetry.lock`) to ensure reproducible builds.
  4. The `moviepy` library in particular has historically bundled other packages and had known CVEs.
  5. `google-api-python-client` is a wide surface-area library with transitive dependencies that have had past CVEs.

- **Attack Scenario**:  
  An attacker publishes a malicious version of `moviepy` (e.g., via a compromised PyPI account). Any developer running `pip install -r requirements.txt` will install the malicious version, which may exfiltrate the `.env` file or execute arbitrary code during installation via a `setup.py` hook.

- **Recommendation**:
  1. Pin all dependencies to specific versions using `pip freeze > requirements-lock.txt` and commit the lockfile.
  2. Separate pinned runtime dependencies from the loose version specs in `pyproject.toml` (keep loose specs there for library compatibility) and use `requirements.txt` as the pinned lockfile.
  3. Run `pip audit` (or `safety check`) as part of the development workflow to identify known CVEs in current dependencies.
  4. Add a GitHub Actions workflow that runs `pip audit` on every pull request.
  5. Consider migrating to `poetry` or `pipenv` for integrated lockfile management.

- **Source**: Manual Review

---

#### [HIGH] — Unvalidated URL Downloaded from API Response (Partial SSRF)

- **SDL Phase**: Implementation
- **File**: `src/render/veo_horoscope_pipeline.py` (lines 258–264)
- **STRIDE Category**: Spoofing / Information Disclosure
- **Description**:  
  After polling for video generation completion, the code extracts a URL from the Google API JSON response and downloads it:
  ```python
  video_url = video.get("gcsUri") or video.get("bytesBase64Encoded")
  ...
  elif video_url.startswith("http"):
      with requests.get(video_url, stream=True) as resp:
          resp.raise_for_status()
          with open(video_path_raw, "wb") as f:
              for chunk in resp.iter_content(chunk_size=8192):
                  f.write(chunk)
  ```
  Issues:
  1. The URL check `video_url.startswith("http")` matches both `https://` and `http://`. A response containing an `http://` URL would be downloaded without TLS, exposing the video content to interception and enabling a downgrade attack.
  2. There is no validation that the URL points to a trusted domain (e.g., `storage.googleapis.com`, `*.googleusercontent.com`). If a response contained an `http://` URL pointing to an internal network address (e.g., `http://169.254.169.254/` — the AWS/GCP metadata service), the application would attempt to fetch it.
  3. There is no file size limit on the download, making it possible for an oversized or infinite response to exhaust disk space or memory.

- **Attack Scenario**:  
  If a network attacker can intercept the HTTPS response from Google's API (e.g., via a compromised corporate proxy with TLS inspection), they could inject an `http://` video URL pointing to an internal network service or a very large file, causing the application to exfiltrate internal data to disk or exhaust disk space.

- **Recommendation**:
  1. Validate that the URL uses HTTPS: `if video_url.startswith("https://")`.
  2. Validate the URL hostname against an allowlist of expected Google domains:
     ```python
     from urllib.parse import urlparse
     ALLOWED_VIDEO_HOSTS = {"storage.googleapis.com", "googleusercontent.com"}
     parsed = urlparse(video_url)
     if parsed.scheme != "https" or not any(parsed.hostname.endswith(h) for h in ALLOWED_VIDEO_HOSTS):
         raise ValueError(f"Unexpected video URL host: {parsed.hostname}")
     ```
  3. Add a `stream=True` maximum response size check during download (track bytes written and abort if over a threshold, e.g., 500 MB).

- **Source**: Manual Review

---

### Medium Findings

---

#### [MEDIUM] — Sensitive API Error Responses Printed to Stdout

- **SDL Phase**: Implementation
- **File**: `src/render/veo_horoscope_pipeline.py` (lines 182–187); `src/write/horoscope_writer.py` (lines 178–181)
- **STRIDE Category**: Information Disclosure
- **Description**:  
  When API requests fail, the application prints full error details to stdout, including raw API response bodies:
  ```python
  # veo_horoscope_pipeline.py
  error_data = e.response.json()
  print(f"   Error details: {error_data}")
  # (bare except fallback:)
  print(f"   Response text: {e.response.text}")

  # horoscope_writer.py
  print(f"       ERROR: {str(e)}")
  ```
  API error responses may contain:
  - Internal Google Cloud service error messages with infrastructure details.
  - Quota or billing account identifiers.
  - OpenAI API error messages with model version or account-level details.
  - Partial request payloads that are echoed back in error responses.

  Additionally, the bare `except:` block at line 185 catches all exceptions including `AttributeError`, meaning it will attempt to print `e.response.text` even if `e.response` is `None`, causing a secondary unhandled exception in error paths.

- **Attack Scenario**:  
  If this application is deployed as a service or its stdout is captured in a logging system (e.g., CloudWatch, Splunk), an attacker who can generate API errors (or who has read access to logs) can extract internal infrastructure details from the error messages.

- **Recommendation**:
  1. Log error details at a `DEBUG` level to a secure, restricted log target rather than `print()` to stdout.
  2. Sanitize error messages before display — never print raw API response bodies in production.
  3. Fix the bare `except:` clause: use `except Exception as err:` and check `if hasattr(e, 'response') and e.response is not None`.
  4. Add a logging configuration that redacts known sensitive patterns (API keys, project IDs) from log output.

- **Source**: Manual Review

---

#### [MEDIUM] — Unvalidated CLI Output Path (Potential Directory Traversal)

- **SDL Phase**: Implementation
- **File**: `src/render/veo_horoscope_pipeline.py` (line 708)
- **STRIDE Category**: Tampering
- **Description**:  
  The `--out` CLI argument is used directly without validation:
  ```python
  out_dir = Path(args.out)
  ...
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "prompts").mkdir(exist_ok=True)
  (out_dir / "renders").mkdir(exist_ok=True)
  ```
  An operator or script caller can pass any path, including:
  - Absolute paths outside the project directory: `--out /etc/cron.d`
  - Parent-traversal paths: `--out ../../sensitive_dir`
  - System directories: `--out C:\Windows\System32`

  While this is a CLI tool run by the developer, if it is ever wrapped in a web interface, scheduled task runner, or Makefile with external inputs, this becomes a write-anywhere vulnerability. The pipeline writes prompt `.txt` files, `manifest.json`, and video `.mp4` files to this directory.

- **Attack Scenario**:  
  A crafted Makefile target or CI/CD integration that passes a user-controlled `--out` value could cause the pipeline to write files (e.g., `manifest.json`) to sensitive system directories, overwriting existing files.

- **Recommendation**:
  1. Validate the output path is within an expected base directory:
     ```python
     import os
     base_dir = Path(__file__).parent.parent.parent.resolve()
     out_dir = Path(args.out).resolve()
     if not str(out_dir).startswith(str(base_dir)):
         raise ValueError(f"Output path must be within project directory: {out_dir}")
     ```
  2. Alternatively, restrict to relative paths only: reject any `args.out` that is an absolute path.

- **Source**: Manual Review

---

#### [MEDIUM] — Committed Output Files and Data Despite `.gitignore` Rules

- **SDL Phase**: Release
- **File**: `out_real/manifest.json`, `out_real/prompts/*.txt`, `data/horoscopes/2025-09-23/*` (all tracked in git)
- **STRIDE Category**: Information Disclosure
- **Description**:  
  The `.gitignore` correctly lists `out_real/` and `data/` for exclusion, but these files are already tracked in git history (they were committed before the `.gitignore` rules were added, or after a `git add -f`). This means:
  - **`out_real/prompts/*.txt`**: Contains full Veo video generation prompt templates including the precise wording used for video generation, the zodiac sign substitution pattern, and the project's stylistic approach — this is proprietary operational data.
  - **`out_real/manifest.json`**: Contains job IDs, video file paths, and run configurations.
  - **`data/horoscopes/2025-09-23/*.txt`**: Contains horoscope content generated by the OpenAI API (placeholder content in this case, but the mechanism is the same for real runs).
  - **`data/horoscopes/.DS_Store`**: macOS metadata file committed, revealing the developer's macOS environment and directory structure.

- **Attack Scenario**:  
  Any person who clones the repository (if it is ever made public, or is shared with collaborators) will have access to the operational prompt templates and can replicate the exact video generation approach without authorization.

- **Recommendation**:
  1. Remove the committed files from git tracking (they remain on disk but are no longer in git history):
     ```bash
     git rm --cached out_real/manifest.json
     git rm --cached -r out_real/prompts/
     git rm --cached -r data/horoscopes/
     git commit -m "Remove accidentally tracked output files"
     ```
  2. To fully purge from git history, use `git filter-repo` or BFG Repo Cleaner.
  3. Add `.DS_Store` to `.gitignore`.
  4. Consider adding a `pre-commit` hook that prevents committing files to `out*/` or `data/` directories.

- **Source**: Manual Review

---

#### [MEDIUM] — ASS Subtitle Injection via Unsanitized Caption Text

- **SDL Phase**: Implementation
- **File**: `src/render/subtitle_utils.py` (lines 875, 910)
- **STRIDE Category**: Tampering
- **Description**:  
  Caption text (sourced from OpenAI API output) is written directly into ASS subtitle format with only newline sanitization:
  ```python
  cleaned_text = text.replace('\r\n', '\\N').replace('\r', '\\N').replace('\n', '\\N')
  ...
  Dialogue: 0,...,,{animation}{cleaned_text}
  ```
  ASS subtitle format uses `{...}` tags for inline style overrides (e.g., `{\pos(0,0)}`, `{\c&H0000FF&}`, `{\an8}`). If the OpenAI API response or horoscope text contains curly braces with ASS control codes — whether by coincidence or through a prompt injection attack — these would be passed verbatim to FFmpeg's ASS renderer.

  While current content (horoscopes) is low-risk, this is a generic text processing pipeline. A future change that passes less controlled text through this path could result in unintended subtitle rendering behavior.

  Additionally, the `fontname` parameter (sourced from `os.path.splitext(os.path.basename(fontfile))[0]` in `reference/app.py`) is written directly into the ASS `Style` line without sanitization and could contain commas or special characters that break the ASS format.

- **Attack Scenario**:  
  An attacker who can influence the horoscope text (e.g., via a prompt injection attack targeting the OpenAI API call) could inject ASS tags into the caption text. For example, injecting `{\pos(960,540)\bord0\shad0\c&H00000000&}` could make all subtitle text invisible, or `{\r}` could reset all styling.

- **Recommendation**:
  1. Strip or escape ASS control code delimiters from caption text:
     ```python
     import re
     cleaned_text = re.sub(r'\{[^}]*\}', '', text)  # Remove ASS tags
     cleaned_text = cleaned_text.replace('\r\n', '\\N').replace('\r', '\\N').replace('\n', '\\N')
     ```
  2. Validate `fontname` contains only safe characters (alphanumeric, spaces, hyphens) before writing to the ASS style header.

- **Source**: Manual Review

---

#### [MEDIUM] — Temporary ASS Files Created Without Restrictive Permissions

- **SDL Phase**: Implementation
- **File**: `src/render/subtitle_utils.py` (lines 920–922)
- **STRIDE Category**: Information Disclosure / Tampering
- **Description**:  
  The function creates temporary ASS subtitle files using `tempfile.mkstemp()`:
  ```python
  fd, ass_path = tempfile.mkstemp(suffix=".ass", text=True)
  with os.fdopen(fd, 'w', encoding='utf-8', newline='\r\n') as f:
      f.write(ass_content)
  ```
  On Windows, `tempfile.mkstemp()` creates files with permissions that may be readable by other processes under the same user session. The file contains the full caption text and ASS format metadata, and its path is passed to FFmpeg. If another process modifies the temp file between creation and FFmpeg's read, it could alter the caption content in the output video.

  Additionally, the cleanup code silently swallows cleanup failures:
  ```python
  try:
      os.unlink(ass_path)
  except:
      pass
  ```
  This bare `except: pass` pattern hides errors and may leave sensitive temp files on disk if cleanup fails.

- **Attack Scenario**:  
  On a multi-user system or in a containerized environment where multiple pipeline instances run concurrently, a race condition between ASS file creation and FFmpeg reading it could allow process A to tamper with process B's subtitle content.

- **Recommendation**:
  1. Replace `except: pass` with `except OSError as e: logger.warning("Failed to clean up temp file %s: %s", ass_path, e)` to at least log cleanup failures.
  2. Use a `try/finally` block to guarantee cleanup even on FFmpeg failure.
  3. Consider using Python's `tempfile.TemporaryDirectory()` context manager to guarantee cleanup.

- **Source**: Manual Review

---

#### [MEDIUM] — Bare `except:` Clauses Swallowing All Exceptions

- **SDL Phase**: Implementation
- **File**: `src/render/veo_horoscope_pipeline.py` (line 185); `src/render/subtitle_utils.py` (lines 1058–1060)
- **STRIDE Category**: Tampering (reliability/integrity)
- **Description**:  
  The codebase uses bare `except:` clauses in two places:
  ```python
  # veo_horoscope_pipeline.py
  except:
      print(f"   Response text: {e.response.text}")
  
  # subtitle_utils.py
  try:
      os.unlink(ass_path)
  except:
      pass
  ```
  Bare `except:` catches `SystemExit`, `KeyboardInterrupt`, `GeneratorExit`, and all other base exceptions in addition to `Exception`. This is an anti-pattern that:
  - Prevents `Ctrl+C` from working properly during execution.
  - Hides programming errors (e.g., `AttributeError`, `NameError`) that should surface as bugs.
  - In the `veo_horoscope_pipeline.py` case, `e` is the outer `requests.exceptions.RequestException` but the bare `except:` catches a *different* exception from the `e.response.json()` call — so `e.response.text` might reference the original exception variable, creating confusing error messages.

- **Recommendation**:
  1. Replace all bare `except:` with `except Exception as e:` at minimum.
  2. For the cleanup case, use `except OSError:` to catch only file deletion errors.
  3. Never use bare `except:` — add a linting rule (e.g., `flake8 E722`) to enforce this.

- **Source**: Manual Review

---

#### [MEDIUM] — No Rate Limiting or API Cost Cap Protection

- **SDL Phase**: Design
- **File**: `src/write/horoscope_writer.py` (lines 1153–1184); `src/render/veo_horoscope_pipeline.py` (lines 555–562)
- **STRIDE Category**: Denial of Service
- **Description**:  
  The pipeline makes up to 12 sequential OpenAI API calls (one per zodiac sign) and up to 12 sequential Veo video generation job submissions with no per-run cost ceiling or circuit breaker. The only rate limiting is a `time.sleep(0.3)` between OpenAI calls.

  Issues:
  1. If called in a loop (e.g., a bug in a wrapper script), the pipeline will continue submitting expensive API calls without any budgetary check.
  2. Veo video generation is charged per video second generated; 12 × 8-second videos at 1080p could cost hundreds of dollars per run.
  3. There is no check of remaining API quota before beginning a batch run.
  4. The `--signs` argument provides some mitigation but is optional and not enforced by default.

- **Attack Scenario**:  
  A misconfigured cron job or CI/CD pipeline trigger calls the script in a tight loop, exhausting OpenAI and Google Cloud API quotas and generating large unexpected bills before the developer notices.

- **Recommendation**:
  1. Add a `--max-cost` or `--dry-run` flag that estimates and displays API costs before executing.
  2. Implement a configurable run limit (e.g., `MAX_DAILY_RUNS=1` via env var) checked against a local counter file.
  3. Set spending limits in the Google Cloud Console and OpenAI platform billing settings.
  4. Add a pre-flight check that verifies available quota before starting a batch run.

- **Source**: Manual Review

---

### Low Findings

---

#### [LOW] — Author Email (PII) Committed in `pyproject.toml`

- **SDL Phase**: Release
- **File**: `pyproject.toml` (line 12)
- **STRIDE Category**: Information Disclosure
- **Description**:  
  The developer's corporate email address (`davidmo@microsoft.com`) is hardcoded in the package metadata:
  ```toml
  authors = [
      {name = "David Moore", email = "davidmo@microsoft.com"}
  ]
  ```
  If this repository is ever made public or shared externally, this email address will be exposed in the git history and any published package on PyPI.

- **Recommendation**:
  1. Use a personal or role-based email if the package is intended for public release.
  2. If this is an internal-only tool, ensure the repository access controls are appropriately restricted.

- **Source**: Manual Review

---

#### [LOW] — Reference Streamlit App Exposes FFmpeg stderr in UI

- **SDL Phase**: Implementation
- **File**: `reference/app.py` (lines 141, 169–170)
- **STRIDE Category**: Information Disclosure
- **Description**:  
  The reference Streamlit application outputs raw FFmpeg stderr to the UI in two places:
  ```python
  # On failure:
  st.code(result.stderr)
  # On success:
  st.code("\n".join(result.stderr.splitlines()[-50:]))
  ```
  FFmpeg's stderr output contains:
  - Full file system paths to input and output files (including the temp directory path).
  - System codec version information.
  - Detailed encoding parameters.

  While this is a `reference/` utility and not the production pipeline, its code patterns may be copied into production.

- **Recommendation**:
  1. On success, show only a brief success message, not raw FFmpeg output.
  2. On failure, show a sanitized error message to the user and log full FFmpeg stderr to a server-side log file.
  3. Apply the same principle in `subtitle_utils.py`: the `result.stderr` on failure is returned as a string from `add_caption_to_video()` — callers should log it securely, not display it to end users.

- **Source**: Manual Review

---

#### [LOW] — Reference Streamlit App Allows Arbitrary Font File Path Input

- **SDL Phase**: Implementation
- **File**: `reference/app.py` (lines 59, 103–104)
- **STRIDE Category**: Information Disclosure
- **Description**:  
  The Streamlit reference UI accepts a user-supplied font file path:
  ```python
  fontfile = st.text_input("Font file path (.ttf)", value=config.DEFAULT_FONT)
  ...
  if not os.path.exists(fontfile):
      st.error("Font file not found. Please provide a valid path to a .ttf font.")
  ```
  The `os.path.exists()` call acts as a filesystem oracle — a user can probe for the existence of any file on the server by entering arbitrary paths and observing whether the error message appears. This is an information disclosure issue if the Streamlit app is hosted on a shared server.

- **Recommendation**:
  1. Restrict font file paths to a pre-approved list or a specific fonts directory.
  2. If user-supplied font files are required, use a file-upload widget rather than a path input, and validate the MIME type of the uploaded font.

- **Source**: Manual Review

---

#### [LOW] — No CI/CD Pipeline and No Automated Security Controls

- **SDL Phase**: Verification / Release
- **File**: `.github/workflows/` (empty directory)
- **STRIDE Category**: (Process Gap)
- **Description**:  
  The `.github/workflows/` directory is empty. There are no automated workflows for:
  - Secret scanning (e.g., GitHub Advanced Security, `gitleaks`, `detect-secrets`).
  - Dependency vulnerability scanning (`pip audit`, `safety`).
  - Linting/SAST (e.g., `flake8`, `bandit`, `semgrep`).
  - Unit or integration testing (`pytest`).

  The `Makefile` `test` target prints "No tests configured yet." Without any automated gates, security regressions and new vulnerabilities introduced by dependency updates or code changes will not be detected before deployment.

- **Recommendation**:
  1. Add a GitHub Actions workflow with at minimum:
     - `pip audit` on every push and PR to scan for known CVEs.
     - `bandit -r src/` for Python SAST (it would flag the bare `except:` clauses and subprocess calls).
     - `detect-secrets scan` or `gitleaks` to prevent credential commits.
  2. Add `flake8` or `ruff` linting with rules E722 (bare `except`) and W605 (invalid escape sequences).
  3. Add a `pytest` test suite with at least happy-path and error-path tests for `horoscope_writer.py` and `subtitle_utils.py`.

- **Source**: Manual Review

---

#### [LOW] — No Test Suite

- **SDL Phase**: Verification
- **File**: (no `tests/` directory exists)
- **STRIDE Category**: (Process Gap)
- **Description**:  
  The project has no automated tests. The `pyproject.toml` specifies `testpaths = ["tests"]` but the `tests/` directory does not exist. The `Makefile` test target explicitly states "No tests configured yet."

  Without tests:
  - Security-relevant changes (e.g., changes to URL construction, subprocess arguments, or credential loading) cannot be regression-tested.
  - Dependency version bumps may silently introduce behavioral changes in API client code.
  - The `subtitle_utils.py` ASS injection sanitization (once added) cannot be verified automatically.

- **Recommendation**:
  1. Create `tests/test_subtitle_utils.py` with unit tests for `create_ass_file()`, `color_to_ass()`, and `build_ass_filter()`.
  2. Create `tests/test_horoscope_writer.py` with mocked OpenAI calls to test error handling paths.
  3. Add a `tests/test_pipeline.py` with mocked `requests` calls to test the `VeoClient` URL construction and response parsing.

- **Source**: Manual Review

---

## 4. Positive Security Controls

The following security practices are already in place and should be maintained:

| Control | Location | Notes |
|---------|----------|-------|
| ✅ `.env` excluded from git | `.gitignore` line 4 | Correctly prevents credential commits |
| ✅ Subprocess uses list form (no `shell=True`) | `subtitle_utils.py` line 966–984 | FFmpeg called with `cmd` list, preventing shell injection |
| ✅ HTTP requests use `timeout` parameter | `veo_horoscope_pipeline.py` lines 169, 214 | 60s and 30s timeouts set, preventing indefinite hangs |
| ✅ `raise_for_status()` called on API responses | `veo_horoscope_pipeline.py` lines 170, 220 | HTTP errors surface as exceptions |
| ✅ No `eval()`, `exec()`, or `pickle` usage | All source files | No code execution from untrusted input |
| ✅ No SQL or database access | All source files | No SQL injection surface |
| ✅ Streaming download with `iter_content()` | `veo_horoscope_pipeline.py` line 263 | Prevents loading entire video into memory |
| ✅ Type hints on functions | Multiple files | Improves code correctness and static analysis accuracy |
| ✅ HTTPS used for all API calls | `veo_horoscope_pipeline.py` | URLs use `https://` scheme |
| ✅ Zodiac sign names validated by CLI `choices` | `veo_horoscope_pipeline.py` lines 625–635 | Prevents invalid model/aspect/resolution values |
| ✅ File writes use `encoding="utf-8"` | Multiple files | Consistent encoding handling |

---

## 5. SDL Checklist Status

| SDL Requirement | Status | Notes |
|----------------|--------|-------|
| Secrets managed outside source code | ⚠️ PARTIAL | `.env` not in git, but plaintext on disk with live keys |
| API keys passed securely (headers not URLs) | ❌ FAIL | Google API key in URL query params |
| Dependencies pinned and audited | ❌ FAIL | All deps unpinned, no audit tooling |
| No banned functions (`eval`, `exec`, `pickle`) | ✅ PASS | None found |
| Subprocess calls without `shell=True` | ✅ PASS | FFmpeg uses list form |
| TLS enabled for all external HTTP calls | ✅ PASS | HTTPS used; NOTE: video download accepts `http://` |
| Input validation on CLI/external inputs | ⚠️ PARTIAL | `--out` path unvalidated; model/aspect use `choices` |
| Error messages do not leak sensitive data | ❌ FAIL | Raw API error responses printed to stdout |
| Temporary files cleaned up securely | ⚠️ PARTIAL | Cleanup exists but errors silently swallowed |
| No hardcoded credentials in source code | ✅ PASS | No keys in `.py` files |
| Automated secret scanning in CI/CD | ❌ FAIL | No CI/CD pipeline |
| Automated dependency vulnerability scanning | ❌ FAIL | No CI/CD pipeline |
| SAST tooling applied | ❌ FAIL | No CI/CD pipeline, no `bandit` or `semgrep` |
| Test coverage for security-sensitive code | ❌ FAIL | No test suite |
| Output files excluded from version control | ⚠️ PARTIAL | `.gitignore` rules exist but files already tracked |
| Least-privilege API credentials | ⚠️ UNKNOWN | API key scoping unknown; no service account used |

---

## 6. Recommended Next Steps

The following actions are prioritized by severity and implementation effort:

### Priority 1 (Immediate — within 24 hours)
**Rotate API Credentials**  
The `.env` file contains what appear to be live API keys for OpenAI (`sk-proj-...`) and Google Cloud (`AQ.Ab8RN6...`). These should be rotated immediately via the respective provider dashboards regardless of whether they have been misused. This takes less than 5 minutes per key.
- OpenAI: https://platform.openai.com/api-keys
- Google Cloud: https://console.cloud.google.com/apis/credentials

### Priority 2 (This Sprint — within 1 week)
**Fix API Key Transmission and Add Basic CI/CD**  
Replace the `?key={self.api_key}` URL query parameter pattern with an `Authorization` header in `VeoClient.submit()` and `VeoClient.poll_until_done()`. This is a 5-line change with high security impact. Simultaneously, add a minimal GitHub Actions workflow (`pip audit` + `detect-secrets`) to prevent regression.

### Priority 3 (This Sprint — within 1 week)
**Pin All Dependencies**  
Run `pip freeze > requirements-lock.txt` in the current working environment and commit the lockfile. This is a one-command fix that eliminates the entire class of supply-chain risk from unpinned dependencies.

### Priority 4 (Next Sprint — within 2 weeks)
**Remove Tracked Output Files and Add Input Validation**  
Use `git rm --cached` to stop tracking `out_real/` and `data/` files (which are already in `.gitignore`). Add output path validation to the `--out` CLI argument. Sanitize ASS control codes from caption text. Fix all bare `except:` clauses.

### Priority 5 (Backlog — within 1 month)
**Establish Test Suite and Secure Logging**  
Add a `pytest` test suite covering `subtitle_utils.py` and the `VeoClient` URL construction and response parsing. Replace `print()` error output with a proper `logging` configuration that sanitizes sensitive values. Add a URL allowlist for video downloads.

---

## Appendix: Files Reviewed

| File | Lines | Status |
|------|-------|--------|
| `src/main.py` | 5 | Reviewed — minimal bootstrap |
| `src/render/veo_horoscope_pipeline.py` | 744 | Reviewed — multiple findings |
| `src/render/subtitle_utils.py` | ~300 | Reviewed — findings noted |
| `src/write/horoscope_writer.py` | ~120 | Reviewed — findings noted |
| `src/render/__init__.py` | ~10 | Reviewed — no findings |
| `src/write/__init__.py` | 1 | Reviewed — no findings |
| `reference/app.py` | 171 | Reviewed — findings noted |
| `reference/config.py` | 49 | Reviewed — no findings |
| `reference/utils.py` | 425 | Reviewed — no findings |
| `requirements.txt` | 7 | Reviewed — finding noted |
| `pyproject.toml` | ~70 | Reviewed — finding noted |
| `Makefile` | ~50 | Reviewed — no findings |
| `.env` | 50 | Reviewed — **CRITICAL finding** |
| `.env.example` | 70 | Reviewed — no findings |
| `.gitignore` | 15 | Reviewed — finding noted |
| `out_real/manifest.json` | 50 | Reviewed — finding noted |
| `out_real/prompts/*.txt` | 12 files | Reviewed — finding noted |
| `data/horoscopes/2025-09-23/*.txt` | 12 files | Reviewed — finding noted |
| `.github/workflows/` | 0 files | **Empty — no CI/CD** |

---

*Report generated by Security Auditor (Microsoft SDL methodology). This report is for internal use only and should not be shared publicly. All findings are based on static analysis of the codebase as of 2026-03-21.*
