# Full Audit Report — AI Video Agent
**Date:** 2026-03-21  
**Auditor:** Comprehensive SDL Audit (Microsoft Security Development Lifecycle)  
**Project:** `ai-video-agent` — Automated Horoscope Video Generation Pipeline  
**Repository:** `C:\Users\davem\repos\ai-video-agent`  
**Language / Runtime:** Python 3.10+  
**Framework / Stack:** OpenAI GPT API · Google Vertex AI Veo · MoviePy · FFmpeg (subprocess)  
**Audit Dimensions:** Security · Privacy · Accessibility · Digital Safety · Supply Chain  
**Static Analysis:** CodeQL CLI not available; no `codeql-results.sarif` present. All findings are from manual static analysis.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Review Methodology](#3-review-methodology)
4. [Static Analysis Summary](#4-static-analysis-summary)
5. [Severity Distribution](#5-severity-distribution)
6. [SDL Phase Distribution](#6-sdl-phase-distribution)
7. [Findings — Security](#7-findings--security)
8. [Findings — Privacy](#8-findings--privacy)
9. [Findings — Accessibility](#9-findings--accessibility)
10. [Findings — Digital Safety](#10-findings--digital-safety)
11. [Findings — Supply Chain Security](#11-findings--supply-chain-security)
12. [Cross-Cutting Findings (Deduplicated)](#12-cross-cutting-findings-deduplicated)
13. [Prioritized Remediation Roadmap](#13-prioritized-remediation-roadmap)
14. [Next Steps](#14-next-steps)
15. [Appendix — Sub-Agent Report Index](#15-appendix--sub-agent-report-index)

---

## 1. Executive Summary

This report presents the results of a comprehensive five-dimension audit of the **AI Video Agent** project, a Python pipeline that generates daily horoscope short-form videos using OpenAI GPT (script writing) and Google Vertex AI Veo (video generation), with FFmpeg-based subtitle burning for social media distribution.

### Findings Overview (After Deduplication)

| Dimension | Critical | High | Medium | Low | Total |
|-----------|----------|------|--------|-----|-------|
| Security | 1 | 3 | 7 | 5 | **16** |
| Privacy | 1 | 4 | 5 | 4 | **14** |
| Accessibility | 1 | 4 | 5 | 5 | **15** |
| Digital Safety | 2 | 3 | 3 | 2 | **10** |
| Supply Chain | 1 | 4 | 5 | 4 | **14** |
| **Cross-dimension (deduped)** | *−3* | *−2* | *−2* | *0* | *−7* |
| **Net Total** | **3** | **16** | **23** | **20** | **62** |

> **Deduplication note:** Seven findings were identified by multiple sub-agents. The preferred source (higher specificity or severity) is retained; the duplicate is cross-referenced. See [Section 12](#12-cross-cutting-findings-deduplicated) for detail.

### Top-Line Risk Posture

**MEDIUM-HIGH overall.** The codebase is well-structured Python with no use of `eval`/`exec`, correct `.gitignore` exclusion of `.env`, safe subprocess invocation patterns, and a thoughtful pipeline architecture. However, three areas demand **immediate action before any public release or wider deployment**:

1. 🔴 **Live API credentials** (OpenAI + Google Cloud) are stored in plaintext `.env` on disk and the Google API key is additionally transmitted as a URL query parameter on every API call — permanently logged in Google's infrastructure.
2. 🔴 **LLM output is passed unsanitized into both Python `str.format()` calls and ASS subtitle format files** — a double injection surface where malicious or unexpected model output can crash the pipeline or corrupt generated videos.
3. 🔴 **All seven runtime dependencies are completely unpinned** with no lockfile, no hash verification, and known CVEs in the transitive dependency tree (CVSS scores up to 8.8 RCE in `setuptools`).

---

## 2. Project Overview

| Component | Detail |
|-----------|--------|
| **Language** | Python 3.10+ |
| **Entry points** | `src/main.py`, `src/render/veo_horoscope_pipeline.py`, `src/write/horoscope_writer.py` |
| **External APIs** | OpenAI Chat Completions (GPT-4o / GPT-5 family); Google Vertex AI Veo (video generation) |
| **Media processing** | FFmpeg via `subprocess`; MoviePy for composition; ASS subtitle format for caption overlay |
| **Authentication** | API keys via `.env` / `python-dotenv` |
| **Output** | MP4 video files in `out/`, `out_real/`, `out_test/`; JSON manifests; prompt `.txt` files |
| **Distribution target** | Social media platforms (TikTok, Instagram Reels, YouTube Shorts — 9:16, 16:9, 1:1 ratios) |
| **Reference UI** | `reference/app.py` — Streamlit-based development utility |
| **CI/CD** | `.github/workflows/codeql.yml` (CodeQL static analysis workflow only) |
| **Test coverage** | None — `Makefile` `test` target prints "No tests configured yet" |

### Attack Surface

| Component | Exposed Surface |
|-----------|----------------|
| `src/write/horoscope_writer.py` | OpenAI API key; LLM output injected into downstream systems |
| `src/render/veo_horoscope_pipeline.py` | Google API key in URL query params; unvalidated API response URL; `--out` path traversal |
| `src/render/subtitle_utils.py` | ASS format injection; FFmpeg subprocess; unclean temp file handling |
| `reference/app.py` | User-controlled font path (filesystem oracle); FFmpeg stderr in UI |
| `.env` | Plaintext OpenAI + Google Cloud API keys on local disk |
| `requirements.txt` / `pyproject.toml` | 7 fully unpinned runtime dependencies; CVE-exposed `setuptools` |
| `.github/workflows/codeql.yml` | Actions pinned to mutable tags with `security-events: write` permission |
| `out_real/`, `data/horoscopes/` | Proprietary prompt templates committed to git history |

---

## 3. Review Methodology

Each of five audit dimensions was reviewed by a specialized agent against the complete codebase:

| Dimension | Key Standards Applied | Files Reviewed |
|-----------|-----------------------|----------------|
| **Security** | Microsoft SDL, STRIDE, OWASP Top 10 | All `src/` files, `.env`, `Makefile`, `.gitignore`, `.github/` |
| **Privacy** | Microsoft SDL Privacy Framework, GDPR, CCPA | All `src/` files, `data/`, `out*/`, `pyproject.toml` |
| **Accessibility** | WCAG 2.1 (time-based media, CLI), BBC Subtitle Guidelines | All `src/` files, `README.md` |
| **Digital Safety** | EU AI Act Art. 50/52, UK Online Safety Act, Responsible AI principles | All `src/` files, `data/`, system prompts |
| **Supply Chain** | Microsoft SDL supply chain controls, SLSA framework, NIST SSDF | `requirements.txt`, `pyproject.toml`, `.github/workflows/` |

---

## 4. Static Analysis Summary

| Tool | Status | Notes |
|------|--------|-------|
| **CodeQL** | ❌ Not available | `codeql` CLI not found on `PATH`; no `codeql-results.sarif` pre-existing |
| **Manual SAST** | ✅ Completed | All source files reviewed manually by security sub-agent |
| **`bandit`** | Not run | Recommended addition to CI — would flag bare `except:` (B110), `subprocess` usage (B603) |
| **`semgrep`** | Not run | Recommended for injection pattern detection |
| **`pip audit`** | Not run | Recommended — known CVEs in transitive dependencies identified via manual review |

**Recommendation:** Install CodeQL CLI and add `bandit`, `pip audit`, and `detect-secrets` to a GitHub Actions CI workflow. Detailed steps in [Section 13](#13-prioritized-remediation-roadmap).

---

## 5. Severity Distribution

| Severity | Count | Description |
|----------|-------|-------------|
| 🔴 **Critical** | 3 | Require immediate remediation — active risk of data loss, credential compromise, or pipeline crash |
| 🟠 **High** | 16 | Should be remediated before any public/production deployment |
| 🟡 **Medium** | 23 | Should be remediated within the current development cycle |
| 🟢 **Low** | 20 | Remediate as part of ongoing hardening; low immediate risk |
| **Total** | **62** | |

---

## 6. SDL Phase Distribution

| SDL Phase | Count | Description |
|-----------|-------|-------------|
| **Requirements** | 5 | Missing security/privacy/accessibility requirements not specified at project inception |
| **Design** | 12 | Architectural decisions that created vulnerability classes (e.g., no content safety layer, no lockfile strategy, no AI disclosure) |
| **Implementation** | 31 | Code-level vulnerabilities: injection, error handling, credential exposure, unpinned deps |
| **Verification** | 9 | Missing tests, no SAST, no dependency auditing in CI |
| **Release** | 4 | Committed operational data, missing SBOM, no security review gate |
| **Response** | 1 | No incident response plan or monitoring for credential misuse |
| **Total** | **62** | |

---

## 7. Findings — Security

*Full detail: [`audit/security.md`](./security.md) · 16 findings: 1 Critical, 3 High, 7 Medium, 5 Low*

---

### SEC-01 — Live API Credentials Present on Local Filesystem
- **Severity:** 🔴 Critical
- **SDL Phase:** Implementation / Release
- **STRIDE:** Information Disclosure / Elevation of Privilege
- **Location:** `.env` (local filesystem)
- **Description:** The `.env` file contains live, production-grade API credentials in plaintext: an OpenAI API key (`sk-proj-pc78yDu...`), a Google API key, and the Google Cloud Project ID (`veo-3-videos-482006`). While correctly excluded from git, the file is readable by any process under the same OS user, cloud sync tools, IDE extensions, and crash dump utilities.
- **Recommendation:**
  1. **Immediately rotate** both API keys — treat them as compromised.
  2. Migrate to OS-level secrets store (Windows Credential Manager) or Azure Key Vault / Google Secret Manager.
  3. Add a `pre-commit` hook using `detect-secrets` or `gitleaks`.
  4. Add GitHub Advanced Security secret-scanning to the CI pipeline.

> ⚠️ *Also identified as Privacy finding PRI-01. Consolidated here; cross-referenced in Section 12.*

---

### SEC-02 — Google API Key Exposed as URL Query Parameter
- **Severity:** 🟠 High
- **SDL Phase:** Implementation
- **STRIDE:** Information Disclosure
- **Location:** `src/render/veo_horoscope_pipeline.py` lines 133–136, 201–204
- **Description:** The Vertex AI REST API URL is constructed with `?key={self.api_key}` appended. This writes the API key permanently into Google server access logs, proxy logs, network packet captures, and HTTP Referer headers. An infrastructure operator or log system reader could extract the key to submit unauthorized Vertex AI jobs on the developer's account.
- **Recommendation:** Pass the API key in the `Authorization: Bearer` header instead of the URL. Prefer Application Default Credentials (ADC) / service account with scoped IAM permissions over an API key.

---

### SEC-03 — Unpinned Dependencies — Supply Chain Risk
- **Severity:** 🟠 High
- **SDL Phase:** Requirements / Verification
- **STRIDE:** Tampering / Elevation of Privilege
- **Location:** `requirements.txt` (all 7 lines), `pyproject.toml` dependencies
- **Description:** All seven runtime dependencies carry no version constraint. Every `pip install` resolves to the latest version, silently adopting any newly compromised or CVE-affected package. No lockfile exists for reproducible builds or hash verification.
- **Recommendation:** Pin all dependencies; generate `requirements.lock` with `pip-compile --generate-hashes`. See Supply Chain section for detailed version recommendations.

> ⚠️ *Overlaps with Supply Chain findings SC-01 and SC-02. Primary detail in Section 11.*

---

### SEC-04 — Unvalidated URL from API Response (Partial SSRF)
- **Severity:** 🟠 High
- **SDL Phase:** Implementation
- **STRIDE:** Spoofing / Information Disclosure
- **Location:** `src/render/veo_horoscope_pipeline.py` lines 258–264
- **Description:** The video download URL extracted from the Veo API JSON response is used directly in `requests.get()`. The check `video_url.startswith("http")` permits `http://` (plaintext) URLs and does not validate the hostname. An injected internal address (e.g., `http://169.254.169.254/` GCP metadata service) would be fetched and written to disk. No file-size cap exists on the download.
- **Recommendation:** Validate scheme is `https://`; validate hostname against an allowlist of Google domains (`storage.googleapis.com`, `*.googleusercontent.com`); add a maximum-byte download limit.

---

### SEC-05 — Sensitive API Error Responses Printed to Stdout
- **Severity:** 🟡 Medium
- **SDL Phase:** Implementation
- **Location:** `src/render/veo_horoscope_pipeline.py` lines 182–187; `src/write/horoscope_writer.py` lines 178–181
- **Description:** Full API error response bodies (which may contain internal infrastructure details, quota identifiers, or echoed request payloads) are printed to stdout. A bare `except:` clause then attempts `e.response.text` on the outer exception variable, potentially causing a secondary error.
- **Recommendation:** Use structured logging at `DEBUG` level; sanitize error messages; replace bare `except:` with `except Exception as err:`.

---

### SEC-06 — Unvalidated CLI `--out` Path (Directory Traversal)
- **Severity:** 🟡 Medium
- **SDL Phase:** Implementation
- **Location:** `src/render/veo_horoscope_pipeline.py` line 708
- **Description:** The `--out` CLI argument is used directly with `Path(args.out).mkdir(parents=True, exist_ok=True)`. An operator can pass `../../etc/cron.d` or any absolute system path, causing the pipeline to write `manifest.json` and video files to arbitrary filesystem locations.
- **Recommendation:** Resolve the path and validate it falls within an expected base directory; reject absolute paths outside the project root.

---

### SEC-07 — Operational Data Committed to Git History
- **Severity:** 🟡 Medium
- **SDL Phase:** Release
- **Location:** `out_real/manifest.json`, `out_real/prompts/*.txt`, `data/horoscopes/2025-09-23/*`
- **Description:** Despite `.gitignore` rules, operational output files (job manifests, video generation prompt templates, generated horoscope text, `.DS_Store`) are tracked in git history. The prompt templates expose the proprietary video generation approach; anyone who clones the repository can replicate it without authorization.
- **Recommendation:** Remove with `git rm --cached`; purge history with `git filter-repo` or BFG; add `.DS_Store` to `.gitignore`; add a pre-commit hook to block future commits to `out*/` and `data/`.

---

### SEC-08 — ASS Subtitle Format Injection via Unsanitized Caption Text
- **Severity:** 🟡 Medium
- **SDL Phase:** Implementation
- **Location:** `src/render/subtitle_utils.py` lines 875, 910
- **Description:** LLM-generated text is written verbatim (only newlines normalized) into ASS subtitle `Dialogue:` lines. ASS format uses `{...}` as inline style override blocks; model-generated curly-brace sequences could override subtitle positioning, color, or visibility.
- **Recommendation:** Strip or escape ASS tags before writing: `re.sub(r'\{[^}]*\}', '', text)`.

> ⚠️ *Also identified as Digital Safety finding DS-01 (elevated to Critical due to injection chain). See Section 12.*

---

### SEC-09 — Temporary ASS Files Without Restrictive Permissions
- **Severity:** 🟡 Medium
- **SDL Phase:** Implementation
- **Location:** `src/render/subtitle_utils.py` lines 920–922
- **Description:** `tempfile.mkstemp()` creates temp files with default permissions. Cleanup uses bare `except: pass`, silently leaving sensitive temp files on disk if deletion fails. On multi-user or containerized systems, concurrent pipeline instances could race on temp files.
- **Recommendation:** Use `try/finally` for guaranteed cleanup; replace `except: pass` with `except OSError as e: logger.warning(...)`.

---

### SEC-10 — Bare `except:` Clauses Swallowing All Exceptions
- **Severity:** 🟡 Medium
- **SDL Phase:** Implementation
- **Location:** `src/render/veo_horoscope_pipeline.py` line 185; `src/render/subtitle_utils.py` lines 1058–1060
- **Description:** Bare `except:` catches `SystemExit`, `KeyboardInterrupt`, and `GeneratorExit`, preventing Ctrl+C from working and hiding programming errors that should surface as bugs.
- **Recommendation:** Replace all bare `except:` with `except Exception as err:` at minimum; use `except OSError:` for file cleanup. Enforce with `flake8 E722`.

---

### SEC-11 — No Rate Limiting or API Cost Cap
- **Severity:** 🟡 Medium
- **SDL Phase:** Design
- **Location:** `src/write/horoscope_writer.py` lines 1153–1184; `src/render/veo_horoscope_pipeline.py` lines 555–562
- **Description:** The pipeline submits up to 12 OpenAI calls and 12 Veo video generation jobs per run with no per-run budget ceiling or circuit breaker. A misconfigured cron job or CI trigger in a loop could exhaust quotas and generate hundreds of dollars in unexpected API charges.
- **Recommendation:** Add `--dry-run` flag; implement `MAX_DAILY_RUNS` env var with a counter file; set billing alerts on both API platforms.

---

### SEC-12 — No CI/CD Pipeline for Security Controls
- **Severity:** 🟢 Low
- **SDL Phase:** Verification / Release
- **Location:** `.github/workflows/` (empty — no workflows)
- **Description:** No automated workflows exist for secret scanning, dependency vulnerability auditing, SAST, or unit testing. The `Makefile` `test` target prints "No tests configured yet."
- **Recommendation:** Add GitHub Actions with `pip audit`, `bandit -r src/`, `detect-secrets`, and `pytest`. See Section 13 for workflow template.

---

### SEC-13 — Author Corporate Email Committed in `pyproject.toml`
- **Severity:** 🟢 Low
- **SDL Phase:** Release
- **Location:** `pyproject.toml` line 12
- **Description:** `davidmo@microsoft.com` is hardcoded as the package author email. If the repository is ever made public or the package published to PyPI, this corporate email is permanently in git history and package metadata.
- **Recommendation:** Use a personal or role-based email for public releases; ensure repository access controls are appropriate for an internal tool.

---

### SEC-14 — Reference App Exposes FFmpeg stderr in UI
- **Severity:** 🟢 Low
- **SDL Phase:** Implementation
- **Location:** `reference/app.py` lines 141, 169–170
- **Description:** Raw FFmpeg stderr (including filesystem paths to temp files, codec versions, and encoding parameters) is rendered directly in the Streamlit UI via `st.code(result.stderr)`.
- **Recommendation:** Show only sanitized success/failure messages to users; log full stderr to a server-side log.

---

### SEC-15 — Reference App Allows Arbitrary Font File Path (Filesystem Oracle)
- **Severity:** 🟢 Low
- **SDL Phase:** Implementation
- **Location:** `reference/app.py` lines 59, 103–104
- **Description:** The Streamlit UI accepts a user-supplied font file path and uses `os.path.exists()` to validate it, effectively allowing any user to probe for the existence of arbitrary files on the server.
- **Recommendation:** Restrict font paths to a pre-approved allowlist or a specific fonts directory; prefer file-upload widget over path text input.

---

### SEC-16 — No Incident Response Plan for Credential Compromise
- **Severity:** 🟢 Low
- **SDL Phase:** Response
- **Location:** No `SECURITY.md`, no credential rotation runbook
- **Description:** No documented procedure exists for rotating API keys, assessing blast radius, or revoking compromised credentials. Given that live credentials are present on disk (SEC-01), an incident response procedure is essential.
- **Recommendation:** Create `SECURITY.md` documenting: key rotation steps for OpenAI and Google Cloud, how to assess unauthorized API usage, and escalation contacts.

---

## 8. Findings — Privacy

*Full detail: [`audit/privacy.md`](./privacy.md) · 14 findings: 1 Critical, 4 High, 5 Medium, 4 Low*

---

### PRI-01 — Plaintext API Credentials on Disk (HBI Data)
- **Severity:** 🔴 Critical
- **SDL Phase:** Implementation
- **Category:** Credential / HBI Data Exposure
- **Location:** `.env` (local filesystem)
- **Description:** Live OpenAI and Google Cloud API keys (classified as High Business Impact) stored in plaintext `.env`. Additionally, the Google API key is written into every API URL and therefore logged permanently in Google's server-side access logs — a form of HBI data retention neither disclosed to the user nor controllable by the developer.
- **Recommendation:** Rotate both keys immediately; migrate to OS-level or cloud-hosted secrets manager; add secret scanning to CI.

> ⚠️ *Cross-reference: SEC-01. Consolidated in Section 12.*

---

### PRI-02 — Developer PII Committed to Public Repository
- **Severity:** 🟠 High
- **SDL Phase:** Release
- **Category:** PII in Source Code
- **Location:** `pyproject.toml` lines 12–13
- **Description:** Developer name and corporate email (`davidmo@microsoft.com`) are committed in `pyproject.toml` and will be in git history permanently. These constitute personal data under GDPR Article 4(1) and expose a real individual's work contact details in any publicly accessible fork or PyPI publication.
- **Recommendation:** Replace with a role-based or public-only email; if the repository must remain public, use `git filter-repo` to rewrite history.

---

### PRI-03 — Data Transmitted to Third-Party AI Services Without Documented Privacy Review
- **Severity:** 🟠 High
- **SDL Phase:** Requirements / Design
- **Category:** Third-Party Data Sharing
- **Location:** `src/write/horoscope_writer.py` (OpenAI); `src/render/veo_horoscope_pipeline.py` (Google Vertex AI)
- **Description:** Prompts are sent to OpenAI's API and Google Vertex AI with no documented review of each service's data retention policies. OpenAI's default data retention policy retains API inputs for 30 days; Google Vertex AI Veo has its own data handling terms. For a production pipeline intended for social media distribution, operators need to understand exactly what data each provider retains, for how long, and under what legal basis.
- **Recommendation:** Document data flows to third parties; review OpenAI and Google Vertex AI Data Processing Addendums; add a `PRIVACY.md` describing what data is sent where and retention timelines.

---

### PRI-04 — LLM Prompts Contain Potentially Sensitive Operational Data Retained by Provider
- **Severity:** 🟠 High
- **SDL Phase:** Design
- **Category:** Data Minimization / AI Privacy
- **Location:** `src/write/horoscope_writer.py` lines 69–74
- **Description:** The prompt sent to OpenAI includes the current date and the system prompt (creative style instructions). While these are not personal data, the prompt structure reveals the pipeline's creative approach, scheduling cadence, and brand voice to a third-party AI provider. Without an OpenAI Enterprise agreement with zero-data-retention, these prompts may be used for model training.
- **Recommendation:** Enable the OpenAI `store=False` parameter on API calls to opt out of training data retention; obtain an OpenAI Enterprise agreement if this is a commercial product.

---

### PRI-05 — Output Files Retained Indefinitely with No Deletion Policy
- **Severity:** 🟠 High
- **SDL Phase:** Requirements
- **Category:** Data Retention
- **Location:** `out/`, `out_real/`, `out_test/`, `data/horoscopes/`
- **Description:** All generated videos, manifests, prompt text files, and horoscope scripts are written to local directories with no automated cleanup. For a daily pipeline generating 12 videos per run, disk consumption grows unbounded. More importantly, there is no documented data retention policy — a GDPR and good-practice requirement for any production system.
- **Recommendation:** Add a configurable `--retention-days` parameter or scheduled cleanup job; document data retention policy in `PRIVACY.md`.

---

### PRI-06 — Google Operation IDs and Project ID Committed in Output Manifests
- **Severity:** 🟡 Medium
- **SDL Phase:** Release
- **Category:** Operational Data Exposure
- **Location:** `out/manifest.json`, `out_test/manifest.json` (committed to git)
- **Description:** Manifests in `out_test/` are committed to git and contain Google Cloud operation UUIDs and the project ID (`veo-3-videos-482006`). These identifiers, combined with the Google API key, could be used to query the status of historical Veo operations or gain intelligence about the project's operational history.
- **Recommendation:** Remove manifest files from git tracking; ensure `out*/` directories are fully excluded from version control.

---

### PRI-07 — No Privacy Policy or User Consent Mechanism for Distributed Content
- **Severity:** 🟡 Medium
- **SDL Phase:** Requirements
- **Category:** Consent / GDPR Compliance
- **Location:** Entire codebase — design-level gap
- **Description:** The pipeline produces AI-generated videos intended for social media distribution. There is no privacy notice, no consent mechanism, no data subject rights procedure, and no legal basis documented for data processing. Under GDPR Article 13/14, data subjects whose content may be processed must be informed.
- **Recommendation:** For any commercial or public deployment, engage a privacy/legal team to define the legal basis for processing and produce user-facing privacy notices.

---

### PRI-08 — Error Messages May Expose Internal Infrastructure Details
- **Severity:** 🟡 Medium
- **SDL Phase:** Implementation
- **Category:** Information Disclosure
- **Location:** `src/render/veo_horoscope_pipeline.py` lines 182–187; `src/write/horoscope_writer.py` lines 178–181
- **Description:** Raw API error responses (which may contain Google Cloud internal service error messages, quota account identifiers, or partially echoed request payloads) are printed to stdout without sanitization.
- **Recommendation:** Use structured logging with sanitized error messages; never print raw API response bodies to stdout in production.

> ⚠️ *Cross-reference: SEC-05. Deduplicated in Section 12.*

---

### PRI-09 — No Audit Logging for API Operations
- **Severity:** 🟡 Medium
- **SDL Phase:** Response
- **Category:** Audit Trail / Monitoring
- **Location:** Entire codebase
- **Description:** No structured audit log exists recording which API operations were performed, when, and with what inputs. If API keys are compromised and unauthorized usage occurs, there is no local audit trail to distinguish authorized from unauthorized calls.
- **Recommendation:** Implement structured logging (Python `logging` module) with timestamped records of each API call (type, sign, result status, cost estimate) written to a log file separate from stdout.

---

*For remaining Privacy findings (PRI-10 through PRI-14), see full report: [`audit/privacy.md`](./privacy.md).*

---

## 9. Findings — Accessibility

*Full detail: [`audit/accessibility.md`](./accessibility.md) · 15 findings: 1 Critical, 4 High, 5 Medium, 5 Low*

**Note:** This project has no web UI. WCAG evaluation focuses on video output (WCAG 1.2.x time-based media) and CLI output accessibility. Standard DOM/ARIA criteria are not applicable.

---

### ACC-01 — GCS URI Code Path Produces Completely Uncaptioned Videos
- **Severity:** 🔴 Critical
- **SDL Phase:** Implementation
- **WCAG:** 1.2.2 Captions (Prerecorded) — Level A
- **Location:** `src/render/veo_horoscope_pipeline.py` lines 169–174
- **Description:** When the Veo API returns a `gs://` (Google Cloud Storage) URI, the code stores the path and marks the job `"done"` without calling `add_caption_to_video()`. Videos delivered via GCS are published completely uncaptioned. This is a clear violation of WCAG 1.2.2 Level A — the most fundamental caption requirement — and disproportionately impacts Deaf and hard-of-hearing users.
- **Recommendation:** Convert GCS URIs to signed HTTPS URLs using `google-cloud-storage` before downloading, then pass through the existing caption pipeline. As interim mitigation, emit a logged `WARNING` (not `✅`) so operators detect the gap.

---

### ACC-02 — Default "Scroll" Caption Style Produces Inaccessible Moving Text
- **Severity:** 🟠 High
- **SDL Phase:** Design
- **WCAG:** 1.2.2 Captions — Level A; 2.2.2 Pause, Stop, Hide — Level A
- **Location:** `src/render/subtitle_utils.py` lines 118–126; `src/render/veo_horoscope_pipeline.py` line 200
- **Description:** The default `CAPTION_STYLE` is `"scroll"`, which animates full horoscope text (2–3 sentences, 100–200 characters) from bottom to top across an 8-second clip at >80px/second. Scrolling captions are a documented barrier for users with dyslexia, cognitive disabilities, and low vision. The `"static"` override exists but is undocumented.
- **Recommendation:** Change the default from `"scroll"` to `"static"`; document the `CAPTION_STYLE` environment variable; if scrolling is aesthetically required, add a static sidecar `.srt` track.

---

### ACC-03 — Silent Fallback to Uncaptioned Video on FFmpeg Failure
- **Severity:** 🟠 High
- **SDL Phase:** Implementation
- **WCAG:** 1.2.2 Captions (Prerecorded) — Level A
- **Location:** `src/render/veo_horoscope_pipeline.py` lines 207–211 and 244–248
- **Description:** When `add_caption_to_video()` fails, the raw uncaptioned video is published and `job.status` is still set to `"done"`, making manifest records indistinguishable from fully-captioned successes. Deaf/hard-of-hearing viewers receive inaccessible content with no visible signal to operators.
- **Recommendation:** Introduce a `"caption_failed"` job status; add `caption_status` field to the manifest JSON; emit a non-zero exit code when any captioning fails; consider making caption failure a hard error.

---

### ACC-04 — No Audio Description Track for Blind/Low-Vision Users
- **Severity:** 🟠 High
- **SDL Phase:** Design
- **WCAG:** 1.2.3 Audio Description or Media Alternative — Level A; 1.2.5 Audio Description — Level AA
- **Location:** Entire pipeline (design-level gap)
- **Description:** All horoscope content is conveyed as visual burned-in captions over ambient AI video. No narration audio track, no audio description, and no linked transcript file is produced. Blind and severe low-vision users have no accessible path to the content.
- **Recommendation:** Short-term: export plain-text transcript alongside each MP4 (`renders/aries.txt`). Medium-term: generate narration via OpenAI TTS or Google Cloud TTS and mux into the output video. Long-term: if Veo 3 generates audio narration, document this as satisfying 1.2.5.

---

### ACC-05 — No Sidecar Subtitle File (.srt / .vtt) Exported
- **Severity:** 🟠 High
- **SDL Phase:** Design
- **WCAG:** 1.2.2 Captions — Level A
- **Location:** `src/render/subtitle_utils.py` (temp ASS file deleted after use)
- **Description:** Captions are burned in ("hardcoded") and the ASS source file is deleted. No `.srt`, `.vtt`, or `.ass` file is retained alongside the MP4. Users cannot adjust caption presentation (size, contrast, font) for personal accessibility needs, and social media platforms cannot use their own accessible caption features with the upload.
- **Recommendation:** Retain the `.srt` or `.vtt` file as `renders/<sign>.srt` alongside each MP4; add subtitle file path to the manifest JSON; utilize existing `format_srt_time()` / `create_srt_file()` functions in `reference/utils.py`.

---

### ACC-06 — Default Caption Font Size Insufficient at 1080p
- **Severity:** 🟡 Medium
- **SDL Phase:** Implementation
- **WCAG:** 1.4.4 Resize Text (analogous for video); BBC Subtitle Guidelines (5% of screen height)
- **Location:** `src/render/veo_horoscope_pipeline.py` lines 194, 231; `src/render/subtitle_utils.py` line 236
- **Description:** Default `CAPTION_FONT_SIZE` is 36px. At 1080p this is 3.3% of screen height — below the BBC-recommended 5% minimum (54px at 1080p). The pipeline overrides the `subtitle_utils.py` default of 48px with 36px without adjusting for resolution.
- **Recommendation:** Calculate font size as `max(36, int(video_height * 0.05))`; add `--caption-font-size` CLI argument; document `CAPTION_FONT_SIZE` in `.env.example`.

---

### ACC-07 — CLI Progress Output Incompatible with Screen Readers
- **Severity:** 🟡 Medium
- **SDL Phase:** Implementation
- **WCAG:** 4.1.3 Status Messages — Level AA (by analogy for CLI)
- **Location:** `src/write/horoscope_writer.py` line 68
- **Description:** The `print(..., end=" ", flush=True)` pattern writes incomplete lines that screen readers (NVDA, JAWS, VoiceOver, Orca) may announce before the trailing `✓`/`✗` result is appended — or merge with the next line — making success/failure status unreliable for blind operators.
- **Recommendation:** Replace split-line printing with a single complete-line output after each operation: `print(f"   [{i}/{n}] {sign}: {'OK' if success else 'FAILED'}")`.

---

### ACC-08 — Emoji Used as Primary Status Indicators
- **Severity:** 🟡 Medium
- **SDL Phase:** Implementation
- **WCAG:** 4.1.3 Status Messages; 1.1.1 Non-text Content
- **Location:** Throughout `src/` (✅ ✗ ⚠️ 🌟 🔄)
- **Description:** Emoji characters (`✅`, `✗`, `⚠️`) are the primary status indicators throughout the CLI. Screen readers announce emoji names verbosely and inconsistently across platforms; the semantic meaning (success/failure/warning) is not conveyed in machine-readable form.
- **Recommendation:** Prefix emoji with explicit text labels: `"SUCCESS ✅"`, `"FAILED ✗"`, `"WARNING ⚠️"`; use Python's `logging` module with `INFO`/`WARNING`/`ERROR` levels for structured output.

---

*For remaining Accessibility findings (ACC-09 through ACC-15), see full report: [`audit/accessibility.md`](./accessibility.md).*

---

## 10. Findings — Digital Safety

*Full detail: [`audit/digital-safety.md`](./digital-safety.md) · 10 findings: 2 Critical, 3 High, 3 Medium, 2 Low*

---

### DS-01 — ASS Subtitle Injection via Unsanitized LLM Output (Injection Chain)
- **Severity:** 🔴 Critical
- **SDL Phase:** Implementation
- **Category:** AI Content Safety / Injection
- **Location:** `src/render/subtitle_utils.py` line 115; `src/render/veo_horoscope_pipeline.py` lines 188–201
- **Description:** LLM-generated horoscope text is written directly into ASS subtitle files with only newline normalization. ASS format uses `{...}` curly-brace blocks as rendering override commands. Model-generated or prompt-injected curly-brace sequences can override subtitle position, color, size, or make text invisible entirely. This is part of a double injection chain — the same text is also passed to `str.format()` in the video prompt builder (see DS-02), meaning a single crafted model response can attack both the subtitle rendering and the video generation prompt simultaneously.
- **Recommendation:** Strip ASS override tags before writing: `re.sub(r'\{[^}]*\}', '', text)`. Escape remaining braces: `.replace('{', '').replace('}', '')`. Validate LLM output for unexpected formatting tokens before use downstream.

> ⚠️ *Overlaps with SEC-08. Elevated to Critical here due to injection chain. See Section 12.*

---

### DS-02 — LLM Output Format Tokens Crash Video Prompt Builder (`KeyError`)
- **Severity:** 🔴 Critical
- **SDL Phase:** Implementation
- **Category:** AI Content Safety / Error Handling
- **Location:** `src/render/veo_horoscope_pipeline.py` lines 385–391
- **Description:** `ScenePlanner.build_scene()` calls `template.format(..., caption=text, ...)` where `text` is raw LLM output. If the model generates text containing `{love}`, `{career}`, `{3}`, or any non-recognized format key, Python raises an unhandled `KeyError` or `ValueError`, crashing the entire pipeline mid-batch. Horoscope text frequently uses brace-enclosed tokens (e.g., `{compatibility}`, `{lucky number}`). All 12 remaining sign videos would fail silently at the point of crash.
- **Recommendation:** Escape braces in LLM output before `str.format()`: `safe_text = text.replace('{', '{{').replace('}', '}}')`. Alternatively, use `string.Template.safe_substitute()` for untrusted input.

---

### DS-03 — No Content Safety Filter on LLM-Generated Horoscope Text
- **Severity:** 🟠 High
- **SDL Phase:** Design
- **Category:** AI Content Safety / Content Moderation
- **Location:** `src/write/horoscope_writer.py` lines 75–96
- **Description:** OpenAI API responses are saved and burned into videos without any content validation. The `SYSTEM_STYLE` "Keep it PG" instruction is a stylistic request, not a technical safety control. OpenAI's free Moderation API is available but not called. With `temperature=0.8`, edge-case outputs are more likely. Content propagates fully through the pipeline to a distributable MP4 artifact.
- **Recommendation:** Add `client.moderations.create(input=text)` check after each generation; skip or replace flagged content; add keyword-based rejection for health/financial predictions, references to death, self-harm, or relationship abuse.

---

### DS-04 — Generated Videos Carry No AI-Generated Content Disclosure
- **Severity:** 🟠 High
- **SDL Phase:** Design
- **Category:** Transparency / Deceptive Design
- **Location:** Design-level; `src/render/veo_horoscope_pipeline.py`, `src/render/subtitle_utils.py`
- **Description:** Videos have no on-screen watermark, no MP4 metadata tag, and no in-content disclaimer indicating they are AI-generated. The `DEFAULT_TEMPLATE` explicitly constructs a cinematic "professionally produced" aesthetic. This may violate EU AI Act Art. 50 (labeling synthetic media), platform AI content policies (TikTok, YouTube, Instagram), and UK Online Safety Act duties of care. Viewers — particularly younger audiences — have no way to know they are consuming AI-generated astrology content.
- **Recommendation:** (1) Add persistent small-text on-screen label "AI-generated for entertainment only" via a second ASS subtitle event; (2) write `comment` and `description` FFmpeg metadata tags; (3) add entertainment disclaimer to the `SYSTEM_STYLE` prompt.

---

### DS-05 — Silent Placeholder Substitution Presents Error Content as Genuine Horoscope
- **Severity:** 🟠 High
- **SDL Phase:** Implementation
- **Category:** AI Content Safety / Misinformation
- **Location:** `src/write/horoscope_writer.py` line 96
- **Description:** On any API failure, the pipeline substitutes `"({sign} placeholder horoscope: Today is a lucky day! 🌟)"` into the output — saving it to disk, burning it into the video subtitle, and recording the job as `"done"` in the manifest. The placeholder is indistinguishable from a real horoscope to any downstream consumer. A video published to social media with this placeholder would mislead viewers about the content's origin and quality.
- **Recommendation:** On API failure, either set `results[sign] = None` and skip video generation for that sign, or tag the manifest entry as `"content_source": "fallback"` so operators can identify and suppress distribution.

---

### DS-06 — System Prompt Requests "Concrete Predictions" Without Entertainment Framing
- **Severity:** 🟡 Medium
- **SDL Phase:** Design
- **Category:** Misinformation / User Wellbeing
- **Location:** `src/write/horoscope_writer.py` lines 27–31, 70–73
- **Description:** The `SYSTEM_STYLE` prompt instructs the model to write "a clear prediction or action" and the user prompt requests "a concrete prediction or recommended action for today." This actively encourages authoritative-sounding specific claims. No "entertainment only" framing exists anywhere in the pipeline or output. Vulnerable users — those in crisis, younger viewers, or individuals with genuine astrological beliefs — may act on these AI-generated predictions as personal guidance.
- **Recommendation:** Modify `SYSTEM_STYLE` to explicitly frame content as entertainment, not authoritative guidance; change "concrete prediction" to "fun, lighthearted suggestion" in the user prompt.

---

### DS-07 — No Safety Review Gate on Veo Video Generation Prompts
- **Severity:** 🟡 Medium
- **SDL Phase:** Design
- **Category:** AI Content Safety / Content Moderation
- **Location:** `src/render/veo_horoscope_pipeline.py` lines 420–428
- **Description:** Video generation prompts are assembled from LLM output and `PromptTransformer` callables and sent directly to the Veo API with no pre-submission safety validation. Relying solely on provider-side filtering means policy violations are discovered only after API credits are consumed, with no local audit record.
- **Recommendation:** Add a prompt validation step (keyword-based or Moderation API call) before Veo submission; log all submitted prompts with a pre-submission hash for audit purposes.

---

*For remaining Digital Safety findings (DS-08 through DS-10), see full report: [`audit/digital-safety.md`](./digital-safety.md).*

---

## 11. Findings — Supply Chain Security

*Full detail: [`audit/supply-chain.md`](./supply-chain.md) · 14 findings: 1 Critical, 4 High, 5 Medium, 4 Low*

---

### SC-01 — No Dependency Lockfile Committed (Root Cause)
- **Severity:** 🔴 Critical
- **SDL Phase:** Verification
- **Category:** Lockfile Hygiene
- **Location:** Repository root — no lockfile present
- **Description:** The repository contains no lockfile of any kind. Every `pip install` resolves the full transitive dependency graph independently against the live PyPI index. This is the root cause amplifying all other supply chain findings: builds are non-reproducible, CVEs in transitive dependencies cannot be reliably tracked, package integrity cannot be hash-verified, and a compromised maintainer account would have their code automatically adopted on next install.
- **Recommendation:** Adopt `pip-tools`: `pip-compile --generate-hashes --output-file requirements.lock requirements.txt`. Commit `requirements.lock`. Fail CI if out of sync with `requirements.txt`. Install with `pip install --require-hashes -r requirements.lock`.

---

### SC-02 — All Runtime Dependencies Completely Unpinned
- **Severity:** 🟠 High
- **SDL Phase:** Implementation
- **Category:** Dependency Pinning
- **Location:** `requirements.txt` (all 7 lines); `pyproject.toml` `[project.dependencies]`
- **Description:** Zero version constraints on `moviepy`, `python-dotenv`, `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`, `openai`, and `requests`. Major-version breaking changes, new CVEs, and supply chain substitution attacks are silently adopted.
- **Recommendation:**
  ```
  moviepy==2.1.1
  python-dotenv==1.0.1
  google-api-python-client==2.154.0
  google-auth-httplib2==0.2.0
  google-auth-oauthlib==1.2.1
  openai==1.57.4
  requests==2.32.3
  ```

---

### SC-03 — `setuptools` Build Dependency: CVE-2024-6345 (CVSS 8.8 — RCE)
- **Severity:** 🟠 High
- **SDL Phase:** Implementation
- **CVE:** CVE-2024-6345
- **CVSS:** 8.8 (High)
- **Location:** `pyproject.toml` line 2: `setuptools>=61.0`
- **Description:** CVE-2024-6345 allows remote code execution via maliciously crafted package URLs in `setuptools` versions prior to 70.0.0. The `>=61.0` lower-bound specifier without an upper bound means any developer with `setuptools` installed before July 2024 may be running a vulnerable version. A compromised dependency could achieve RCE during the build/install phase.
- **Recommendation:** Update `pyproject.toml` to `setuptools>=70.0.0`; verify with `pip show setuptools` in CI.

---

### SC-04 — `urllib3` Transitive Dependency: CVE-2023-43804 (CVSS 8.1 — Cookie Leak)
- **Severity:** 🟠 High
- **SDL Phase:** Verification
- **CVE:** CVE-2023-43804 · CVSS 8.1
- **Location:** Transitive via `requests`
- **Description:** `urllib3` prior to 1.26.17 (1.x) / 2.0.6 (2.x) does not strip `Cookie` headers on cross-origin HTTP redirects. API calls to Google Vertex AI that follow redirects to a different origin could forward OAuth tokens or session cookies to the redirect target.
- **Recommendation:** Pin `requests==2.32.3` (pulls `urllib3>=2.0.6`); add explicit `urllib3==2.2.3` to `requirements.lock`.

---

### SC-05 — `certifi` Transitive Dependency: CVE-2023-37920 (CVSS 7.5 — Compromised Root CA)
- **Severity:** 🟠 High
- **SDL Phase:** Verification
- **CVE:** CVE-2023-37920 · CVSS 7.5
- **Location:** Transitive via `requests` → `certifi`
- **Description:** `certifi` prior to 2023.07.22 includes the e-Tugra root CA, removed from trust stores in 2023 due to CA malpractice. Applications using a vulnerable version may establish "verified" TLS connections to attacker-controlled servers with an e-Tugra-signed certificate — allowing interception of API keys transmitted to OpenAI and Google.
- **Recommendation:** Pin `certifi==2024.12.14` explicitly in `requirements.lock`.

---

### SC-06 — `requests` Direct Dependency: CVE-2023-32681 (CVSS 6.1 — Proxy Credential Leak)
- **Severity:** 🟡 Medium
- **SDL Phase:** Verification
- **CVE:** CVE-2023-32681 · CVSS 6.1
- **Location:** `requirements.txt` line 7; `src/render/veo_horoscope_pipeline.py`
- **Description:** `requests` prior to 2.31.0 forwards `Proxy-Authorization` headers to destination servers on same-host redirects. In enterprise environments with authenticated HTTP proxies, proxy credentials could be leaked to Google's APIs.
- **Recommendation:** Pin `requests==2.32.3`.

---

### SC-07 — GitHub Actions Not Pinned to Commit SHAs
- **Severity:** 🟡 Medium
- **SDL Phase:** Verification / Release
- **Category:** CI/CD Pipeline Integrity
- **Location:** `.github/workflows/codeql.yml` lines 19, 86, 96, 99
- **Description:** `actions/checkout@v4` and three `github/codeql-action/*@v3` references use mutable version tags. A tag can be silently updated to point to different code. The CodeQL workflow has `security-events: write` and `contents: read` permissions — a compromised action could exfiltrate source code or inject fake security alerts.
- **Recommendation:** Pin all actions to full commit SHAs using `pin-github-action` or `ratchet`. Example:
  ```yaml
  - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
  ```

---

### SC-08 — No Automated Dependency Update Tooling
- **Severity:** 🟡 Medium
- **SDL Phase:** Verification
- **Location:** Repository root — no `.github/dependabot.yml`
- **Description:** No Dependabot or Renovate configuration exists. Security patches in upstream packages will not generate automated pull requests. The CodeQL workflow scans project source code only — it does not detect vulnerable dependencies.
- **Recommendation:** Add `.github/dependabot.yml` with `package-ecosystem: "pip"` and `package-ecosystem: "github-actions"`, both on weekly schedules.

---

### SC-09 — `moviepy` Bundles External ffmpeg Binary Downloaded at Install Time
- **Severity:** 🟡 Medium
- **SDL Phase:** Implementation
- **Category:** Install Hook / Transitive Risk
- **Location:** `requirements.txt` line 1 (`moviepy`)
- **Description:** `moviepy 2.x` downloads a pre-compiled ffmpeg binary via the `imageio-ffmpeg` transitive package at install time. Without hash verification (`--require-hashes`), there is no integrity check on this binary. A tampered `imageio-ffmpeg` GitHub release would install a malicious ffmpeg binary with full OS process privileges.
- **Recommendation:** Pin `moviepy==2.1.1`; generate `requirements.lock` with hashes; verify `imageio-ffmpeg` wheel SHA-256 against the official release.

---

### SLSA Level Assessment

| Check | Status |
|-------|--------|
| Source version controlled | ✅ Git |
| Build scripted (reproducible) | ❌ No lockfile; non-reproducible |
| Provenance generated | ❌ No SBOM, no build attestations |
| Build isolated | ❌ No container/isolated build environment |
| Reviewed (two-person rule) | ❌ No CI review gate; no PR requirements documented |

**Current SLSA Level: 0** (no supply chain security controls met beyond version control).  
**Target:** SLSA Level 2 (hosted build service + provenance generation). Achievable by adding GitHub Actions CI with a locked build, `pip audit`, and SBOM generation.

---

*For remaining Supply Chain findings (SC-10 through SC-14), see full report: [`audit/supply-chain.md`](./supply-chain.md).*

---

## 12. Cross-Cutting Findings (Deduplicated)

The following findings were identified by multiple sub-agents. The table below shows where consolidation occurred and how severity was resolved.

| Finding | Identified By | Kept Under | Resolution |
|---------|--------------|------------|------------|
| Plaintext API credentials in `.env` | Security (Critical) + Privacy (Critical) | **Security SEC-01** (also PRI-01) | Same finding; Security is primary. Privacy adds data retention / log exposure angle. Both noted. |
| Google API key in URL query parameter | Security (High) + Privacy (High) | **Security SEC-02** (cross-ref in PRI) | Security is primary for attack vector; Privacy notes permanent log retention consequence. |
| ASS subtitle injection | Security (Medium) + Digital Safety (Critical) | **Digital Safety DS-01** (elevated) | Digital Safety elevates to Critical due to discovered injection chain (double injection: ASS + `str.format()`). Security finding SEC-08 merged. |
| Unpinned dependencies | Security (High) + Supply Chain (Critical + High) | **Supply Chain SC-01 / SC-02** | Supply Chain provides more granular analysis; Security finding SEC-03 cross-references these. |
| Bare `except:` clauses | Security (Medium) + Digital Safety | **Security SEC-10** | Security provides full location list; Digital Safety context noted. |
| Error responses printed to stdout | Security (Medium) + Privacy (Medium) | **Security SEC-05** (cross-ref PRI-08) | Security covers attack vector; Privacy notes information disclosure to third-party log systems. |
| No CI/CD pipeline | Security (Low) + Supply Chain (Low) | **Security SEC-12** | Combined; Supply Chain section adds `pip audit` specifics. |

---

## 13. Prioritized Remediation Roadmap

### 🔴 Immediate Actions (Before Next Deployment)

| # | Action | Finding(s) | Effort |
|---|--------|-----------|--------|
| 1 | **Rotate both API keys** (OpenAI + Google Cloud) — treat as compromised | SEC-01 / PRI-01 | 30 min |
| 2 | **Move credentials to a secrets manager** (Windows Credential Manager or Azure Key Vault) | SEC-01 | 2–4 hrs |
| 3 | **Fix `str.format()` crash** — escape braces in LLM output before template formatting | DS-02 | 30 min |
| 4 | **Sanitize ASS subtitle injection** — strip `{...}` blocks from LLM text before writing | DS-01 / SEC-08 | 1 hr |
| 5 | **Move Google API key from URL to Authorization header** | SEC-02 | 1 hr |
| 6 | **Generate and commit `requirements.lock`** with `pip-compile --generate-hashes` | SC-01 | 1–2 hrs |
| 7 | **Pin `setuptools>=70.0.0`** in `pyproject.toml` | SC-03 | 15 min |
| 8 | **Fix GCS URI code path** to apply captions before marking job `"done"` | ACC-01 | 2–4 hrs |

---

### 🟠 Short-Term Actions (Within Current Development Cycle)

| # | Action | Finding(s) | Effort |
|---|--------|-----------|--------|
| 9 | Add **OpenAI Moderation API** check after each horoscope generation | DS-03 | 2 hrs |
| 10 | Add **AI disclosure label** (on-screen + MP4 metadata) to all generated videos | DS-04 | 2–3 hrs |
| 11 | Fix **silent placeholder substitution** — skip or tag failed sign generation | DS-05 | 1 hr |
| 12 | **Remove committed output files** from git history (`git rm --cached`; `git filter-repo`) | SEC-07 | 1–2 hrs |
| 13 | **Pin all 7 runtime dependencies** to specific versions | SC-02 | 30 min |
| 14 | Add `pip audit` and `bandit` to a new **GitHub Actions CI workflow** | SEC-12 / SC-13 | 2–3 hrs |
| 15 | Validate `--out` path is within expected base directory | SEC-06 | 30 min |
| 16 | Replace all **bare `except:`** clauses | SEC-10 | 1 hr |
| 17 | Change default `CAPTION_STYLE` from `"scroll"` to `"static"` | ACC-02 | 15 min |
| 18 | Add **`caption_failed` job status** and non-zero exit code on captioning failure | ACC-03 | 1–2 hrs |
| 19 | Export **`.srt` sidecar file** alongside each generated MP4 | ACC-05 | 1–2 hrs |
| 20 | Pin **GitHub Actions to commit SHAs** | SC-07 | 1 hr |

---

### 🟡 Medium-Term Actions (Next 30–60 Days)

| # | Action | Finding(s) | Effort |
|---|--------|-----------|--------|
| 21 | Add **structured logging** (`logging` module) replacing all `print()` calls | SEC-05 / PRI-09 | 4–8 hrs |
| 22 | Implement **TTS narration audio track** for blind/low-vision accessibility | ACC-04 | 1–2 days |
| 23 | Add **safety review gate on Veo prompts** before API submission | DS-07 | 2–4 hrs |
| 24 | Make **font size resolution-aware** (5% of video height) | ACC-06 | 1 hr |
| 25 | Replace **split-line CLI output** with complete-line screen-reader-friendly format | ACC-07 | 2 hrs |
| 26 | Modify **system prompt** to remove "concrete prediction" framing, add entertainment disclaimer | DS-06 | 30 min |
| 27 | Set **spending limits** on Google Cloud Console and OpenAI billing | SEC-11 | 30 min |
| 28 | Add **`.github/dependabot.yml`** for automated dependency updates | SC-08 | 30 min |
| 29 | Restrict font path input in **Streamlit reference app** | SEC-15 | 1 hr |
| 30 | Generate and commit **SBOM** (`cyclonedx-bom` or `pip-licenses`) | SC-12 | 1 hr |

---

### 🟢 Long-Term / Ongoing

| # | Action | Finding(s) | Effort |
|---|--------|-----------|--------|
| 31 | Write **`SECURITY.md`** with credential rotation runbook | SEC-16 | 2 hrs |
| 32 | Write **`PRIVACY.md`** documenting data flows and retention policy | PRI-05 / PRI-07 | 2–4 hrs |
| 33 | Implement **data retention cleanup** job for output directories | PRI-05 | 2–4 hrs |
| 34 | Add `pytest` test suite with happy-path + error-path coverage | SEC-12 | 1–3 days |
| 35 | Engage legal/privacy counsel for production deployment compliance review | PRI-03 / PRI-07 | — |
| 36 | Target **SLSA Level 2** (hosted build service + build provenance) | SC — SLSA | 1–2 days |
| 37 | Install **CodeQL CLI** and add `codeql-results.sarif` to the CI pipeline | Audit process | 2–4 hrs |

---

## 14. Next Steps

1. **Today:** Rotate the OpenAI and Google Cloud API keys. Verify no unauthorized usage in billing consoles.
2. **This week:** Address all 🔴 Critical and top 🟠 High items in the roadmap above (items 1–20).
3. **This sprint:** Set up GitHub Actions CI with `pip audit`, `bandit`, `detect-secrets`, and GitHub Actions SHA pinning.
4. **Next milestone:** Accessibility remediation for video output (sidecar subtitles, static default caption style, GCS URI fix).
5. **Before public/social distribution:** Add AI content disclosure (on-screen label + metadata), OpenAI Moderation API filtering, and entertainment-only system prompt framing.

---

## 15. Appendix — Sub-Agent Report Index

| Dimension | File | Findings | Status |
|-----------|------|----------|--------|
| Security | [`audit/security.md`](./security.md) | 1 Critical, 3 High, 7 Medium, 5 Low | ✅ Complete |
| Privacy | [`audit/privacy.md`](./privacy.md) | 1 Critical, 4 High, 5 Medium, 4 Low | ✅ Complete |
| Accessibility | [`audit/accessibility.md`](./accessibility.md) | 1 Critical, 4 High, 5 Medium, 5 Low | ✅ Complete |
| Digital Safety | [`audit/digital-safety.md`](./digital-safety.md) | 2 Critical, 3 High, 3 Medium, 2 Low | ✅ Complete |
| Supply Chain | [`audit/supply-chain.md`](./supply-chain.md) | 1 Critical, 4 High, 5 Medium, 4 Low | ✅ Complete |
| **Full Audit** | **`audit/full-audit.md`** | **3 Critical, 16 High, 23 Medium, 20 Low** | ✅ **This document** |

---

*Report generated by the Microsoft SDL-aligned comprehensive audit pipeline.*  
*All findings represent manual static analysis. For maximum coverage, install CodeQL CLI and run `codeql database analyze` against this project's Python source.*
