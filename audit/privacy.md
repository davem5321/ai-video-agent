# Privacy Audit Report — AI Video Agent
**Audit Date:** 2025-07-14  
**Auditor:** Privacy Audit Agent (Microsoft SDL-aligned)  
**Codebase:** `C:\Users\davem\repos\ai-video-agent`  
**Audit Methodology:** Microsoft SDL Privacy Framework + GDPR/CCPA Principles  
**Skill References:** `.github/skills/privacy-impact-assessment/SKILL.md`, `.github/skills/microsoft-sdl/SKILL.md`

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview & Tech Stack](#project-overview--tech-stack)
3. [Data Inventory Table](#data-inventory-table)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Third-Party Data Sharing](#third-party-data-sharing)
6. [Findings (Prioritized)](#findings-prioritized)
7. [Compliance Gap Analysis](#compliance-gap-analysis)
8. [Privacy Principles Checklist](#privacy-principles-checklist)
9. [User Rights Assessment](#user-rights-assessment)
10. [Recommendations Summary](#recommendations-summary)
11. [Conclusion](#conclusion)

---

## Executive Summary

This report documents the findings of a structured privacy impact assessment (PIA) conducted against the **AI Video Agent** — a Python pipeline that generates daily horoscope videos using OpenAI GPT models for script generation and Google Vertex AI Veo for AI video synthesis.

**Total Findings: 14**

| Severity | Count |
|----------|-------|
| 🔴 Critical | 1 |
| 🟠 High | 4 |
| 🟡 Medium | 5 |
| 🟢 Low | 4 |

The most significant finding is a **Critical** credential exposure issue: live API keys for both OpenAI and Google Cloud are stored as plaintext in the `.env` file on disk. Additionally, the Google API key is transmitted as a URL query parameter on every API call, meaning it is written into Google's server-side access logs permanently.

The pipeline itself processes **no direct user PII** — it generates astrology content (sign + date) with no personal data collection from consumers. However, the infrastructure secrets that power the pipeline carry HBI (High Business Impact) sensitivity and require immediate remediation.

---

## Project Overview & Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM API | OpenAI (`openai` SDK) — GPT-4o, GPT-5 family |
| Video API | Google Vertex AI Veo (`requests` + direct REST) |
| Video Processing | FFmpeg (via `subprocess`) |
| Config Management | `python-dotenv` (`.env` file) |
| Data Storage | Local filesystem (`data/`, `out*/`) |
| Authentication | API keys (OPENAI_API_KEY, GOOGLE_API_KEY) |
| Package Metadata | `pyproject.toml` |

**External Services Receiving Data:**
- **OpenAI API** (`api.openai.com`) — receives date + zodiac sign prompts; returns horoscope text
- **Google Vertex AI** (`{region}-aiplatform.googleapis.com`) — receives video generation prompts; returns video files

---

## Data Inventory Table

| Data Element | Collection Point | Stored Where | Transmitted To | Retention | Sensitivity | GDPR Personal Data? |
|---|---|---|---|---|---|---|
| `OPENAI_API_KEY` (live credential) | `.env` file | `.env` on local disk | OpenAI via `Authorization` header | Indefinite | **HBI** | No (credential, not personal data) |
| `GOOGLE_API_KEY` (live credential) | `.env` file | `.env` on disk; appended to every API URL | Google server-side logs (URL query param) | Indefinite (in Google logs) | **HBI** | No (credential) |
| `GOOGLE_PROJECT_ID` | `.env` file | `.env`, `out/manifest.json`, `out_test/manifest.json`, API URLs | Google Vertex AI | Indefinite | **MBI** | No |
| Google Operation IDs (UUIDs) | API response | `out/manifest.json`, `out_test/manifest.json` | Google (origin) | Indefinite | **MBI** | No |
| Horoscope generation prompts (date + sign) | `horoscope_writer.py` L.69–74 | Sent transiently to OpenAI | OpenAI API | Per OpenAI data retention policy | **LBI** | No |
| LLM-generated horoscope text | `horoscope_writer.py` L.88–89 | `data/horoscopes/{date}/*.txt` + `horoscopes.json` | Google Vertex AI (as video prompt component) | Indefinite (local) | **LBI** | No |
| Video generation prompts | `veo_horoscope_pipeline.py` L.427 | `out*/prompts/*.txt` | Google Vertex AI | Indefinite (local) | **LBI** | No |
| Generated video files (`*.mp4`) | Google Veo API response | `out*/renders/` | None (local only) | Indefinite (local) | **LBI** | No |
| Job manifest metadata | `veo_horoscope_pipeline.py` L.488–503 | `out*/manifest.json` | None (local only) | Indefinite (local) | **MBI** | No |
| Developer name + email | `pyproject.toml` L.12–13 | Git repository (committed) | Public (GitHub) | Indefinite | **MBI** | **Yes** — name + work email |
| API error details | `horoscope_writer.py` L.95; `veo_horoscope_pipeline.py` L.103–105 | stdout (console only) | None | Session (not persisted) | **MBI** | No |

**Sensitivity Tier Definitions:**
- **HBI (High Business Impact):** Credentials, API keys, direct identifiers — requires strongest protection
- **MBI (Medium Business Impact):** Pseudonymous or operational identifiers — requires moderate protection
- **LBI (Low Business Impact):** Non-personal, public, or anonymized data — standard protection

---

## Data Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     AI VIDEO AGENT — DATA FLOWS                          ║
╚══════════════════════════════════════════════════════════════════════════╝

 Operator/Developer
        │
        ▼
  ┌─────────────┐
  │  .env file   │  ← OPENAI_API_KEY (HBI)
  │  (local disk)│  ← GOOGLE_API_KEY (HBI)
  │              │  ← GOOGLE_PROJECT_ID (MBI)
  └──────┬───────┘
         │  python-dotenv loads at runtime
         ▼
  ┌─────────────────────────────────────────────┐
  │         horoscope_writer.py                  │
  │                                              │
  │  Input: date + zodiac sign (LBI)             │
  │  API call → OpenAI Chat Completions          │─────────────────────┐
  │  Output: 12x horoscope texts (LBI)           │     TRUST BOUNDARY  │
  └─────────────────────────────────────────────┘                      │
         │                                                              │
         │ saves to                                             ┌───────▼──────────┐
         ▼                                                      │   OpenAI API     │
  ┌──────────────────────────┐                                 │  api.openai.com  │
  │  data/horoscopes/{date}/ │                                 │                  │
  │    horoscopes.json (LBI) │                                 │ Receives:        │
  │    {Sign}.txt     (LBI)  │                                 │  - Date          │
  └──────────────────────────┘                                 │  - Zodiac sign   │
                                                               │  - System prompt │
  ┌─────────────────────────────────────────────┐             └──────────────────┘
  │      veo_horoscope_pipeline.py               │
  │                                              │
  │  Input: horoscope texts (LBI)                │
  │  Builds: video prompt (LBI)                  │
  │  Saves: out*/prompts/{Sign}.txt              │
  │                                              │
  │  API call → Google Vertex AI Veo             │─────────────────────┐
  │  !⚠ API KEY IN URL QUERY PARAM (HBI leak)!  │     TRUST BOUNDARY  │
  └─────────────────────────────────────────────┘                      │
         │                                                              │
         │ downloads video                                     ┌────────▼─────────┐
         ▼                                                     │ Google Vertex AI  │
  ┌─────────────────────────────┐                             │ {region}-         │
  │   out*/renders/{sign}_raw.mp4│                            │ aiplatform.       │
  └────────────┬────────────────┘                             │ googleapis.com    │
               │ FFmpeg subtitle burn                         │                   │
               ▼                                              │ Receives:         │
  ┌─────────────────────────────┐                            │  - Video prompt   │
  │   out*/renders/{sign}.mp4   │                            │  - Render params  │
  └─────────────────────────────┘                            │  - API key (URL!) │
               │                                              │  - Project ID     │
               │ metadata                                     └──────────────────┘
               ▼
  ┌─────────────────────────────┐
  │   out*/manifest.json        │ ← Contains GCP Project ID + Operation UUIDs (MBI)
  └─────────────────────────────┘

 Legend:
   ── Standard data flow
   !⚠ Security/Privacy risk at this boundary
   HBI = High Business Impact (credentials)
   MBI = Medium Business Impact (operational IDs)
   LBI = Low Business Impact (generated content)
```

---

## Third-Party Data Sharing

| Third Party | Data Shared | Purpose | Data Sent | Legal Mechanism | DPA Referenced? |
|---|---|---|---|---|---|
| **OpenAI** (`api.openai.com`) | Horoscope prompts (date + zodiac sign), system prompt text | LLM text generation | Non-personal (date, sign) | Legitimate interest / Contract | ❌ Not documented |
| **Google Vertex AI** (`aiplatform.googleapis.com`) | Video generation prompts (LBI), GOOGLE_PROJECT_ID, GOOGLE_API_KEY (in URL) | AI video generation | Non-personal content + credentials | Legitimate interest / Contract | ❌ Not documented |
| **Google Server Logs** (implicit) | GOOGLE_API_KEY (via URL query parameter) | Automatic server logging | HBI credential | Unintentional — no legal basis | ❌ N/A |

> ⚠️ **Key risk**: The `GOOGLE_API_KEY` is appended as a query parameter (`?key=...`) on every API call, causing it to be written into Google's server-side access logs. This is a credential transmission pattern that permanently records the credential in logs outside the developer's control.

---

## Findings (Prioritized)

---

### 🔴 [CRITICAL] — Live API Credentials Stored as Plaintext in `.env` File

- **SDL Phase:** Implementation
- **File:** `.env` (lines 2, 20)
- **Category:** Credential Exposure / Data Storage
- **Data Elements Affected:** `OPENAI_API_KEY` (sk-proj-…), `GOOGLE_API_KEY` (AQ.Ab8…)
- **Description:**  
  The `.env` file on disk contains live, production-grade API credentials for both OpenAI and Google Cloud in plaintext. While the `.env` file is correctly listed in `.gitignore` (preventing it from being committed to the Git repository under normal operations), plaintext credential files on disk represent a significant risk vector:
  - **Accidental commit exposure**: A `git add -f .env`, misconfigured `.gitignore`, or IDE auto-stage could expose keys to the remote repository.
  - **Backup and sync exposure**: Developer machines that sync to cloud storage (OneDrive, iCloud, Dropbox) or backup solutions could exfiltrate these credentials.
  - **Filesystem access**: Any process with filesystem read access (malware, rogue dependency, misconfigured permissions) can harvest these keys.
  - **Credential rotation**: Live credentials committed to or stored on shared machines have no automatic expiry enforcement.  
  Additionally, the file contains the `GOOGLE_PROJECT_ID=veo-3-videos-482006`, which combined with the API key allows full resource control.

- **Regulatory Reference:** GDPR Art. 32 (security of processing); Microsoft SDL — "Hardcoded secrets in source" is an explicitly banned pattern; OWASP A07:2021 — Identification and Authentication Failures
- **Recommendation:**
  1. **Immediately rotate** both the `OPENAI_API_KEY` and `GOOGLE_API_KEY` — treat them as compromised since they exist in a readable file outside of a secrets vault.
  2. Replace plaintext `.env` usage with a **secrets manager** (Azure Key Vault, HashiCorp Vault, AWS Secrets Manager, or `keyring` for local dev).
  3. For local development, use OS-level credential stores (e.g., `keyring`, macOS Keychain, Windows Credential Manager) rather than plaintext files.
  4. Add a **pre-commit hook** (e.g., `detect-secrets`, `git-secrets`, `trufflehog`) to scan for accidental credential commits.
  5. Consider scoping API keys to minimum required permissions and adding IP restrictions in the OpenAI and Google Cloud consoles.
  6. Document key rotation procedures in the project README.

---

### 🟠 [HIGH] — Google API Key Transmitted as URL Query Parameter (Logged by Google Servers)

- **SDL Phase:** Design / Implementation
- **File:** `src/render/veo_horoscope_pipeline.py` (lines 51–54, 119–122)
- **Category:** Data Transmission / Credential Exposure
- **Data Elements Affected:** `GOOGLE_API_KEY` (HBI)
- **Description:**  
  Both the video submission URL and the polling URL are constructed with the API key appended as a query parameter:
  ```python
  # Line 51-54 — submit
  url = (f"https://{self.region}-aiplatform.googleapis.com/v1/"
         f"projects/{self.project_id}/locations/{self.region}/"
         f"publishers/google/models/{self.model_id}:predictLongRunning"
         f"?key={self.api_key}")  # ← API key in URL
  
  # Line 119-122 — poll
  fetch_url = (f"https://{self.region}-aiplatform.googleapis.com/v1/"
               f"projects/{self.project_id}/locations/{self.region}/"
               f"publishers/google/models/{model_id}:fetchPredictOperation"
               f"?key={self.api_key}")  # ← API key in URL
  ```
  Query parameters are logged by:
  - **Google's server-side access logs** (permanent, outside developer control)
  - **Network proxy and monitoring tools** (Fiddler, Charles, Wireshark, corporate proxies)
  - **Application Performance Monitoring (APM)** tools that capture outbound URLs
  - **Debugging output** from the `requests` library if verbose logging is enabled
  
  This means the `GOOGLE_API_KEY` is written into Google's infrastructure logs every single time the pipeline runs, where it persists beyond the developer's ability to control or purge.
  
- **Regulatory Reference:** GDPR Art. 32; Microsoft SDL — credential transmission security; OWASP API3:2023 — Broken Object Property Level Authorization
- **Recommendation:**
  1. Remove the `?key=` query parameter from all URLs.
  2. Pass the API key as an HTTP `Authorization` header instead:
     ```python
     headers = {
         "Content-Type": "application/json",
         "Authorization": f"Bearer {self.api_key}"
     }
     r = requests.post(url, json=payload, headers=headers, timeout=60)
     ```
  3. Alternatively, use Google's official `google-auth` library (already in `requirements.txt`) with Application Default Credentials (ADC) to avoid managing raw API keys altogether:
     ```python
     import google.auth
     import google.auth.transport.requests
     credentials, project = google.auth.default()
     credentials.refresh(google.auth.transport.requests.Request())
     headers["Authorization"] = f"Bearer {credentials.token}"
     ```
  4. Audit whether Google's access logs can be purged or restricted to reduce the blast radius of previously logged keys.

---

### 🟠 [HIGH] — Google Cloud Project ID and Operation IDs Persisted in Manifest Files Without Access Controls

- **SDL Phase:** Design / Implementation
- **File:** `out/manifest.json` (line 7), `out_test/manifest.json` (line 7)
- **Category:** Data Storage / Information Disclosure
- **Data Elements Affected:** `GOOGLE_PROJECT_ID` (`veo-3-videos-482006`), Google Cloud operation UUIDs (MBI)
- **Description:**  
  The pipeline writes a `manifest.json` file to each output directory after a run. These files contain:
  - The full Google Cloud operation name (e.g., `projects/veo-3-videos-482006/locations/us-central1/publishers/google/models/veo-3.1-fast-generate-001/operations/{uuid}`)
  - The Google Cloud Project ID
  - Operation UUIDs that could be used to query job status
  - Timestamps and model identifiers
  
  While the output directories (`out/`, `out_test/`, `out_real/`) are listed in `.gitignore`, relying solely on `.gitignore` as the access control mechanism for sensitive operational metadata is insufficient. These files accumulate indefinitely on disk with no retention limit, and the project ID embedded in them (combined with any leaked credential) could enable unauthorized access to the Google Cloud project.
  
- **Regulatory Reference:** GDPR Art. 5(1)(e) — storage limitation; Microsoft SDL — data minimization
- **Recommendation:**
  1. Strip the full operation name from manifest files after job completion; retain only the short operation UUID.
  2. Implement a manifest retention policy (e.g., delete manifests older than 30 days via `make clean` or a scheduled task).
  3. Apply restrictive filesystem permissions (`chmod 600`) to the output directories on Unix/macOS systems.
  4. Consider storing the project ID only in the `.env` file and removing it from serialized output.

---

### 🟠 [HIGH] — No Data Retention Policy or Automated Deletion Mechanism

- **SDL Phase:** Design / Requirements
- **File:** `src/write/horoscope_writer.py` (lines 104–123), `src/render/veo_horoscope_pipeline.py` (lines 488–506), `Makefile`
- **Category:** Data Retention / Compliance
- **Data Elements Affected:** All generated content in `data/horoscopes/`, `out*/prompts/`, `out*/renders/`, `out*/manifest.json`
- **Description:**  
  The pipeline accumulates files indefinitely:
  - `data/horoscopes/{date}/` — LLM-generated horoscope text, persisted per execution
  - `out*/prompts/*.txt` — Video generation prompts, one file per zodiac sign per run
  - `out*/renders/*.mp4` — Generated video files (can be hundreds of MB each)
  - `out*/manifest.json` — Job metadata including operational IDs
  
  There is no automated retention policy, no TTL on stored data, and no deletion mechanism beyond the manual `make clean` target (which only removes `out*/` directories, not `data/horoscopes/`). As the pipeline runs daily, the `data/horoscopes/` directory will grow continuously with no bound.
  
  While the current content is LBI (no personal data in the generated horoscopes), the pattern of indefinite retention creates a precedent that is problematic if the pipeline is ever extended to process personal data. Storage limitation is a foundational GDPR principle (Art. 5(1)(e)) and applies here as operational hygiene.
  
- **Regulatory Reference:** GDPR Art. 5(1)(e) — storage limitation; Microsoft SDL — data minimization; general best practice
- **Recommendation:**
  1. Define and document a retention period for each data category (e.g., horoscope text: 7 days; rendered videos: 30 days; manifests: 14 days).
  2. Extend the `make clean` target to also purge old dated directories from `data/horoscopes/` beyond the defined retention window.
  3. Implement an automated cleanup function in the pipeline (e.g., at pipeline start, delete `data/horoscopes/` subdirectories older than N days).
  4. Add a `--retain-days` CLI flag to control retention at runtime.
  5. Document the retention policy in `README.md`.

---

### 🟠 [HIGH] — No Data Processing Agreements (DPAs) or Third-Party Privacy Disclosures Documented

- **SDL Phase:** Requirements / Release
- **File:** `README.md`, `pyproject.toml`
- **Category:** Compliance / Third-Party Data Sharing
- **Data Elements Affected:** All data transmitted to OpenAI and Google Vertex AI
- **Description:**  
  The pipeline transmits operational data (prompts, render parameters, API metadata) to two major AI service providers:
  - **OpenAI**: Receives text prompts via the Chat Completions API. OpenAI's standard API terms include a data processing addendum, but there is no documentation in this project confirming that a DPA is in place, nor any reference to OpenAI's data retention settings (e.g., zero-data-retention tier).
  - **Google Vertex AI**: Receives video generation prompts and operational parameters. Google Cloud's Data Processing Amendment applies by default when using Google Cloud services, but this is not documented.
  
  The absence of documentation means:
  - Developers joining the project have no visibility into what data is shared with third parties.
  - There is no record that the applicable DPAs were reviewed and accepted.
  - If any prompt content is inadvertently enriched with personal data in the future, there would be no framework in place to handle it compliantly.
  
- **Regulatory Reference:** GDPR Art. 28 — Processor obligations; GDPR Art. 13/14 — Transparency; CCPA §1798.100 — Consumer rights notice
- **Recommendation:**
  1. Add a `PRIVACY.md` or `DATA_PROCESSING.md` file documenting: (a) what data is sent to each third party, (b) the legal basis for doing so, (c) links to the applicable DPAs/Terms, and (d) data retention settings configured at those services.
  2. Enable **zero data retention** on the OpenAI API (via API settings or by setting `store: false` in requests) since horoscope prompts do not need to be retained by OpenAI.
  3. Review and explicitly accept Google Cloud's Data Processing Amendment in the Google Cloud Console.
  4. Document these decisions in the project's security/privacy artifacts.

---

### 🟡 [MEDIUM] — Developer PII (Name and Work Email) in Committed `pyproject.toml`

- **SDL Phase:** Implementation
- **File:** `pyproject.toml` (lines 12–13)
- **Category:** PII Handling / Information Disclosure
- **Data Elements Affected:** Developer full name (`David Moore`), Microsoft work email (`davidmo@microsoft.com`)
- **Description:**  
  The `pyproject.toml` file contains the developer's personal information in the `authors` field:
  ```toml
  authors = [
      {name = "David Moore", email = "davidmo@microsoft.com"}
  ]
  ```
  This file is committed to the Git repository and published on the public GitHub repository referenced in the `[project.urls]` section (`https://github.com/davem5321/ai-video-agent`). The email address (`davidmo@microsoft.com`) is a Microsoft corporate email and qualifies as personal data under GDPR (Art. 4(1)).
  
  Risks include:
  - **Spam and phishing**: Corporate email addresses harvested from public repositories are targeted by automated phishing campaigns.
  - **GDPR Art. 17 erasure complexity**: Once the email is indexed by package registries (PyPI) or code search engines, removal is not guaranteed even if the repository is updated.
  - **Social engineering surface**: Combining the name, employer, and repository context gives threat actors a targeted profile.
  
- **Regulatory Reference:** GDPR Art. 4(1), Art. 5(1)(c) — data minimization; Art. 17 — right to erasure
- **Recommendation:**
  1. Replace the work email with a **non-personal contact** (e.g., a team alias, a GitHub noreply email `github-noreply@`, or a project-specific address).
  2. If listing a personal name is required for attribution, consider using an alias or GitHub username instead.
  3. Review whether this package will be published to PyPI; if so, treat the author metadata as public information with no path to erasure.

---

### 🟡 [MEDIUM] — API Error Details Logged to stdout Without Sanitization

- **SDL Phase:** Implementation
- **File:** `src/write/horoscope_writer.py` (line 95); `src/render/veo_horoscope_pipeline.py` (lines 100–105)
- **Category:** PII Handling / Logging
- **Data Elements Affected:** API error messages (may contain account-level details, billing status, quota thresholds)
- **Description:**  
  Error handling in `horoscope_writer.py` prints the full exception string directly to stdout:
  ```python
  # horoscope_writer.py line 95
  print(f"       ERROR: {str(e)}")
  ```
  Similarly, in `veo_horoscope_pipeline.py`:
  ```python
  # lines 103-105
  error_data = e.response.json()
  print(f"   Error details: {error_data}")
  ```
  API error responses from OpenAI and Google can contain:
  - **Account identifiers**: Organization IDs, project IDs, user references
  - **Billing information**: Quota limits, tier information, usage thresholds
  - **Rate limit context**: Headers that identify the account's usage tier
  - **Request IDs**: Traceable back to specific API calls in vendor logs
  
  Printing this information to stdout without sanitization means it can appear in CI/CD logs, terminal recordings, log aggregation tools, or screen-sharing sessions where it is visible to unintended parties.
  
- **Regulatory Reference:** GDPR Art. 32 — security of processing; Microsoft SDL — "Never log PII to application logs"; General best practice
- **Recommendation:**
  1. Replace `print()` calls with a structured logging framework (`import logging`), which allows log-level filtering and redirection.
  2. Sanitize error messages before logging — strip API keys, account IDs, and billing-related fields from error responses before printing.
  3. In production use, configure the logger to write to a file with appropriate access controls rather than stdout.
  4. Consider using a dedicated error class that strips sensitive fields from the error context before surfacing them.

---

### 🟡 [MEDIUM] — FFmpeg stderr Propagated Unsanitized in Error Return Values

- **SDL Phase:** Implementation
- **File:** `src/render/subtitle_utils.py` (line 303)
- **Category:** Information Disclosure / Logging
- **Data Elements Affected:** System file paths, FFmpeg operational details, temporary file names
- **Description:**  
  When FFmpeg processing fails, the full `stderr` output is returned in the error tuple:
  ```python
  # subtitle_utils.py line 303
  if result.returncode != 0:
      return False, f"FFmpeg failed: {result.stderr}"
  ```
  FFmpeg's `stderr` output typically contains:
  - **Full system paths** to temporary files (e.g., `C:\Users\davem\AppData\Local\Temp\tmpXXXXXX.ass`)
  - **System codec library paths** that reveal OS and software installation details
  - **Detailed video metadata** from the source file
  - **Hardware/driver information** in some configurations
  
  This string is passed up the call stack and printed by the pipeline, potentially exposing internal system paths in user-facing output or logs. In the `reference/app.py` Streamlit UI (line 141, 170), FFmpeg stderr is rendered directly in the browser UI (`st.code(result.stderr)`), which would expose system paths to web users if that pattern were adopted in a deployed application.
  
- **Regulatory Reference:** Microsoft SDL — information disclosure; General best practice — principle of least exposure
- **Recommendation:**
  1. Truncate or summarize FFmpeg error output rather than returning the full `stderr` string:
     ```python
     # Return only the last 500 characters of stderr, or the first error line
     error_summary = result.stderr.strip().splitlines()[-5:] if result.stderr else []
     return False, f"FFmpeg failed (exit {result.returncode}): {'; '.join(error_summary)}"
     ```
  2. Log the full stderr at `DEBUG` level to a controlled log file rather than surfacing it in return values.
  3. Never render raw FFmpeg stderr in user-facing UI (flag the pattern in `reference/app.py` for avoidance).

---

### 🟡 [MEDIUM] — No Privacy Notice or AI Content Disclosure for End Consumers

- **SDL Phase:** Requirements / Design
- **File:** `README.md`, project-wide
- **Category:** Consent / Compliance / Transparency
- **Data Elements Affected:** Generated video content distributed to end users on social platforms
- **Description:**  
  The pipeline is explicitly designed to produce content for social media distribution (TikTok/Reels: 9:16, YouTube: 16:9, Instagram: 1:1 aspect ratios). The generated videos will be consumed by end users on these platforms with no indication that:
  - The horoscope text was generated by an AI (OpenAI GPT)
  - The video was generated by an AI (Google Veo)
  - The content was produced by an automated pipeline
  
  Regulatory and platform-specific requirements are emerging around AI-generated content disclosure:
  - **EU AI Act (2024)**: Deep fake and AI-generated content must be labeled when targeting consumers.
  - **FTC Guidance (US)**: AI-generated endorsements or advice must be disclosed.
  - **Platform Policies**: TikTok, YouTube, and Instagram have varying AI-content disclosure policies.
  - **CCPA**: If California residents interact with content, there may be disclosure obligations.
  
  Additionally, there is no privacy policy documenting how operator data (API keys, configurations) is handled, nor any terms of service for the tool.
  
- **Regulatory Reference:** EU AI Act Art. 50; FTC AI disclosure guidance; GDPR Art. 13/14 (transparency); CCPA §1798.100
- **Recommendation:**
  1. Add "AI-generated" metadata or watermark to output videos using FFmpeg's metadata injection capability.
  2. Include a disclosure statement in the README about AI content labeling requirements for social media distribution.
  3. Add an `AI_GENERATED=true` metadata tag to output MP4 files (C2PA content credentials standard is recommended).
  4. Create a minimal `PRIVACY.md` documenting the project's data practices for operators.

---

### 🟡 [MEDIUM] — No Explicit TLS Version Enforcement on Outbound HTTP Clients

- **SDL Phase:** Implementation
- **File:** `src/render/veo_horoscope_pipeline.py` (lines 87, 132); `src/write/horoscope_writer.py` (line 88)
- **Category:** Data Transmission / Encryption in Transit
- **Data Elements Affected:** All data transmitted to OpenAI and Google Vertex AI
- **Description:**  
  The `requests` library (used for Google API calls) and the `openai` SDK (used for OpenAI calls) are initialized without explicit TLS version configuration. Both libraries default to the system SSL/TLS settings, which may allow TLS 1.0 or 1.1 connections on older systems or in certain Python environments.
  
  While the URLs use `https://` (enforcing HTTPS), the actual TLS protocol version negotiated depends on:
  - The Python installation's OpenSSL version
  - The system's SSL/TLS configuration
  - Whether the `certifi` certificate bundle is current
  
  Microsoft SDL requires TLS 1.2+ for all data in transit. Both OpenAI and Google support TLS 1.2+, but the client does not enforce this minimum.
  
- **Regulatory Reference:** GDPR Art. 32 — encryption in transit; Microsoft SDL — TLS 1.2+ requirement; NIST SP 800-52 Rev. 2
- **Recommendation:**
  1. Configure the `requests` session with an explicit TLS 1.2 minimum:
     ```python
     import ssl
     import requests
     from requests.adapters import HTTPAdapter
     from urllib3.util.ssl_ import create_urllib3_context
     
     class TLSAdapter(HTTPAdapter):
         def init_poolmanager(self, *args, **kwargs):
             ctx = create_urllib3_context(ssl_version=ssl.PROTOCOL_TLS_CLIENT)
             ctx.minimum_version = ssl.TLSVersion.TLSv1_2
             kwargs['ssl_context'] = ctx
             return super().init_poolmanager(*args, **kwargs)
     
     session = requests.Session()
     session.mount("https://", TLSAdapter())
     ```
  2. Pin the `certifi` package version in `requirements.txt` to ensure a current CA certificate bundle is used.
  3. Consider adding certificate pinning for the Google API endpoint in high-security deployments.

---

### 🟢 [LOW] — No Path Validation on `--out` CLI Argument

- **SDL Phase:** Implementation
- **File:** `src/render/veo_horoscope_pipeline.py` (lines 541, 626)
- **Category:** Input Validation / Data Storage
- **Data Elements Affected:** Output directory path; all generated files
- **Description:**  
  The `--out` argument is accepted as a raw string and passed directly to `Path()` without sanitization:
  ```python
  p.add_argument("--out", type=str, default="./out", help="Output directory")
  # ...
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  ```
  A maliciously crafted path (e.g., `--out /etc/cron.d/` or `--out ../../sensitive_dir`) could create directories in unexpected system locations. The `parents=True` flag in `mkdir()` amplifies this by allowing creation of intermediate directories. While this requires local access to execute, it represents a path traversal risk in automated or CI/CD environments where the argument may be constructed programmatically.
  
- **Regulatory Reference:** Microsoft SDL — input validation for file system operations; OWASP Path Traversal
- **Recommendation:**
  1. Validate that the output path resolves to a location within an expected base directory:
     ```python
     import os
     base_dir = Path.cwd()
     out_dir = (base_dir / Path(args.out)).resolve()
     if not str(out_dir).startswith(str(base_dir)):
         raise ValueError(f"Output path escapes working directory: {out_dir}")
     ```
  2. Add a whitelist of allowed output directory prefixes (e.g., `./out`, `./out_test`, `./out_real`).

---

### 🟢 [LOW] — No Disclosure of Third-Party Telemetry Collected by AI Service SDKs

- **SDL Phase:** Requirements / Design
- **File:** `requirements.txt`, `src/write/horoscope_writer.py` (line 38), `src/render/veo_horoscope_pipeline.py`
- **Category:** Compliance / Telemetry / Transparency
- **Data Elements Affected:** Usage telemetry collected by OpenAI and Google SDKs
- **Description:**  
  The `openai` Python SDK and `google-api-python-client` / `google-auth` packages may collect telemetry about SDK usage (versions, request counts, error rates) as part of their standard operation. Neither the project documentation nor the codebase acknowledges or discloses this:
  - **OpenAI SDK**: Collects request metadata tied to the API key; this usage data is retained per OpenAI's privacy policy.
  - **Google Client Libraries**: May send telemetry via the `x-goog-request-reason` header and standard Google usage analytics.
  
  Operators deploying this pipeline in an enterprise context may need to account for these data flows in their own privacy and security compliance posture.
  
- **Regulatory Reference:** GDPR Art. 13/14 — transparency about data flows; CCPA disclosure requirements
- **Recommendation:**
  1. Add a section to `README.md` and/or `PRIVACY.md` documenting that OpenAI and Google SDKs collect usage telemetry, with links to their respective privacy policies.
  2. Review OpenAI's `store` parameter and set `store=False` in API calls to opt out of prompt storage where supported.
  3. Investigate whether Google's client libraries can be configured for reduced telemetry in enterprise deployments.

---

### 🟢 [LOW] — LLM-Generated Output Passed to Secondary API Without Output Validation

- **SDL Phase:** Implementation
- **File:** `src/render/veo_horoscope_pipeline.py` (lines 384–391, 421)
- **Category:** Data Transmission / AI/ML-Specific Risk
- **Data Elements Affected:** LLM-generated horoscope text (LBI) used as Veo video prompt component
- **Description:**  
  Horoscope text generated by OpenAI is used directly as a caption component in the video generation prompt sent to Google Veo, without intermediate validation or sanitization:
  ```python
  # veo_horoscope_pipeline.py lines 384-391
  def build_scene(self, sign: str, text: str, template: str = DEFAULT_TEMPLATE) -> SceneSpec:
      prompt = template.format(
          aspect=self.render.aspect_ratio,
          sign=sign,
          caption=text,  # ← raw LLM output injected into prompt template
          seconds=self.render.seconds,
      )
  ```
  While current horoscope content is benign (LBI), LLM outputs can occasionally include unexpected content (hallucinations, prompt injection artifacts, inappropriate content). Passing unvalidated LLM output as a prompt to a second generative AI system creates an **LLM-to-LLM prompt injection chain** where unexpected content from OpenAI could influence Veo's video generation in unintended ways.
  
- **Regulatory Reference:** Microsoft SDL — input validation; OWASP LLM Top 10 (LLM02: Insecure Output Handling)
- **Recommendation:**
  1. Add a lightweight content validation step between OpenAI output and Veo prompt construction — check for unexpected length, disallowed keywords, or formatting anomalies.
  2. Cap the maximum length of the horoscope text that can be injected into the video prompt (e.g., 500 characters).
  3. Sanitize the LLM output to remove special characters that could be interpreted as prompt control sequences.

---

### 🟢 [LOW] — No Formal Audit Log for API Operations

- **SDL Phase:** Design / Implementation
- **File:** `src/render/veo_horoscope_pipeline.py`, `src/write/horoscope_writer.py`
- **Category:** Compliance / Non-Repudiation
- **Data Elements Affected:** API call metadata (timestamps, model IDs, operation IDs, success/failure status)
- **Description:**  
  All operational logging is performed via `print()` statements to stdout with no structured format, no log levels, no log rotation, and no persistent log files. Operational metadata (which models were called, when, with what parameters, success/failure status) is ephemeral — it exists only in the terminal session and is lost when the session ends.
  
  This creates a **non-repudiation gap**: there is no persistent record of what API calls were made, what content was generated, or what errors occurred. In a regulatory context, being able to demonstrate what the system did and when is an important accountability mechanism (GDPR Art. 5(2) — accountability principle).
  
  The manifest files partially fill this gap for successful Veo jobs, but there is no equivalent for OpenAI calls, failed jobs, or configuration at the time of each run.
  
- **Regulatory Reference:** GDPR Art. 5(2) — accountability; Microsoft SDL — audit logging; General best practice
- **Recommendation:**
  1. Replace `print()` with Python's `logging` module, writing structured logs to a dated log file (e.g., `logs/pipeline_{date}.log`).
  2. Log: timestamp, model used, operation ID, success/failure, duration, and error type (not full error details) for each API call.
  3. Apply log retention policy consistent with the data retention policy (e.g., 30-day log retention).
  4. Exclude all credential values from log entries (explicitly scrub `GOOGLE_API_KEY`, `OPENAI_API_KEY` from any accidental log interpolation).

---

## Compliance Gap Analysis

### GDPR Compliance

| GDPR Requirement | Article | Status | Notes |
|---|---|---|---|
| Legal basis documented for each processing activity | Art. 6 | ❌ Missing | No legal basis documented; likely "legitimate interest" but not recorded |
| Privacy notice / policy exists | Art. 13/14 | ❌ Missing | No `PRIVACY.md` or privacy policy |
| Data minimization | Art. 5(1)(c) | ⚠️ Partial | LLM/video content is minimal; developer PII in `pyproject.toml` could be reduced |
| Retention limits defined and enforced | Art. 5(1)(e) | ❌ Missing | No retention policy; data accumulates indefinitely |
| User rights fulfillable (access, erasure, portability) | Art. 15–20 | ❌ N/A | No user data collected directly; but developer PII in repo cannot be easily erased |
| Data Processing Agreements for all processors | Art. 28 | ❌ Not documented | OpenAI and Google DPAs not referenced |
| Encryption in transit (HTTPS/TLS) | Art. 32 | ⚠️ Partial | HTTPS used but TLS version not enforced; API key in URL query parameter |
| Encryption at rest for HBI data | Art. 32 | ❌ Missing | Credentials stored as plaintext in `.env` |
| AI-generated content transparency | EU AI Act Art. 50 | ❌ Missing | No AI content labeling on output videos |

### CCPA Compliance

| CCPA Requirement | Section | Status | Notes |
|---|---|---|---|
| "Do Not Sell or Share" mechanism | §1798.120 | ➖ Not applicable | No consumer data collected |
| Privacy notice with CCPA disclosures | §1798.100 | ❌ Missing | No privacy notice exists |
| Consumer rights request mechanism | §1798.105–130 | ➖ Not applicable | No consumer data collected |
| Disclosure of data sold/shared with third parties | §1798.115 | ❌ Missing | OpenAI/Google data flows not disclosed |

### General Privacy Hygiene

| Check | Status | Notes |
|---|---|---|
| No PII in log files | ✅ Pass | Logs go to stdout only; no PII in current content |
| No credentials in source code | ✅ Pass (source) ❌ Fail (disk) | `.gitignore` protects git, but `.env` has live keys |
| API keys not in URLs | ❌ Fail | `GOOGLE_API_KEY` in URL query parameter |
| Encryption at rest for HBI | ❌ Fail | Plaintext `.env` file |
| TLS 1.2+ enforced in transit | ⚠️ Partial | HTTPS used; TLS version not pinned |
| Data minimization practiced | ✅ Pass | No unnecessary data collection |
| Retention limits defined | ❌ Fail | No retention policy |
| Deletion mechanism exists | ⚠️ Partial | `make clean` covers `out*/` but not `data/horoscopes/` |
| Third-party DPAs documented | ❌ Fail | Not documented |
| AI content labeled | ❌ Fail | No output labeling |
| Pre-commit secret scanning | ❌ Fail | No hooks configured |

---

## Privacy Principles Checklist

| Principle | Status | Finding Reference |
|---|---|---|
| **Data Minimization** — only collect what's necessary | ✅ Pass | No unnecessary personal data collected |
| **Purpose Limitation** — data not reused for undisclosed purposes | ✅ Pass | All data used for video generation only |
| **Storage Limitation** — retention periods defined; deletion mechanism exists | ❌ Fail | Finding: No Data Retention Policy [HIGH] |
| **Accuracy** — users can correct inaccurate data | ➖ N/A | No user data collected |
| **Integrity & Confidentiality** — HBI/MBI encrypted in transit and at rest | ❌ Fail | Findings: Plaintext Credentials [CRITICAL], API Key in URL [HIGH], TLS [MEDIUM] |
| **Privacy by Design** — default settings are most privacy-protective | ❌ Fail | API key in URL is default; no secret manager; no retention policy |

---

## User Rights Assessment

| Right | GDPR Article | CCPA Section | Applicable? | Fulfillable? | Notes |
|---|---|---|---|---|---|
| Right to be informed | Art. 13-14 | §1798.100 | ⚠️ Partial | ❌ No | No privacy notice exists |
| Right of access | Art. 15 | §1798.110 | ❌ No | ➖ N/A | No consumer personal data collected |
| Right to rectification | Art. 16 | — | ⚠️ Partial | ❌ No | Developer PII in committed `pyproject.toml` cannot be easily corrected post-publish |
| Right to erasure | Art. 17 | §1798.105 | ⚠️ Partial | ❌ No | Developer PII in git history; API keys in Google server logs cannot be deleted |
| Right to restrict processing | Art. 18 | — | ❌ No | ➖ N/A | No consumer processing |
| Right to data portability | Art. 20 | §1798.100 | ❌ No | ➖ N/A | No consumer data |
| Right to object | Art. 21 | §1798.120 | ❌ No | ➖ N/A | No consumer data processed |

---

## Recommendations Summary

Ordered by priority:

| Priority | Finding | Action | Effort |
|---|---|---|---|
| 🔴 Critical | Plaintext credentials in `.env` | Rotate keys immediately; adopt secrets manager | Medium |
| 🟠 High | API key in URL query param | Move to `Authorization: Bearer` header or ADC | Low |
| 🟠 High | Manifest files expose GCP Project ID | Strip sensitive fields; add retention cleanup | Low |
| 🟠 High | No data retention policy | Define TTLs; extend `make clean`; document | Low |
| 🟠 High | No DPAs documented | Document OpenAI/Google DPAs; create `PRIVACY.md` | Low |
| 🟡 Medium | Developer PII in `pyproject.toml` | Replace work email with alias | Low |
| 🟡 Medium | API errors logged unsanitized | Adopt structured logging; sanitize error output | Low |
| 🟡 Medium | FFmpeg stderr in return values | Truncate/summarize stderr in error returns | Low |
| 🟡 Medium | No AI content disclosure | Add metadata/watermark to output videos | Medium |
| 🟡 Medium | TLS version not enforced | Configure `requests` session with TLS 1.2 minimum | Low |
| 🟢 Low | No path validation on `--out` | Add path traversal check | Low |
| 🟢 Low | Third-party SDK telemetry undisclosed | Document in `PRIVACY.md`; configure opt-outs | Low |
| 🟢 Low | LLM output to API without validation | Add content validation step; cap length | Low |
| 🟢 Low | No formal audit logging | Adopt `logging` module; write structured log files | Medium |

---

## Conclusion

The **AI Video Agent** is a focused content generation pipeline with a relatively narrow data footprint. The content it processes (zodiac sign labels, dates, AI-generated text) is **not personal data** in the regulatory sense, which significantly limits the GDPR/CCPA compliance surface for consumer data. The pipeline is well-structured and the `.gitignore` correctly excludes the most sensitive artifacts from version control.

However, the audit identified **one Critical and four High severity findings** that require prompt remediation:

1. **The Critical finding** — live API credentials stored as plaintext in `.env` — is the highest priority. These credentials should be considered compromised and rotated immediately, with future credentials managed through a secrets manager rather than plaintext files.

2. **The most architecturally significant High finding** — the Google API key transmitted as a URL query parameter — means the credential is permanently recorded in Google's server-side access logs on every pipeline execution. This pattern must be replaced with header-based authentication.

3. **The remaining High findings** (no retention policy, no DPA documentation, manifest files exposing GCP metadata) are straightforward to address through documentation and minor code changes, but represent meaningful compliance gaps if this project is deployed in an enterprise or regulated environment.

The Medium and Low findings are best-practice improvements that would improve the security posture and auditability of the pipeline without being regulatory blockers for the current use case.

**Recommended immediate actions (within 24 hours):**
- [ ] Rotate `OPENAI_API_KEY` and `GOOGLE_API_KEY`
- [ ] Move Google API key from URL query parameter to `Authorization` header
- [ ] Add `detect-secrets` or equivalent pre-commit hook

**Recommended short-term actions (within 2 weeks):**
- [ ] Migrate credential storage to a secrets manager or OS keychain
- [ ] Define and document data retention policy
- [ ] Create `PRIVACY.md` documenting all third-party data flows
- [ ] Adopt structured logging framework

---

*Report generated by Privacy Audit Agent | Microsoft SDL-aligned | GDPR + CCPA framework*  
*Audit scope: Full codebase static analysis — no source files were modified*  
*Files audited: `.env`, `.env.example`, `.gitignore`, `pyproject.toml`, `requirements.txt`, `Makefile`, `README.md`, `CHANGES.md`, `src/main.py`, `src/render/veo_horoscope_pipeline.py`, `src/render/subtitle_utils.py`, `src/write/horoscope_writer.py`, `data/horoscopes/`, `out/manifest.json`, `out_real/manifest.json`, `out_test/manifest.json`, `reference/app.py`, `reference/config.py`*
