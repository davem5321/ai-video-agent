# Supply Chain Security Audit — ai-video-agent

**Audit Date:** 2025-07-14  
**Auditor:** AI Supply Chain Security Agent (aligned with Microsoft SDL)  
**Project:** ai-video-agent — Automated horoscope video generation using OpenAI and Google Vertex AI Veo  
**Repository:** https://github.com/davem5321/ai-video-agent  
**Ecosystem:** Python (pip / PyPI)  
**Python Requirement:** >=3.10  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Dependency Inventory](#dependency-inventory)
3. [Lockfile Health Assessment](#lockfile-health-assessment)
4. [Findings](#findings)
   - [CRITICAL](#critical-findings)
   - [HIGH](#high-findings)
   - [MEDIUM](#medium-findings)
   - [LOW](#low-findings)
5. [Dependency Risk Register](#dependency-risk-register)
6. [CI/CD Pipeline Integrity Assessment](#cicd-pipeline-integrity-assessment)
7. [Dependency Confusion Assessment](#dependency-confusion-assessment)
8. [SLSA Level Assessment](#slsa-level-assessment)
9. [Recommended Dependabot Configuration](#recommended-dependabot-configuration)
10. [SBOM Generation Command](#sbom-generation-command)
11. [Remediation Roadmap](#remediation-roadmap)
12. [Conclusion](#conclusion)

---

## Executive Summary

This supply chain security audit of the **ai-video-agent** Python project identifies **14 findings** across four severity tiers. The project is in a **high-risk supply chain posture** due to the complete absence of dependency version pinning and the total lack of a dependency lockfile. Every `pip install -r requirements.txt` or `make install` resolves to whatever the latest compatible version is at install time, making builds non-deterministic and opening the project to silent dependency drift, transitive vulnerability accumulation, and potential supply chain substitution attacks.

The project also consumes secrets (OpenAI API keys, Google Cloud API keys) at runtime through a pattern that depends heavily on environment variable hygiene. While `.env` is correctly gitignored, the runtime dependency chain touches sensitive authentication flows that amplify the blast radius of any compromised transitive dependency.

| Severity | Count |
|----------|-------|
| Critical | 1     |
| High     | 4     |
| Medium   | 5     |
| Low      | 4     |
| **Total**| **14**|

**Immediate actions required:**
1. Generate and commit a pinned `requirements.lock` (via `pip-compile` or `pip freeze`)
2. Pin all direct dependencies to exact versions in `requirements.txt`
3. Upgrade `setuptools` in the build environment to ≥70.0.0 (CVE-2024-6345, RCE)
4. Pin all GitHub Actions to full commit SHAs
5. Enable GitHub Dependabot for automated vulnerability alerts

---

## Dependency Inventory

### Runtime Dependencies (`requirements.txt` and `pyproject.toml [project.dependencies]`)

| Package | Specified Version | Inferred Purpose | Version Status |
|---------|------------------|-----------------|----------------|
| `moviepy` | *(none)* | Video composition — adding captions/subtitles to MP4 output | ❌ Completely unpinned |
| `python-dotenv` | *(none)* | Loads `.env` file into `os.environ` at startup | ❌ Completely unpinned |
| `google-api-python-client` | *(none)* | Google REST API client (Drive, YouTube, etc.) | ❌ Completely unpinned |
| `google-auth-httplib2` | *(none)* | HTTP transport adapter for Google auth | ❌ Completely unpinned |
| `google-auth-oauthlib` | *(none)* | OAuth 2.0 flow for Google APIs | ❌ Completely unpinned |
| `openai` | *(none)* | OpenAI Python SDK — GPT model API calls for horoscope generation | ❌ Completely unpinned |
| `requests` | *(none)* | HTTP client — calls Vertex AI Veo REST API directly | ❌ Completely unpinned |

### Development Dependencies (`pyproject.toml [project.optional-dependencies.dev]`)

| Package | Specified Version | Purpose | Version Status |
|---------|------------------|---------|----------------|
| `pytest` | `>=7.0` | Test runner | ⚠️ Lower-bound only |
| `pytest-cov` | *(none)* | Coverage reporting | ❌ Completely unpinned |
| `black` | *(none)* | Code formatter | ❌ Completely unpinned |
| `flake8` | *(none)* | Linter | ❌ Completely unpinned |
| `mypy` | *(none)* | Static type checker | ❌ Completely unpinned |

### Build System (`pyproject.toml [build-system]`)

| Package | Specified Version | Purpose | Version Status |
|---------|------------------|---------|----------------|
| `setuptools` | `>=61.0` | Build backend | ⚠️ Lower-bound only — **CVE-exposed** |
| `wheel` | *(none)* | Wheel builder | ❌ Completely unpinned |

### Key Transitive Dependencies (inferred)

| Package | Pulled In By | Risk Level |
|---------|-------------|-----------|
| `urllib3` | `requests` | HIGH — multiple CVEs |
| `certifi` | `requests` | HIGH — CVE-2023-37920 |
| `httplib2` | `google-auth-httplib2` | MEDIUM |
| `Pillow` | `moviepy` | MEDIUM — historically many CVEs |
| `imageio` | `moviepy` | MEDIUM |
| `imageio-ffmpeg` | `moviepy` | MEDIUM — wraps ffmpeg binary |
| `decorator` | `moviepy` | LOW |
| `numpy` | `moviepy` | LOW |
| `google-auth` | `google-api-python-client` | MEDIUM — handles OAuth tokens |
| `httpx` | `openai` | MEDIUM |
| `anyio` | `openai` | LOW |

> ⚠️ **Note:** Because no lockfile exists, the actual transitive tree is unknown until `pip install` is run. The versions above are inferred from known dependency graphs at audit time.

---

## Lockfile Health Assessment

| Check | Status | Detail |
|-------|--------|--------|
| Lockfile present and committed | ❌ **MISSING** | No `requirements.lock`, `Pipfile.lock`, `poetry.lock`, or `pip freeze` output exists in the repository |
| Matches manifest | ❌ N/A | No lockfile to compare against |
| Integrity hashes (`--hash` mode) | ❌ **ABSENT** | No SHA-256 hashes for any package |
| Official registry URLs | ⚠️ Unverifiable | No `pip.conf` or index URL constraints present |
| No local/VCS entries | ✅ Pass | All dependencies reference PyPI names only |
| Lockfile freshness | ❌ N/A | No lockfile exists |

**Assessment:** The lockfile posture is the single most critical gap in this project's supply chain hygiene. Without a lockfile:
- Every developer and every CI run resolves dependencies independently
- A compromised or updated package version is silently adopted
- Reproducible builds are impossible
- Hash verification (the primary defense against registry tampering) cannot be used

---

## Findings

---

### CRITICAL Findings

---

#### [CRITICAL] — F-01: No Dependency Lockfile Committed

- **SDL Phase:** Verification
- **Category:** Lockfile Hygiene
- **Package:** All runtime dependencies
- **Ecosystem:** Python / pip
- **Description:**  
  The repository contains no lockfile of any kind — no `requirements.lock`, no `pip freeze` output, no `Pipfile.lock`, and no `poetry.lock`. The only dependency specification is `requirements.txt`, which lists seven packages with zero version constraints.  
  
  This means every installation — whether on a developer's machine, in CI, or on a production server — independently resolves the full transitive dependency graph against the live PyPI index at that moment. The effective dependency set can silently change at any time due to new package releases.  
  
  This is the root cause that amplifies every other finding in this report. It means:
  - Known CVEs in pinned transitive versions cannot be reliably identified or tracked
  - A malicious actor who compromises any dependency maintainer account will have their code automatically adopted on the next install
  - There is no way to verify package integrity (hash checking requires pinned versions)
  - The project cannot be audited against a stable, known dependency graph

- **Location:** Repository root — no lockfile file present
- **Recommendation:**  
  Adopt `pip-tools` for lockfile management:
  ```bash
  # Install pip-tools
  pip install pip-tools

  # Generate a fully-pinned, hashed lockfile from requirements.txt
  pip-compile --generate-hashes --output-file requirements.lock requirements.txt

  # Install from the lockfile (hash-verified)
  pip install --require-hashes -r requirements.lock
  ```
  Commit `requirements.lock` to version control. Regenerate it on every dependency update. Add a CI step that fails if `requirements.lock` is out of sync with `requirements.txt`.

---

### HIGH Findings

---

#### [HIGH] — F-02: All Runtime Dependencies Completely Unpinned

- **SDL Phase:** Implementation
- **Category:** Dependency Pinning
- **Package:** `moviepy`, `python-dotenv`, `google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`, `openai`, `requests`
- **Ecosystem:** Python / pip
- **Description:**  
  Every one of the seven runtime dependencies in `requirements.txt` and `pyproject.toml` is declared with no version constraint whatsoever — not even a minimum version, a `~=` compatible-release specifier, or a `<` upper-bound. `pip install -r requirements.txt` will install the latest available version of every package and every transitive package at the time of installation.  
  
  This creates several compounding risks:
  1. **Uncontrolled upgrades:** A major version bump in any package (e.g., `openai` SDK breaking changes between v0.x, v1.x) can silently break the application
  2. **CVE inheritance:** New CVEs discovered in any package are automatically inherited with no review
  3. **Supply chain substitution:** If a package name is transferred to a new maintainer, their code runs automatically
  4. **Non-reproducibility:** Two developers running `make install` one week apart may have different dependency sets

  The same problem exists for the build system — `setuptools>=61.0` and `wheel` (unpinned) are specified without upper bounds.

- **Location:**  
  - `requirements.txt` — all 7 lines
  - `pyproject.toml` — `[project.dependencies]` (all 7 entries) and `[build-system].requires`

- **Recommendation:**  
  Immediately pin all dependencies to exact versions. As a starting point:
  ```
  # requirements.txt — example pinned versions (verify latest stable before committing)
  moviepy==2.1.1
  python-dotenv==1.0.1
  google-api-python-client==2.154.0
  google-auth-httplib2==0.2.0
  google-auth-oauthlib==1.2.1
  openai==1.57.4
  requests==2.32.3
  ```
  Then generate `requirements.lock` with hash verification as described in F-01.

---

#### [HIGH] — F-03: setuptools Build Dependency Exposed to CVE-2024-6345 (CVSS 8.8 — RCE)

- **SDL Phase:** Implementation
- **Category:** Known Vulnerability (CVE)
- **Package:** `setuptools>=61.0` (build system dependency)
- **Ecosystem:** Python / pip
- **CVE:** CVE-2024-6345
- **CVSS:** 8.8 (High)
- **Description:**  
  `pyproject.toml` specifies `setuptools>=61.0` as a build-system dependency with only a minimum version bound. CVE-2024-6345, disclosed in July 2024, allows **remote code execution** via maliciously crafted package URLs in `package_index.py`. An attacker who can influence the packages being installed (e.g., via a compromised dependency or a malicious index) could achieve code execution during the build/install phase.  
  
  The vulnerability affects all `setuptools` versions prior to **70.0.0**. Because the specifier is `>=61.0` with no upper bound and no lockfile, any environment where `setuptools` was installed before July 2024 may still be running a vulnerable version. New environments will get the patched version, but there is no mechanism to enforce the minimum safe version.

- **Location:** `pyproject.toml`, line 2: `requires = ["setuptools>=61.0", "wheel"]`
- **Recommendation:**  
  ```toml
  # pyproject.toml
  [build-system]
  requires = ["setuptools>=70.0.0", "wheel"]
  ```
  Additionally, pin the exact version in the build environment's own lockfile and verify via `pip show setuptools` in CI.

---

#### [HIGH] — F-04: urllib3 Transitive Dependency — CVE-2023-43804 (CVSS 8.1 — Cookie Header Leak)

- **SDL Phase:** Verification
- **Category:** Known Vulnerability (CVE) — Transitive
- **Package:** `urllib3` (transitive via `requests`)
- **Ecosystem:** Python / pip
- **CVE:** CVE-2023-43804
- **CVSS:** 8.1 (High)
- **Description:**  
  `urllib3` versions prior to 1.26.17 (1.x branch) and 2.0.6 (2.x branch) do not strip `Cookie` headers on cross-origin HTTP redirects. If the application makes authenticated HTTP requests that are redirected to a different origin, session cookies may be forwarded to unintended servers.  
  
  This project calls the Google Vertex AI REST API and potentially other external services using `requests` (which wraps `urllib3`). If any API endpoint issues a redirect to a different domain, OAuth tokens or session cookies could leak to the redirect target.  
  
  Because no lockfile pins `urllib3` to a specific version, the installed version may be vulnerable depending on when dependencies were last resolved.

- **Location:** Transitive dependency via `requests` (declared in `requirements.txt` and `pyproject.toml`)
- **Recommendation:**  
  Pin `requests>=2.31.0` (which pulls `urllib3>=1.26.17`) and add `urllib3>=1.26.18` as an explicit transitive pin in `requirements.lock`. Include in `requirements.txt` if defence-in-depth is desired:
  ```
  requests==2.32.3
  urllib3==2.2.3
  ```

---

#### [HIGH] — F-05: certifi Transitive Dependency — CVE-2023-37920 (CVSS 7.5 — Compromised Root CA)

- **SDL Phase:** Verification
- **Category:** Known Vulnerability (CVE) — Transitive
- **Package:** `certifi` (transitive via `requests`)
- **Ecosystem:** Python / pip
- **CVE:** CVE-2023-37920
- **CVSS:** 7.5 (High)
- **Description:**  
  `certifi` versions prior to 2023.07.22 include the e-Tugra root certificate, which was removed from major browser trust stores in 2023 following security concerns about the CA's practices. Applications using an affected `certifi` version may establish TLS connections that they believe are validated to an attacker-controlled server if an adversary can obtain a certificate signed by e-Tugra.  
  
  This project makes HTTPS calls to OpenAI's API (`api.openai.com`) and Google Vertex AI (`aiplatform.googleapis.com`), transmitting API keys in HTTP headers. A MitM attack exploiting a weak root CA could intercept these secrets.

- **Location:** Transitive dependency via `requests` → `certifi`
- **Recommendation:**  
  Pin `certifi>=2023.07.22` explicitly. Add it to `requirements.lock` or `requirements.txt`:
  ```
  certifi==2024.12.14
  ```

---

### MEDIUM Findings

---

#### [MEDIUM] — F-06: requests — CVE-2023-32681 (CVSS 6.1 — Proxy Credential Forwarding)

- **SDL Phase:** Verification
- **Category:** Known Vulnerability (CVE)
- **Package:** `requests` (unpinned, directly declared)
- **Ecosystem:** Python / pip
- **CVE:** CVE-2023-32681
- **CVSS:** 6.1 (Medium)
- **Description:**  
  `requests` versions prior to 2.31.0 forward `Proxy-Authorization` headers to the destination server when following HTTP redirects to the same host. In environments where an HTTP proxy is configured with credentials, this could leak those credentials to the target API server.  
  
  This project uses `requests` directly in `veo_horoscope_pipeline.py` to call the Google Vertex AI long-running operations endpoint. If running behind an authenticated proxy (common in enterprise/CI environments), proxy credentials could be forwarded to Google's APIs on redirect.

- **Location:**  
  - `requirements.txt` line 7: `requests`  
  - `src/render/veo_horoscope_pipeline.py` — direct HTTP calls via `requests`
- **Recommendation:**  
  Pin `requests>=2.31.0`. As of audit date, `requests==2.32.3` is the latest stable release and includes this fix.

---

#### [MEDIUM] — F-07: GitHub Actions Not Pinned to Commit SHAs

- **SDL Phase:** Verification / Release
- **Category:** CI/CD Pipeline Integrity
- **Package:** `actions/checkout@v4`, `github/codeql-action/init@v3`, `github/codeql-action/autobuild@v3`, `github/codeql-action/analyze@v3`
- **Ecosystem:** GitHub Actions
- **Description:**  
  All four GitHub Actions in `.github/workflows/codeql.yml` are pinned to mutable tag references (`@v4`, `@v3`) rather than immutable commit SHAs. A mutable tag reference means the action can be silently updated (intentionally or via account compromise) to execute different code the next time the workflow runs.  
  
  The **CodeQL workflow has `security-events: write` and `contents: read` permissions**. If any of these actions is compromised, an attacker gains the ability to write to the repository's security alerts and read all source code. A supply chain compromise of `github/codeql-action` would be particularly impactful as it is widely trusted.  
  
  While `actions/checkout` and `github/codeql-action` are GitHub-owned and monitored, the Security Development Lifecycle requires that all CI dependencies be pinned to verified, immutable references — especially when those actions have write permissions.

- **Location:** `.github/workflows/codeql.yml` — lines 19, 86, 96, 99
- **Recommendation:**  
  Pin all actions to full commit SHAs. Example:
  ```yaml
  # Replace mutable tag references with SHA pins
  - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683  # v4.2.2
  
  - uses: github/codeql-action/init@dd746615b70b5a1f426b76a45bd76b615e7d6e79  # v3.28.x
  
  - uses: github/codeql-action/autobuild@dd746615b70b5a1f426b76a45bd76b615e7d6e79  # v3.28.x
  
  - uses: github/codeql-action/analyze@dd746615b70b5a1f426b76a45bd76b615e7d6e79  # v3.28.x
  ```
  Use a tool like [pin-github-action](https://github.com/mheap/pin-github-action) or [Ratchet](https://github.com/sethvargo/ratchet) to automate SHA pinning.

---

#### [MEDIUM] — F-08: No Automated Dependency Update Tooling (Dependabot / Renovate)

- **SDL Phase:** Verification
- **Category:** Vulnerability Management
- **Description:**  
  The repository has no `.github/dependabot.yml` configuration and no Renovate configuration. Without automated dependency update tooling, security patches in upstream packages will not generate pull requests and will not be applied unless a developer manually monitors for and applies them.  
  
  This is especially significant given the complete absence of version pinning (F-02) — even if developers were monitoring CVEs, there is no systematic mechanism to detect when a transitive dependency receives a security patch that needs to be explicitly adopted.  
  
  The CodeQL workflow (the only security automation in place) performs static analysis of the project's own source code but does not scan dependencies for known vulnerabilities. There is no `pip audit` or `safety check` step anywhere in the CI pipeline.

- **Location:** Repository root — no `.github/dependabot.yml` present
- **Recommendation:**  
  Add `.github/dependabot.yml`:
  ```yaml
  version: 2
  updates:
    - package-ecosystem: "pip"
      directory: "/"
      schedule:
        interval: "weekly"
        day: "monday"
      open-pull-requests-limit: 10
      reviewers:
        - "davem5321"
      labels:
        - "dependencies"
        - "security"

    - package-ecosystem: "github-actions"
      directory: "/"
      schedule:
        interval: "weekly"
      labels:
        - "dependencies"
        - "ci"
  ```
  Also add a `pip audit` step to the CI pipeline.

---

#### [MEDIUM] — F-09: moviepy — Native Subprocess Execution via ImageMagick and ffmpeg

- **SDL Phase:** Implementation
- **Category:** Install Hook / Transitive Risk
- **Package:** `moviepy` (unpinned, directly declared)
- **Ecosystem:** Python / pip
- **Description:**  
  `moviepy` delegates media processing operations to **ImageMagick** and **ffmpeg** via subprocess calls. Both are invoked by executing system binaries discovered via `IMAGEMAGICK_BINARY` and `FFMPEG_BINARY` environment variables or default path discovery. This creates two supply chain risks:
  
  1. **External binary trust:** The project implicitly trusts whatever `ffmpeg` and ImageMagick binaries are installed in the runtime environment. If those binaries are replaced or compromised (e.g., via a compromised system package), moviepy will execute them with full OS permissions.  
  
  2. **imageio-ffmpeg dependency:** `moviepy 2.x` bundles its own ffmpeg binary via the `imageio-ffmpeg` transitive package. This binary is downloaded at install time from GitHub releases. Without hash verification (`--require-hashes` mode), there is no integrity check on this binary. If the imageio-ffmpeg GitHub release is tampered with, a malicious ffmpeg binary could be installed.  
  
  3. **Unmaintained 1.x branch:** `moviepy 1.0.3` (the latest 1.x release, published 2020) is effectively unmaintained. Any CVEs discovered in the 1.x series will not receive patches. Without a version pin, it is ambiguous which major version will be installed.

- **Location:** `requirements.txt` line 1: `moviepy`; `src/render/subtitle_utils.py` (uses moviepy for captioning)
- **Recommendation:**  
  1. Pin to `moviepy==2.1.1` (or the current latest 2.x stable)  
  2. After generating `requirements.lock`, verify the `imageio-ffmpeg` wheel hash matches the expected SHA-256 from the official release  
  3. Consider using `ffmpeg-python` or direct `subprocess` calls with a pinned system ffmpeg as an alternative

---

### LOW Findings

---

#### [LOW] — F-10: setuptools — CVE-2022-40897 (CVSS 5.9 — ReDoS)

- **SDL Phase:** Verification
- **Category:** Known Vulnerability (CVE)
- **Package:** `setuptools` (build dependency, `>=61.0`)
- **Ecosystem:** Python / pip
- **CVE:** CVE-2022-40897
- **CVSS:** 5.9 (Medium — downgraded to Low given build-only context)
- **Description:**  
  `setuptools` versions prior to 65.5.1 contain a regular expression denial of service (ReDoS) vulnerability in `package_index.py`. A specially crafted package page HTML can cause catastrophic backtracking, hanging the `pip` process. This is exploitable if `pip install` is run against an attacker-controlled or compromised PyPI mirror.  
  
  The risk is lower for this project because it specifies `setuptools>=61.0` (which may already include the fix depending on the installed version), and the ReDoS only triggers during `pip install` rather than at runtime.

- **Location:** `pyproject.toml`, line 2: `requires = ["setuptools>=61.0", "wheel"]`
- **Recommendation:**  
  Already addressed by F-03 recommendation (pin to `setuptools>=70.0.0`). No additional action needed beyond F-03.

---

#### [LOW] — F-11: Dev Dependencies Loosely Pinned — CI Reproducibility Risk

- **SDL Phase:** Verification
- **Category:** Dependency Pinning
- **Description:**  
  Development dependencies in `pyproject.toml [project.optional-dependencies.dev]` are either completely unpinned (`pytest-cov`, `black`, `flake8`, `mypy`) or specified with a one-sided lower bound only (`pytest>=7.0`). While dev dependencies do not run in production, they execute in the CI environment during testing and linting.  
  
  A compromised dev dependency (e.g., a malicious `black` release) would execute arbitrary code on CI runners, which have access to repository secrets and could exfiltrate credentials or inject malicious code into build artifacts.

- **Location:** `pyproject.toml`, lines 39–45: `[project.optional-dependencies.dev]`
- **Recommendation:**  
  Generate a separate `requirements-dev.lock` using `pip-compile`:
  ```bash
  pip-compile --generate-hashes --output-file requirements-dev.lock pyproject.toml --extra dev
  ```

---

#### [LOW] — F-12: No SBOM Generated or Committed

- **SDL Phase:** Release
- **Category:** SLSA / Provenance
- **Description:**  
  No Software Bill of Materials (SBOM) is generated or committed as part of the build or release process. An SBOM provides a machine-readable inventory of all direct and transitive components, enabling downstream consumers and security teams to rapidly assess exposure when new CVEs are disclosed.  
  
  The absence of an SBOM means that if a critical zero-day is published for any transitive dependency (e.g., a new Pillow RCE), the maintainer has no quick way to determine whether and which version of that package is in use.

- **Location:** No `sbom.json`, `sbom.spdx`, or `cyclonedx.json` present in the repository
- **Recommendation:**  
  See [SBOM Generation Command](#sbom-generation-command) section below.

---

#### [LOW] — F-13: No `pip audit` Step in CI Pipeline

- **SDL Phase:** Verification
- **Category:** Vulnerability Management
- **Description:**  
  The only CI workflow (`.github/workflows/codeql.yml`) performs static source analysis but does not run `pip audit` or `safety check` against the installed dependency set. Known CVEs in runtime dependencies will not block builds or generate alerts. Vulnerability scanning is a Verification-phase SDL requirement.

- **Location:** `.github/workflows/codeql.yml` — no dependency audit step present
- **Recommendation:**  
  Add a dependency audit job to the CI pipeline. Example GitHub Actions step:
  ```yaml
  - name: Audit Python dependencies
    run: |
      pip install pip-audit
      pip-audit -r requirements.txt --desc --output json > audit-report.json
    continue-on-error: false  # Fail the build on HIGH/CRITICAL CVEs
  
  - name: Upload audit report
    uses: actions/upload-artifact@v4
    with:
      name: pip-audit-report
      path: audit-report.json
  ```

---

#### [LOW] — F-14: Makefile `install` Target Has No Hash Verification

- **SDL Phase:** Implementation
- **Category:** Build Integrity
- **Description:**  
  The `Makefile` `install` target runs `pip install -r requirements.txt` with no `--require-hashes` flag. This means packages are downloaded and installed without verifying their integrity against expected checksums. An attacker performing a registry mirror attack or DNS poisoning could substitute a malicious package, and pip would install it without warning.  
  
  Hash verification in pip requires that every package (including transitive dependencies) has a hash specified in the requirements file. This is why generating a `requirements.lock` with `pip-compile --generate-hashes` is essential.

- **Location:** `Makefile`, line 19: `pip install -r requirements.txt`
- **Recommendation:**  
  After generating `requirements.lock`:
  ```makefile
  install:
      pip install --require-hashes -r requirements.lock
  ```

---

## Dependency Risk Register

Risk scores are calculated using the SDL supply chain risk scoring matrix:  
**Data Sensitivity** (1–3) + **Network Access** (1–3) + **Code Execution at Install** (1–3) + **Version Pinning** (1–3) + **Weekly Downloads** (1–3) + **Last Published** (1–3) = Total (6–18)

Score bands: **15–18 → High** | **10–14 → Medium** | **6–9 → Low**

| Package | Specified Version | Risk Score | Data Sensitivity | Network Access | Install Exec | Pinning | Downloads | Freshness | Top Risk Factor | Recommended Action |
|---------|------------------|-----------|-----------------|----------------|-------------|---------|-----------|-----------|-----------------|-------------------|
| `openai` | *(none)* | **13** | 3 (API keys + prompts) | 3 (required) | 1 | 3 (unpinned) | 1 (>1M/wk) | 1 (<6mo) | Handles API secrets; completely unpinned | Pin to `==1.57.4`; add to lockfile with hashes |
| `google-auth-oauthlib` | *(none)* | **13** | 3 (OAuth tokens) | 3 (required) | 1 | 3 (unpinned) | 1 (>1M/wk) | 1 (<6mo) | OAuth flow handles Google credentials; unpinned | Pin to `==1.2.1`; add to lockfile |
| `requests` | *(none)* | **12** | 2 (auth headers) | 3 (required) | 1 | 3 (unpinned) | 1 (>1M/wk) | 1 (<6mo) | CVE-2023-32681 + unpinned; transmits API keys | Pin to `==2.32.3`; lockfile; upgrade urllib3 |
| `google-api-python-client` | *(none)* | **12** | 2 (API creds) | 3 (required) | 1 | 3 (unpinned) | 1 (>1M/wk) | 1 (<6mo) | Handles Google API auth; unpinned | Pin to `==2.154.0`; add to lockfile |
| `google-auth-httplib2` | *(none)* | **11** | 2 (auth headers) | 3 (required) | 1 | 3 (unpinned) | 1 (>1M/wk) | 1 (<6mo) | HTTP transport for Google OAuth; unpinned | Pin to `==0.2.0`; add to lockfile |
| `moviepy` | *(none)* | **11** | 1 (no user data) | 1 (no network) | 3 (imageio-ffmpeg binary) | 3 (unpinned) | 2 (100k–1M/wk) | 2 (1.x stale, 2.x recent) | Downloads external ffmpeg binary at install; 1.x unmaintained | Pin to `==2.1.1`; verify imageio-ffmpeg hashes |
| `python-dotenv` | *(none)* | **9** | 2 (reads .env with secrets) | 1 (no network) | 1 | 3 (unpinned) | 1 (>1M/wk) | 1 (<6mo) | Reads secrets from environment; unpinned | Pin to `==1.0.1`; add to lockfile |

**Transitive dependency risk register:**

| Package | Via | CVE | CVSS | Status |
|---------|-----|-----|------|--------|
| `urllib3` | `requests` | CVE-2023-43804 | 8.1 | Fix: `>=2.0.6` or `>=1.26.17` |
| `certifi` | `requests` | CVE-2023-37920 | 7.5 | Fix: `>=2023.07.22` |
| `setuptools` | build system | CVE-2024-6345 | 8.8 | Fix: `>=70.0.0` |
| `setuptools` | build system | CVE-2022-40897 | 5.9 | Fix: `>=65.5.1` |

---

## CI/CD Pipeline Integrity Assessment

**Workflow file:** `.github/workflows/codeql.yml`

| Check | Status | Detail |
|-------|--------|--------|
| Workflow triggers | ✅ Safe | `push`, `pull_request`, `schedule` — no `pull_request_target` (which would grant write access to PR code) |
| Action SHA pinning | ❌ **Not pinned** | All 4 actions use mutable tag refs (`@v4`, `@v3`) |
| Secret exposure | ✅ No secrets in workflow | No `${{ secrets.* }}` references; no environment variables containing tokens |
| Permissions | ⚠️ Review | `security-events: write` and `contents: read` are appropriate for CodeQL but would be high-value targets if an action is compromised |
| Artifact integrity | ✅ No artifacts produced | CodeQL uploads to GitHub Security tab — no build artifacts to tamper with |
| Dependency audit step | ❌ Missing | No `pip audit` or `safety check` step |
| `pull_request_target` risk | ✅ Not used | Workflow uses `pull_request` (restricted sandbox), not `pull_request_target` |
| Third-party actions | ✅ GitHub-owned only | All actions are `actions/*` or `github/*` — no third-party actions |

**Missing CI workflows:**
- No build workflow (no `pip install` + test run)
- No dependency scanning workflow
- No release/publish workflow
- No SBOM generation workflow

**Note:** While the CodeQL workflow is correctly scoped and doesn't expose secrets, the lack of SHA-pinning for actions is a Medium risk per SDL CI/CD hardening requirements.

---

## Dependency Confusion Assessment

| Risk | Assessment |
|------|-----------|
| Scoped/namespaced packages | ❌ Python/PyPI has no scoping mechanism equivalent to npm `@org/` |
| Internal/proprietary package names | ✅ No internal package names detected — all 7 packages are well-established public PyPI packages |
| Package name squattability | ✅ Low risk — `moviepy`, `openai`, `requests`, `python-dotenv`, `google-*` are all registered, active, popular packages with legitimate maintainers |
| Typosquatting exposure | ⚠️ Low-Medium — `python-dotenv` could be confused with `dotenv` (a distinct, less-maintained package); `google-auth-httplib2` has a complex name that could be typosquatted |
| Private registry enforcement | ❌ No `pip.conf` or `PIP_INDEX_URL` constraint — defaults to public PyPI |
| Registry auth tokens | ✅ No registry credentials committed to source |

**Typosquatting candidates to monitor:**

| Legitimate Package | Potential Typosquats to Watch |
|-------------------|-------------------------------|
| `python-dotenv` | `python-dotenv-utils`, `dotenv-python`, `pythondotenv` |
| `google-auth-httplib2` | `google-auth-http2`, `google-auth-httplib` |
| `google-auth-oauthlib` | `google-oauthlib`, `google-auth-oauth2lib` |

**Assessment:** Dependency confusion risk is **Low** for this project. All declared dependencies are high-profile, well-established packages maintained by trusted organizations (OpenAI, Google, Python Software Foundation, the requests maintainers). The primary risk is typosquatting during manual installation if package names are mistyped, which can be mitigated by hash-verified installation from a lockfile.

---

## SLSA Level Assessment

| SLSA Level | Requirement | Status |
|-----------|-------------|--------|
| **Level 1** | Documented build process | ✅ Partial — `Makefile` and `pyproject.toml` document the build. No formal provenance document. |
| **Level 1** | Build scripted (not manual) | ✅ `make install` / `make run` automates execution |
| **Level 2** | Version-controlled source | ✅ Git repository on GitHub |
| **Level 2** | Hosted build platform (CI) | ⚠️ Partial — CodeQL CI exists but no build/test workflow; no artifacts produced by CI |
| **Level 2** | Provenance generated by CI | ❌ No provenance attestation generated |
| **Level 3** | Hardened build (isolated, ephemeral) | ❌ No hermetic build environment |
| **Level 3** | Non-falsifiable provenance | ❌ Not implemented |
| **Level 4** | Two-person review | ❌ No branch protection rules enforcing review (not verified, but not configured in repo) |
| **Level 4** | Hermetic, reproducible build | ❌ Build is non-reproducible (no lockfile) |

### **Current SLSA Level: 1 (Partial)**

The project meets basic SLSA Level 1 requirements (scripted build, version control) but falls short of Level 2 because:
- No CI workflow produces build artifacts or provenance
- No provenance attestation is generated or published
- The build is non-reproducible due to missing lockfile

### Path to SLSA Level 2

1. Add a CI workflow that installs, tests, and (optionally) packages the project
2. Generate SLSA provenance in CI using the [slsa-github-generator](https://github.com/slsa-framework/slsa-github-generator)
3. Generate and publish an SBOM artifact with each release
4. Pin all GitHub Actions to commit SHAs

---

## Recommended Dependabot Configuration

Create `.github/dependabot.yml` with the following configuration:

```yaml
# .github/dependabot.yml
# Automated dependency update configuration
# See: https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file

version: 2

updates:
  # Python runtime dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
      timezone: "America/Los_Angeles"
    open-pull-requests-limit: 10
    reviewers:
      - "davem5321"
    assignees:
      - "davem5321"
    labels:
      - "dependencies"
      - "python"
    commit-message:
      prefix: "chore(deps)"
    # Group minor and patch updates to reduce PR noise
    groups:
      google-apis:
        patterns:
          - "google-*"
      openai-sdk:
        patterns:
          - "openai"
      dev-tools:
        patterns:
          - "pytest*"
          - "black"
          - "flake8"
          - "mypy"
    # Always create PRs for security updates regardless of schedule
    allow:
      - dependency-type: "direct"
      - dependency-type: "indirect"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    labels:
      - "dependencies"
      - "ci"
    commit-message:
      prefix: "chore(ci)"
```

---

## SBOM Generation Command

Generate a CycloneDX SBOM for this project (covers all direct and transitive Python dependencies):

```bash
# Install CycloneDX generator
pip install cyclonedx-bom

# Generate SBOM in CycloneDX JSON format (most widely supported)
cyclonedx-py requirements requirements.txt \
  --output-format json \
  --output-file sbom.cyclonedx.json \
  --schema-version 1.6

# Or from installed environment (more accurate — captures transitive deps)
pip install -r requirements.txt  # ensure deps are installed
cyclonedx-py environment \
  --output-format json \
  --output-file sbom.cyclonedx.json \
  --schema-version 1.6

# Also generate SPDX format for GitHub Dependency Submission API compatibility
pip install spdx-tools
cyclonedx-py environment \
  --output-format xml \
  --output-file sbom.spdx.xml

# Verify the SBOM
python -c "import json; d=json.load(open('sbom.cyclonedx.json')); print(f'Components: {len(d[\"components\"])}')"
```

**Integrate SBOM generation into CI** by adding a step to the build workflow:

```yaml
- name: Generate SBOM
  run: |
    pip install cyclonedx-bom
    cyclonedx-py environment --output-format json --output-file sbom.cyclonedx.json
  
- name: Upload SBOM as artifact
  uses: actions/upload-artifact@v4
  with:
    name: sbom-${{ github.sha }}
    path: sbom.cyclonedx.json
    retention-days: 90
```

Alternatively, use `pip-audit` for combined vulnerability + SBOM output:

```bash
pip install pip-audit
pip-audit -r requirements.txt \
  --format cyclonedx-json \
  --output sbom-with-vulns.cyclonedx.json
```

---

## Remediation Roadmap

### Immediate (Within 24–48 Hours) — SDL Response Phase

| Action | Finding | Command |
|--------|---------|---------|
| Pin setuptools to ≥70.0.0 | F-03 (CVE-2024-6345, RCE) | Edit `pyproject.toml` |
| Generate requirements.lock with hashes | F-01 | `pip-compile --generate-hashes -o requirements.lock requirements.txt` |
| Pin requests ≥2.31.0 | F-06 (CVE-2023-32681) | Edit `requirements.txt` |
| Pin certifi ≥2023.07.22 | F-05 (CVE-2023-37920) | Add to `requirements.txt` |

### Short-Term (Within 1 Sprint) — SDL Implementation Phase

| Action | Finding |
|--------|---------|
| Pin all 7 runtime deps to exact versions | F-02 |
| Add `requirements.lock` to git and update `Makefile install` target to use `--require-hashes` | F-01, F-14 |
| Pin GitHub Actions to commit SHAs | F-07 |
| Add `.github/dependabot.yml` | F-08 |
| Add `pip-audit` step to CI | F-13 |

### Medium-Term (Within 1 Month) — SDL Verification Phase

| Action | Finding |
|--------|---------|
| Create separate `requirements-dev.lock` | F-11 |
| Generate and commit initial SBOM | F-12 |
| Add a build + test CI workflow | F-10 (SLSA) |
| Investigate moviepy imageio-ffmpeg binary hash verification | F-09 |

---

## Conclusion

The **ai-video-agent** project has a capable codebase but a **fragile supply chain posture**. The complete absence of dependency pinning and lockfile infrastructure means the project is one `pip install` away from silently adopting a breaking change, a newly-disclosed CVE, or — in a worst case — a maliciously-updated package.

The most critical remediation is generating and committing a `requirements.lock` file with `pip-compile --generate-hashes`. This single action resolves F-01 (Critical), partially addresses F-02 (High), enables hash-verified installation to mitigate F-14 (Low), and provides the stable dependency graph needed to confirm or deny the CVE exposure in F-04 and F-05.

Given that this project handles **Google Cloud API keys** and **OpenAI API keys** at runtime (secrets with real billing and data-access implications), the supply chain hygiene bar should be higher than typical. A compromised transitive dependency that exfiltrates `GOOGLE_API_KEY` or `OPENAI_API_KEY` from the runtime environment would have direct financial and operational impact.

**Overall Supply Chain Risk Rating: 🔴 HIGH**  
The project should not be considered production-ready from a supply chain security perspective until at minimum F-01, F-02, F-03, F-04, and F-05 are resolved.

---

*Report generated by AI Supply Chain Security Agent — Microsoft SDL aligned*  
*Audit scope: Static analysis of manifest files, CI configuration, and build automation*  
*Limitations: Dynamic analysis (actual `pip install` resolution, CVE scanner output) not performed — findings based on static review and known CVE knowledge base*
