# Accessibility Audit Report — AI Video Agent

**Date:** 2025-07-14  
**Auditor:** Automated Accessibility Audit (WCAG 2.1/2.2, SDL-aligned)  
**Project:** `ai-video-agent` — Horoscope Video Generation Pipeline  
**Repository:** `C:\Users\davem\repos\ai-video-agent`  
**Audit Tool Version:** 1.0  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Scope & Methodology](#scope--methodology)
3. [WCAG Conformance Summary](#wcag-conformance-summary)
4. [Findings](#findings)
   - [Critical](#critical-findings)
   - [High](#high-findings)
   - [Medium](#medium-findings)
   - [Low](#low-findings)
5. [Quick Wins](#quick-wins)
6. [Conclusion](#conclusion)
7. [Appendix — Files Audited](#appendix--files-audited)

---

## Executive Summary

The `ai-video-agent` project is a Python CLI backend pipeline that generates horoscope MP4 videos using OpenAI (script writing) and Google Vertex AI Veo (video generation). It produces short-form social media videos (9:16, 16:9, 1:1) with burned-in text captions via ffmpeg.

Because the project has **no web or desktop UI**, standard DOM/ARIA-focused WCAG criteria do not apply. The audit instead evaluated the areas most relevant to this project type:

- **Video output accessibility** (WCAG 1.2.x — Time-based Media)
- **CLI output accessibility** for users relying on screen readers or text interfaces
- **Documentation accessibility**
- **Error message quality** for all users

**15 findings** were identified across four severity levels:

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High     | 4 |
| Medium   | 5 |
| Low      | 5 |
| **Total**| **15** |

The most serious issue is a **code path that silently produces uncaptioned videos** when Google Cloud Storage (GCS) URIs are returned by the Veo API — a clear violation of WCAG 1.2.2 (Captions, Level A). Additionally, the **default "scroll" caption style** moves text across the screen at high speed and is a significant barrier for users with cognitive, reading, or low-vision disabilities.

---

## Scope & Methodology

### Project Type
**CLI tool / Backend pipeline** — Python 3.10+, no browser UI, no web framework, no HTML/CSS. Outputs MP4 video files intended for downstream consumption on social media platforms.

### Evaluation Areas

| Area | Applicable Standard | Evaluated? |
|------|--------------------|----|
| Web UI / ARIA / DOM | WCAG 2.1 SC 4.1.x | ❌ Not applicable (no UI) |
| Video captions (output) | WCAG 2.1 SC 1.2.2 | ✅ Yes |
| Audio descriptions (output) | WCAG 2.1 SC 1.2.3 / 1.2.5 | ✅ Yes |
| CLI output / screen readers | WCAG 2.1 SC 4.1.3 + best practice | ✅ Yes |
| Color/contrast (video overlays) | WCAG 2.1 SC 1.4.3 | ✅ Yes |
| Documentation | Best practice | ✅ Yes |
| Error messages | Best practice | ✅ Yes |
| Motion/animation (output video) | WCAG 2.1 SC 2.3.1 | ✅ Yes |

### SDL Phase Mapping
Findings are classified under Microsoft SDL phases as follows:
- **Design** — architectural decisions about output format, caption approach, or output structure
- **Implementation** — code-level bugs or oversights that can be fixed without redesign

---

## WCAG Conformance Summary

This evaluation covers **WCAG 2.1** criteria applicable to time-based media output and CLI interfaces.

| Criterion | Level | Description | Status |
|-----------|-------|-------------|--------|
| 1.2.2 Captions (Prerecorded) | A | Captions for all synchronized media | ⚠️ **Partial** — GCS code path produces no captions; fallback path also uncaptioned |
| 1.2.3 Audio Description or Media Alternative | A | Audio description or full text alternative | ❌ **Fail** — No audio description track; horoscope text only in caption |
| 1.2.5 Audio Description (Prerecorded) | AA | Audio descriptions for prerecorded video | ❌ **Fail** — No mechanism to produce audio description track |
| 1.4.3 Contrast (Minimum) | AA | 4.5:1 contrast for normal text | ✅ **Pass** — White (#FFF) + black outline on video: 21:1 against outline |
| 1.4.4 Resize Text | AA | Text can be resized | ⚠️ **Partial** — Burned-in captions cannot be resized by viewer |
| 2.3.1 Three Flashes | A | No content flashes > 3x/second | ✅ **Pass** — No flashing content in overlay pipeline |
| 4.1.3 Status Messages | AA | Status messages programmatically determined | ⚠️ **Partial** — Emoji-dependent status indicators in CLI |

**Overall estimated conformance level: Below WCAG 2.1 Level A** (specifically 1.2.2 and 1.2.3 are not consistently met).

---

## Findings

---

### Critical Findings

---

#### [CRITICAL] — GCS URI Code Path Produces Completely Uncaptioned Videos

- **SDL Phase:** Implementation
- **WCAG Criterion:** 1.2.2 Captions (Prerecorded) — Level A
- **Category:** Video Captions
- **File:** `src/render/veo_horoscope_pipeline.py`, lines 169–174
- **Description:**  
  When the Veo API returns a `gs://` (Google Cloud Storage) URI, the code stores the path and marks the job as `"done"` **without adding any captions**. The caption pipeline (`add_caption_to_video`) is only called for HTTP URLs and base64-encoded responses. Videos delivered via GCS URI are silently published uncaptioned.

  ```python
  if video_url.startswith("gs://"):
      # For GCS URLs, we'd need signed URLs or Cloud Storage API
      # For now, store the GCS path
      print(f"✅ Video ready at: {video_url}")
      job.video_path = video_url
      job.status = "done"
      # ← No call to add_caption_to_video here
  ```

- **Impact:**  
  Deaf and hard-of-hearing viewers of any video produced through this code path receive **no captions**. The horoscope narration is entirely inaccessible. This violates WCAG 1.2.2 at Level A — the lowest (most required) conformance level. GCS-hosted output is likely common in production Google Cloud environments.

- **Recommendation:**  
  1. Convert GCS URIs to signed HTTPS URLs using the Google Cloud Storage Python client library (`google-cloud-storage`) before passing to `add_caption_to_video`.  
  2. As an interim mitigation, emit a clearly visible logged **warning** (not just `✅`) when a GCS URI is returned and captions cannot be applied, so operators are aware of the gap.  
  3. Alternatively, add a `--gcs-caption-mode` CLI flag to control how GCS-hosted videos are processed.

  ```python
  # Example fix approach
  if video_url.startswith("gs://"):
      signed_url = generate_signed_url(video_url)  # implement via google-cloud-storage
      # then fall through to the HTTP download + caption path
  ```

---

### High Findings

---

#### [HIGH] — Default "Scroll" Caption Style Produces Moving Text That Is Inaccessible

- **SDL Phase:** Design
- **WCAG Criterion:** 1.2.2 Captions (Prerecorded) — Level A; WCAG 2.2.2 Pause, Stop, Hide — Level A
- **Category:** Video Captions
- **File:** `src/render/subtitle_utils.py`, lines 118–126; `src/render/veo_horoscope_pipeline.py`, line 200
- **Description:**  
  The default caption `style` is `"scroll"`, which animates text from the bottom of the frame to the top across the full video duration using an ASS `\move()` command:

  ```python
  animation = f"{{\\move({x_center},{start_y},{x_center},{end_y})}}"
  ```

  For an 8-second clip at 720p, text travels 670 pixels (from `y=720` to `y=50`). This means the full horoscope text (2–3 sentences, potentially 100–200 characters) scrolls across the screen in approximately **8 seconds**, averaging over 80px/s of movement.

  Scrolling/moving captions are classified as a barrier for:
  - Users with **dyslexia** or reading difficulties who cannot process text while it is in motion
  - Users with **cognitive disabilities** who require static, predictable text placement
  - Users with **low vision** who may be zoomed in and cannot track moving text
  - Viewers on small screens (mobile) where the scroll distance proportionally reduces legibility

  The `CAPTION_STYLE` environment variable allows overriding to `"static"`, but this is undocumented and not the default.

- **Impact:**  
  The primary accessibility mechanism for deaf/hard-of-hearing users is compromised by the kinetic presentation. Even technically "present" captions can fail WCAG 1.2.2 intent if they are illegible in practice.

- **Recommendation:**  
  1. **Change the default** `CAPTION_STYLE` from `"scroll"` to `"static"`, which uses `alignment=5` (center screen) or `alignment=2` (bottom center, the broadcast standard).  
  2. Document the `CAPTION_STYLE` environment variable and `--caption-style` CLI argument (currently absent) in the README.  
  3. If scrolling is required for aesthetic reasons, add a separate static subtitle track as a sidecar file alongside the MP4.  
  4. Update `add_caption_to_video()` default parameter from `style: str = "scroll"` to `style: str = "static"` in `subtitle_utils.py` line 242.

---

#### [HIGH] — Silent Fallback to Uncaptioned Video on FFmpeg Failure

- **SDL Phase:** Implementation
- **WCAG Criterion:** 1.2.2 Captions (Prerecorded) — Level A
- **Category:** Video Captions, Error Handling
- **File:** `src/render/veo_horoscope_pipeline.py`, lines 207–211 and lines 244–248
- **Description:**  
  In two separate code paths (HTTP download and base64 decode), when `add_caption_to_video()` fails, the pipeline silently falls back to the raw, uncaptioned video:

  ```python
  if success:
      print(f"✅ Caption added successfully: {video_path_final}")
      job.video_path = str(video_path_final)
      job.status = "done"
  else:
      print(f"⚠️  Caption failed: {message}")
      print(f"   Using raw video instead: {video_path_raw}")
      job.video_path = str(video_path_raw)   # ← uncaptioned video published
      job.status = "done"                    # ← still marked "done", no alert
  ```

  The `job.status` is set to `"done"` (not `"captioned_failed"` or `"partial"`), meaning the manifest JSON also records this as a fully successful job. Downstream consumers and operators have no reliable mechanism to detect that the distributed video has no captions unless they manually inspect every `⚠️` line in console output.

- **Impact:**  
  Any video where ffmpeg is unavailable, misconfigured, or encounters an ASS rendering error will be published without captions. Deaf and hard-of-hearing viewers receive no accessible content. This is a silent, production-level regression.

- **Recommendation:**  
  1. Introduce a distinct job status such as `"done_no_caption"` or `"caption_failed"` to separate this from a fully successful run.  
  2. Add a dedicated `caption_status` field to the `VideoJob` dataclass and manifest JSON output.  
  3. Emit a non-zero exit code from the pipeline when any video fails captioning.  
  4. Add a summary of captioning failures in the pipeline completion output.  
  5. Consider making caption failure a **hard error** (raise exception or skip publishing) rather than a soft warning.

---

#### [HIGH] — No Audio Description Track for Blind/Low-Vision Video Consumers

- **SDL Phase:** Design
- **WCAG Criterion:** 1.2.3 Audio Description or Media Alternative (Prerecorded) — Level A; 1.2.5 Audio Description (Prerecorded) — Level AA
- **Category:** Video Accessibility — Audio Description
- **File:** `src/render/veo_horoscope_pipeline.py`, lines 330–332 (`RenderSpec`); entire pipeline
- **Description:**  
  The pipeline generates cinematic AI video content (ambient astrology studio visuals) combined with an AI-generated ambient soundtrack (Veo 3+ audio). The horoscope text is burned in as a visual caption but is **not spoken aloud** in any audio track. No audio description mechanism exists to convey the video's visual content to blind or low-vision users.

  WCAG 1.2.3 (Level A) requires that synchronized media either include an audio description of the video content, or that a full text alternative (transcript) is provided. WCAG 1.2.5 (Level AA) requires audio descriptions specifically.

  The current pipeline:
  - ✅ Saves horoscope text to `prompts/<Sign>.txt` — this is a partial text alternative
  - ❌ Does not produce a transcript file linked to/alongside the video
  - ❌ Does not produce a narration/audio description audio track
  - ❌ Does not expose the `prompts/` directory location to end consumers of the video

- **Impact:**  
  Blind users and users with severe low vision who rely on audio cannot access the horoscope content through the video alone. The content is entirely conveyed visually (text captions over ambient video).

- **Recommendation:**  
  1. **Short-term:** Export a plain-text transcript file (e.g., `renders/aries.txt`) alongside each MP4 containing the horoscope text and date. Document this as the "text alternative" for WCAG 1.2.3.  
  2. **Medium-term:** Use a text-to-speech service (e.g., OpenAI TTS, Google Cloud TTS, or `gTTS`) to generate a narration audio track from `scene.script_text` and mux it into the output video using ffmpeg's `-i` (multiple input) and `-filter_complex amix` flags.  
  3. **Long-term:** If AI audio (Veo 3 generate_audio) is enabled and narrates the horoscope, document that this satisfies 1.2.5 and verify the audio content covers the text.

---

#### [HIGH] — No External Subtitle File (.srt / .vtt) Exported with Output Video

- **SDL Phase:** Design
- **WCAG Criterion:** 1.2.2 Captions (Prerecorded) — Level A
- **Category:** Video Captions
- **File:** `src/render/subtitle_utils.py`, `src/render/veo_horoscope_pipeline.py`
- **Description:**  
  Captions are burned directly into the video pixel data ("open captions" or "hardcoded subtitles"). The ASS subtitle file used to generate the captions is **deleted immediately after use** (line 297–300 in `subtitle_utils.py`):

  ```python
  try:
      os.unlink(ass_path)  # ← temp file deleted
  except:
      pass
  ```

  No `.srt`, `.vtt`, or `.ass` file is retained or exported alongside the final MP4. This creates two accessibility limitations:
  
  1. **Viewer customization is impossible.** Users who need larger text, high-contrast colors, different fonts, or reduced motion cannot modify burned-in captions. This disproportionately affects users with low vision and cognitive disabilities.
  2. **Platform upload accessibility is limited.** Social media platforms (YouTube, TikTok, Instagram) prefer or require separate subtitle tracks for their own accessibility layers (auto-translation, accessibility settings). Without a sidecar file, platform-level caption features are unavailable.

- **Impact:**  
  Users who need to adjust subtitle presentation (a recognized accessibility need under WCAG 1.4 and platform accessibility guidelines) cannot do so. Deaf-blind users relying on braille displays connected to media players that read subtitle streams also cannot access the content.

- **Recommendation:**  
  1. Save the `.srt` or `.vtt` file to `renders/<sign>.srt` (or `.vtt`) alongside the MP4 instead of deleting it.  
  2. Alternatively, generate a dedicated sidecar from `scene.script_text` using the existing `format_srt_time()` / `create_srt_file()` functions already present in `reference/utils.py`.  
  3. Add the subtitle file path to the manifest JSON entry for each job.  
  4. Document the sidecar file in the README output structure section.

---

### Medium Findings

---

#### [MEDIUM] — Caption Font Size Is Insufficient at 1080p Resolution

- **SDL Phase:** Implementation
- **WCAG Criterion:** 1.4.4 Resize Text — Level AA (by analogy for video captions); BBC Subtitle Guidelines
- **Category:** Video Captions — Legibility
- **File:** `src/render/veo_horoscope_pipeline.py`, lines 194 and 231; `src/render/subtitle_utils.py`, line 236
- **Description:**  
  The default caption font size is hardcoded at `36` pixels (via `os.getenv("CAPTION_FONT_SIZE", "36")`). Assessed against video height:

  | Resolution | Video Height | Font Size | % of Screen Height | Assessment |
  |------------|-------------|-----------|-------------------|------------|
  | 720p | 720px | 36px | 5.0% | Borderline acceptable |
  | 1080p | 1080px | 36px | 3.3% | **Below recommended minimum** |

  The BBC Subtitle Production Guidelines (widely cited accessibility standard) recommend captions be at least **5% of screen height**. For 1080p this means a minimum of **54px**. The current default at 1080p is 33% below this threshold.

  Additionally, the default `fontsize` parameter in `create_ass_file()` is `48px` (line 69 of `subtitle_utils.py`) which would be appropriate, but the calling code in the pipeline explicitly overrides this with `fontsize=36` without adjusting for resolution.

- **Impact:**  
  Low-vision users and users viewing video on small screens (mobile phones, where 9:16 content is primarily consumed) may find captions unreadably small, particularly at 1080p. This undermines the accessibility benefit of providing captions at all.

- **Recommendation:**  
  1. Make font size resolution-aware. Use a percentage-of-height calculation:
     ```python
     video_height = 1080 if resolution == "1080p" else 720
     fontsize = max(36, int(video_height * 0.05))  # 5% of height
     # → 36px at 720p, 54px at 1080p
     ```
  2. Add a `--caption-font-size` CLI argument to allow operator control.
  3. Expose `CAPTION_FONT_SIZE` in the `.env.example` file with the recommended values documented.

---

#### [MEDIUM] — CLI Progress Output Uses Same-Line Printing Incompatible with Screen Readers

- **SDL Phase:** Implementation
- **WCAG Criterion:** 4.1.3 Status Messages — Level AA (by analogy for CLI)
- **Category:** CLI Output — Screen Reader Compatibility
- **File:** `src/write/horoscope_writer.py`, line 68
- **Description:**  
  The horoscope generation loop uses Python's `end=" "` and `flush=True` pattern to print a status character on the same line as the sign label:

  ```python
  print(f"   [{i}/{len(signs_to_generate)}] Generating {sign}...", end=" ", flush=True)
  # ... (API call) ...
  print("✓")  # appended to same line
  ```

  Screen readers (NVDA, JAWS, VoiceOver, Orca) typically read **complete lines** when text is output to a terminal. The incomplete line `"   [1/12] Generating Aries... "` is buffered and may be read either immediately (without the `✓`/`✗` result) or combined with the next output in unpredictable ways. This can cause:
  - Missing status feedback (user doesn't know if generation succeeded)
  - Scrambled announcements mixing lines together
  - The `✗` error indicator being merged with the following error message

- **Impact:**  
  Blind operators using screen readers cannot reliably determine success/failure status for individual sign generation without reviewing the full console buffer after completion.

- **Recommendation:**  
  Replace the split-line pattern with a single complete-line print after the operation:
  ```python
  # Before (inaccessible)
  print(f"   [{i}/{len(signs_to_generate)}] Generating {sign}...", end=" ", flush=True)
  # ... call ...
  print("✓")

  # After (screen reader friendly)
  # (perform call, capture result)
  status = "OK" if success else "FAILED"
  print(f"   [{i}/{len(signs_to_generate)}] {sign}: {status}")
  ```
  This also enables easier log parsing and piping to downstream tools.

---

#### [MEDIUM] — Emoji Characters Used as Primary Status/Severity Indicators Throughout CLI

- **SDL Phase:** Implementation
- **WCAG Criterion:** 4.1.3 Status Messages — Level AA; Best Practice (CLI Accessibility)
- **Category:** CLI Output — Screen Reader Compatibility
- **File:** `src/render/veo_horoscope_pipeline.py` (numerous lines); `src/write/horoscope_writer.py` (numerous lines)
- **Description:**  
  The CLI pipeline uses emoji symbols as the primary visual differentiator for message severity and status throughout all output:

  - `✅` / `✓` — success
  - `❌` / `✗` — failure
  - `⚠️` — warning
  - `🎬`, `📁`, `📊`, `⏳`, `🔄` — informational labels
  - `💾`, `✍️`, `🚀`, `🌆` — decorative context

  Screen reader behavior with emoji is inconsistent:
  - **NVDA** reads full Unicode name: "white heavy check mark", "cross mark" — verbose but present
  - **JAWS** may skip emoji or read shortened descriptions
  - **VoiceOver (macOS)** reads "checkmark" or full name depending on version
  - **Some terminal screen readers** (e.g., `brltty` on Linux) may strip emoji entirely

  When emoji are stripped, a message like `"❌ API request failed: {e}"` becomes `" API request failed: {e}"` — still intelligible. However, `"✗"` alone (line 94) becomes entirely silent if stripped.

- **Impact:**  
  Users of screen readers that do not announce emoji lose non-critical but useful status context. The lone `"✗"` on line 94 of `horoscope_writer.py` is particularly problematic — it is the **only indicator** of per-sign generation failure on that line.

- **Recommendation:**  
  1. Ensure all emoji-prefixed status indicators have equivalent text meaning in the same message:
     ```python
     # Instead of: print("✗")
     print("[FAILED]")
     # Or a combined approach:
     print("✗ [FAILED]")
     ```
  2. Add a `--no-emoji` CLI flag that strips emoji from all output (useful for logging and AT users):
     ```python
     p.add_argument("--no-emoji", action="store_true", help="Disable emoji in output (screen reader friendly)")
     ```
  3. Consider a `--quiet` mode that reduces output to errors and final summary only.

---

#### [MEDIUM] — No Accessibility Documentation for Downstream Video Consumers

- **SDL Phase:** Design
- **WCAG Criterion:** Best Practice — WCAG 2.1 Conformance Documentation
- **Category:** Documentation
- **File:** `README.md`
- **Description:**  
  The `README.md` provides comprehensive usage documentation for operators but contains **no information about the accessibility characteristics or limitations of the generated video output**. Specifically missing:

  1. No mention that captions are burned-in (open captions) and cannot be toggled off by viewers
  2. No mention of the `CAPTION_STYLE` environment variable or its accessibility implications
  3. No mention of the `CAPTION_FONT_SIZE` override available via environment variable
  4. No mention of WCAG 1.2.2 / 1.2.5 compliance status
  5. No guidance for operators who need to produce accessible content for regulated environments (government, education, healthcare)
  6. No mention that generated audio is AI ambient music (not narration), which is relevant for deaf/blind users

  The `CHANGES.md` file is similarly silent on accessibility considerations despite documenting the subtitle system implementation.

- **Impact:**  
  Operators deploying this tool in environments where WCAG compliance is required (e.g., public sector, educational institutions, healthcare) have no indication of the tool's accessibility limitations and may unknowingly publish non-compliant content.

- **Recommendation:**  
  Add an "Accessibility" section to `README.md` covering:
  - Caption type (burned-in), configuration options, and limitations
  - Known WCAG gaps (no audio description, no sidecar subtitle files)
  - Recommended settings for most accessible output (`--caption-style static`, font size guidance)
  - Link to WCAG 1.2 time-based media guidelines for context

---

#### [MEDIUM] — No Machine-Readable Output Mode for Automated Accessibility Monitoring

- **SDL Phase:** Implementation
- **WCAG Criterion:** Best Practice — Operator Tooling
- **Category:** CLI Output — Accessibility of Pipeline Operations
- **File:** `src/render/veo_horoscope_pipeline.py`, lines 601–663
- **Description:**  
  All pipeline status, progress, and result information is emitted exclusively as human-readable text with emoji decoration. There is no structured output option (e.g., `--json`, `--quiet`) that would allow:
  
  - Blind operators to pipe output to a screen reader without emoji noise
  - CI/CD systems to parse success/failure programmatically
  - Log aggregators to capture structured accessibility-affecting events (e.g., "caption failed")
  - Accessibility monitoring tools to detect when uncaptioned videos are published

  The `manifest.json` file does capture job results, but notably **does not include a `caption_status` field** and the `status: "done"` value does not distinguish between captioned and uncaptioned success (see HIGH finding on silent fallback).

- **Impact:**  
  Operators using screen readers in headless/server environments face unnecessary friction processing the emoji-heavy output. Accessibility regressions (e.g., captioning silently failing) are not detectable programmatically.

- **Recommendation:**  
  1. Add `--json` flag to emit a final machine-readable JSON summary to stdout (or a file).  
  2. Add `caption_status` field (`"added"`, `"failed"`, `"skipped_gcs"`) to each job in `manifest.json`.  
  3. Add `--quiet` flag to suppress all decorative/progress output, emitting only errors and the final summary.

---

### Low Findings

---

#### [LOW] — Bare `except: pass` Clause May Silently Suppress Real Captioning Errors

- **SDL Phase:** Implementation
- **WCAG Criterion:** Best Practice — Error transparency
- **Category:** Error Handling
- **File:** `src/render/subtitle_utils.py`, lines 296–300
- **Description:**  
  The cleanup of the temporary ASS subtitle file uses a bare `except: pass` clause:

  ```python
  try:
      os.unlink(ass_path)
  except:
      pass
  ```

  While this specific block is intended only to clean up a temp file, the anti-pattern has broader code quality concerns: a bare `except` catches `SystemExit`, `KeyboardInterrupt`, `MemoryError`, and other serious conditions. More relevant to accessibility: if the `ass_path` variable is uninitialized due to a prior error in `create_ass_file()`, this pattern silently ignores the failure without any diagnostic output.

- **Impact:**  
  Low direct accessibility impact. However, unlogged errors in the caption pipeline reduce operator confidence in whether captions were correctly applied. Indirectly undermines the reliability of the captioning system.

- **Recommendation:**  
  Use a specific exception type and log unexpected errors:
  ```python
  try:
      os.unlink(ass_path)
  except OSError:
      pass  # Temp file cleanup failure is non-critical
  except Exception as e:
      print(f"[WARNING] Unexpected error during ASS file cleanup: {e}")
  ```

---

#### [LOW] — README.md Uses Emoji as Decorative Feature List Markers

- **SDL Phase:** Implementation
- **WCAG Criterion:** Best Practice (WCAG 1.1.1 by analogy — non-text content)
- **Category:** Documentation
- **File:** `README.md`, lines 6–11
- **Description:**  
  The Features section uses emoji as visual bullets:
  ```markdown
  - 🌟 **AI-Powered Script Writing**: ...
  - 🎬 **AI Video Generation**: ...
  - 📱 **Multiple Formats**: ...
  - 🎨 **Customizable Styles**: ...
  - ⚙️ **Flexible Parameters**: ...
  ```
  Screen readers announce these emoji by name (e.g., "glowing star", "clapper board") before the feature description. While not a barrier, this adds verbosity and mildly disrupts the reading flow for screen reader users.

- **Impact:**  
  Low: Decorative emoji in documentation are a minor verbosity annoyance for screen reader users, not a functional barrier.

- **Recommendation:**  
  This is low priority. If addressing, replace emoji with plain text bullets (`-`) or use emoji only in contexts where they add semantic meaning. Alternatively, in rendered GitHub Markdown, emoji are generally acceptable as decorative elements.

---

#### [LOW] — `--no-audio` Flag Help Text Does Not Describe Accessibility Implications

- **SDL Phase:** Implementation
- **WCAG Criterion:** Best Practice — Documentation
- **Category:** CLI — Operator Guidance
- **File:** `src/render/veo_horoscope_pipeline.py`, lines 557–559
- **Description:**  
  The `--no-audio` argument help string reads: `"Disable audio generation (Veo 3+ only)"`. This is technically accurate but does not communicate the accessibility implication: **disabling audio removes the only audio channel available to blind or deaf-blind users**, making the video entirely inaccessible without separate narration or transcript.

  Operators unfamiliar with accessibility requirements might use `--no-audio` for bandwidth/cost reasons without realizing they are producing content that fails WCAG 1.2.3.

- **Impact:**  
  Low: This is a documentation gap rather than a functional bug. The flag still works correctly.

- **Recommendation:**  
  Update the help text:
  ```python
  help="Disable audio generation (Veo 3+ only). NOTE: Disabling audio removes all sound; ensure a text transcript is available for accessibility compliance."
  ```

---

#### [LOW] — Output File Names Use Only Lowercase Sign Names Without Date Context

- **SDL Phase:** Implementation
- **WCAG Criterion:** Best Practice — Assistive Technology Compatibility
- **Category:** Output File Naming
- **File:** `src/render/veo_horoscope_pipeline.py`, lines 177, 186
- **Description:**  
  Generated videos are named `aries.mp4`, `taurus.mp4`, etc. (lowercase zodiac sign). While machine-readable and consistent, the filenames contain no date context. When multiple runs are performed:
  - Files from different dates overwrite each other unless `--out` is changed
  - Screen reader users browsing a file system hear only the sign name with no temporal context

  The `--out` flag allows specifying a dated directory, but this is not the default behavior and is not enforced.

- **Impact:**  
  Low: Assistive technology can read the filenames correctly. The primary concern is data loss (overwrite) rather than an AT compatibility issue. The manifest.json captures the date.

- **Recommendation:**  
  Include the date in output filenames by default: `aries_2025-07-14.mp4`. Alternatively, auto-suffix the output directory with the date when `--out` is not explicitly specified (e.g., `./out/2025-07-14/`).

---

#### [LOW] — Informational `--list-models` Output Not Formatted for Screen Reader Scanning

- **SDL Phase:** Implementation
- **WCAG Criterion:** Best Practice — CLI Accessibility
- **Category:** CLI Output
- **File:** `src/render/veo_horoscope_pipeline.py`, lines 605–613
- **Description:**  
  The `--list-models` output uses Python string left-justification (`:<35`, `:<20`) to create visual column alignment:
  ```python
  print(f"  {model_id:<35} - {model_name}")
  ```
  This produces visually aligned columns in a fixed-width terminal, but screen readers read the content linearly, resulting in the padding spaces being announced (as "blank" or ignored) between the model ID and description. The output is still intelligible but slightly noisy.

- **Impact:**  
  Minimal. Screen readers handle padded text adequately. This is a polish/best-practice item.

- **Recommendation:**  
  Consider `f"  {model_id}: {model_name}"` (colon separator) rather than visual column alignment, which reads naturally in linear text mode while remaining readable visually.

---

## Quick Wins

The following **5 fixes** offer the highest accessibility improvement for the lowest implementation effort:

| Priority | Fix | Estimated Effort | WCAG Impact |
|----------|-----|-----------------|-------------|
| 1 | **Change default `CAPTION_STYLE` from `"scroll"` to `"static"`** in `subtitle_utils.py` line 242 and `veo_horoscope_pipeline.py` line 200. One-line change. | 10 minutes | 1.2.2, Cognitive |
| 2 | **Add `caption_status` field to `VideoJob` and manifest.json**, set to `"failed"` when ffmpeg captioning fails. Prevents silent uncaptioned video publication. | 1–2 hours | 1.2.2 Level A |
| 3 | **Export a sidecar `.srt` file** alongside each `.mp4` using the existing `create_srt_file()` function in `reference/utils.py` (already implemented, just not called). Copy that function to `subtitle_utils.py` and save to `renders/<sign>.srt`. | 2–3 hours | 1.2.2 Level A |
| 4 | **Make font size resolution-aware** (`int(video_height * 0.05)`) in `add_caption_to_video()`. Guarantees legibility at 1080p. | 30 minutes | 1.4.3 (legibility) |
| 5 | **Add an accessibility section to `README.md`** describing caption type, known limitations, and recommended settings. Zero code change. | 30 minutes | Documentation |

---

## Conclusion

The `ai-video-agent` pipeline demonstrates a good foundation for accessible video output — it actively produces captions via ffmpeg and ASS subtitles, preserves audio streams, and generates structured manifest output. These are positive design choices that indicate accessibility awareness.

However, **three significant gaps** undermine that foundation in a production context:

1. The **GCS URI code path** leaves an entire class of generated videos completely uncaptioned, with no operator warning.
2. The **"scroll" default** for captions produces kinetic text that creates a significant barrier for users with cognitive, reading, and low-vision disabilities.
3. The **absence of an audio description track** means blind users cannot access the horoscope content through the video.

Addressing the two Critical and High findings — particularly items 1, 2, and 3 — would bring the pipeline substantially closer to WCAG 2.1 Level A conformance for its video output. The quick wins listed above can be implemented with low effort and would materially improve accessibility for the end consumers of these videos.

---

## Appendix — Files Audited

| File | Lines | Notes |
|------|-------|-------|
| `src/main.py` | 5 | Stub entry point, no accessibility-relevant content |
| `src/render/veo_horoscope_pipeline.py` | 664 | Primary pipeline, CLI, VeoClient, HoroscopeVeoPipeline |
| `src/render/subtitle_utils.py` | 312 | Caption/subtitle generation and ffmpeg integration |
| `src/write/horoscope_writer.py` | 133 | OpenAI horoscope generation, CLI output |
| `README.md` | 169 | User documentation |
| `CHANGES.md` | 133 | Change log |
| `pyproject.toml` | 78 | Build configuration |
| `reference/utils.py` | 425 | Reference subtitle utilities (not deployed, informational) |

---

*Report generated by automated accessibility audit | SDL-aligned | WCAG 2.1 reference*
