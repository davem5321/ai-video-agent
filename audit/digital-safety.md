# Digital Safety Audit Report — AI Video Agent

**Date**: 2025-07-14  
**Auditor**: Digital Safety Audit System (Microsoft SDL-aligned)  
**Project**: `ai-video-agent` — Automated Horoscope Video Generation Pipeline  
**Repository**: `C:\Users\davem\repos\ai-video-agent`  
**Audit Scope**: AI Content Safety, Prompt Injection, Misinformation, Content Moderation, Deceptive Design, User Wellbeing, Rate Limiting, Transparency, Error Handling, Data Integrity

---

## Executive Summary

The `ai-video-agent` project is a Python automation pipeline that generates short horoscope videos using OpenAI GPT for text and Google Vertex AI Veo for video. The pipeline is currently a **developer/internal tool** with no end-user interface — videos are written to a local `out/` directory. There is no direct consumer-facing delivery mechanism in this codebase.

Despite the restricted audience today, the README explicitly targets social media distribution (TikTok/Reels, YouTube, Instagram aspect ratios), which significantly raises the stakes for content safety. The pipeline is designed to generate and publish AI content at scale without any human review gate.

**10 findings were identified: 2 Critical, 3 High, 3 Medium, 2 Low.**

The most significant risks are:

1. **AI-generated text is injected without sanitisation into ASS subtitle format files**, which can cause unintended (or manipulated) subtitle rendering.
2. **LLM output containing Python format-string tokens (`{key}`) will crash the video prompt builder** with an unhandled `KeyError`.
3. **Generated videos carry no AI disclosure** — no watermark, no metadata, no on-screen label — which is a legal and ethical transparency risk when distributed to social media audiences.
4. **There is no safety filtering layer** between the LLM output and the final video artefact.

---

## Product Type & Audience Assessment

| Dimension | Assessment |
|---|---|
| **Product type** | B2D (developer tool) / AI content pipeline |
| **Current audience** | Developers and pipeline operators |
| **Intended downstream audience** | Social media consumers (TikTok, Reels, YouTube, Instagram) |
| **Monetisation** | None detected in codebase |
| **User-generated content** | None — all content is AI-generated |
| **Direct user interaction** | CLI only; no web UI, no accounts, no sessions |
| **Vulnerable populations** | Downstream social media viewers, including minors, and individuals susceptible to belief in astrological predictions |
| **Content type** | Speculative/predictive text (horoscopes) rendered as authoritative short-form video |

---

## Compulsive Design Inventory

This is a backend pipeline with no direct user interface. The standard compulsive design patterns (streak counters, variable-ratio rewards, infinite scroll, push notifications) are **not present** in this codebase.

| Pattern | Present | Notes |
|---|---|---|
| Reward loops / streaks | ❌ No | No user-facing UI |
| Variable-ratio rewards | ❌ No | No game/reward mechanics |
| Loss aversion / expiring progress | ❌ No | — |
| Infinite scroll | ❌ No | — |
| Auto-play | ❌ No | — |
| Artificial urgency | ❌ No | — |
| Re-engagement triggers | ❌ No | — |
| Session management / break reminders | ❌ No | — |

**Compulsive design risk: LOW** for this codebase directly. Note: the *output content* (short-form vertical videos for TikTok/Reels) is distributed to platforms where compulsive design is endemic. This pipeline acts as a content supplier to those platforms.

---

## Dark Pattern Checklist

| Pattern | Present | Notes |
|---|---|---|
| Confirmshaming | ❌ No | No UI |
| Roach motel (subscribe/cancel asymmetry) | ❌ No | No subscriptions |
| Hidden costs | ❌ No | — |
| Misdirection / forced defaults | ❌ No | — |
| Privacy zuckering | ❌ No | No consent flows |
| Disguised ads | ❌ No | — |
| Forced continuity | ❌ No | — |

**Dark pattern risk: LOW** for this codebase directly.

---

## Photosensitivity & Motion Assessment

No animations, flashing effects, or browser/UI rendering exist in this Python pipeline. Generated video content (via Veo) is not authored by this codebase and Veo's own content safety filters govern video output. The subtitle overlay in `subtitle_utils.py` uses a slow scrolling animation (bottom-to-top over 8 seconds), which is not a photosensitivity risk.

**Photosensitivity risk: LOW** (not applicable to this pipeline layer; risk exists at the Veo content generation level which is outside this codebase's control).

---

## Age-Appropriateness Assessment

The system prompt (`SYSTEM_STYLE`) specifies "Keep it PG, approachable, and a little playful," which is the sole age-appropriateness control in the pipeline. There is no age-gating, audience targeting restriction, or COPPA/GDPR Art. 8 compliance mechanism. However, as a developer tool producing content for social platforms, age-appropriateness compliance is delegated to the platform's own policies. The content type (daily horoscopes) is generally low-risk for minors, but **the absence of any AI-disclosure label** is an age-appropriateness concern since younger audiences are disproportionately susceptible to anthropomorphising AI content and taking predictions literally.

---

## Findings

---

### [CRITICAL] — ASS Subtitle Format Injection via Unsanitised LLM Output

- **Severity**: Critical
- **SDL Phase**: Implementation
- **Category**: AI Content Safety / Prompt Injection
- **File**: `src/render/subtitle_utils.py`, line 115; `src/render/veo_horoscope_pipeline.py`, line 188–201

**Description**:

LLM-generated horoscope text is passed directly into an ASS (Advanced SubStation Alpha) subtitle file with only newline characters sanitised:

```python
# subtitle_utils.py line 115 — only newlines are handled
cleaned_text = text.replace('\r\n', '\\N').replace('\r', '\\N').replace('\n', '\\N')
```

The cleaned text is then written verbatim into the ASS `Dialogue:` line:

```
Dialogue: 0,{start},{end},Default,,0,0,0,,{\move(960,720,960,50)}{cleaned_text}
```

ASS subtitle format uses `{...}` curly-brace blocks as **override tag sequences** that control font, colour, position, animation, and other rendering attributes. If the LLM generates — or is manipulated via prompt injection to generate — text containing curly-brace sequences, this can:

- **Corrupt the subtitle display** (e.g., `{bold}` suppresses the rest of the caption)
- **Override positioning and style** (e.g., `{\pos(0,0)\c&H0000FF&}` moves text and changes colour)
- **Inject arbitrary ASS commands** that FFmpeg will execute during subtitle burning

Since the horoscope text is also passed to `str.format()` in `ScenePlanner.build_scene()` (see Critical finding #2), a prompt injection chain exists where maliciously crafted content could affect both the video generation prompt and the subtitle rendering.

**Affected Users**: All downstream viewers of the generated videos.

**Recommendation**:  
Strip or escape curly braces in LLM output before writing to ASS files. At minimum:

```python
# Escape ASS override tags — replace { and } with fullwidth equivalents or remove them
cleaned_text = text.replace('{', '').replace('}', '')
cleaned_text = cleaned_text.replace('\r\n', '\\N').replace('\r', '\\N').replace('\n', '\\N')
```

A more robust approach is to use a subtitle library that handles escaping, or to validate that no ASS control sequences exist in LLM output before writing.

---

### [CRITICAL] — LLM Output Containing Format Tokens Crashes Video Prompt Builder

- **Severity**: Critical
- **SDL Phase**: Implementation
- **Category**: AI Content Safety / Error Handling
- **File**: `src/render/veo_horoscope_pipeline.py`, lines 385–391

**Description**:

`ScenePlanner.build_scene()` calls Python's `str.format()` on the `DEFAULT_TEMPLATE`, passing the raw LLM-generated horoscope text as the `caption` parameter:

```python
# veo_horoscope_pipeline.py lines 384–391
def build_scene(self, sign: str, text: str, template: str = DEFAULT_TEMPLATE) -> SceneSpec:
    prompt = template.format(
        aspect=self.render.aspect_ratio,
        sign=sign,
        caption=text,          # ← raw LLM output inserted here
        seconds=self.render.seconds,
    )
```

If the LLM response contains curly-brace tokens that are not valid Python format-string keys (e.g., `{love}`, `{career}`, `{3}`, `{!s}`), `str.format()` will raise an unhandled `KeyError` or `ValueError`, crashing the pipeline mid-run. This is not a theoretical edge case: horoscope text frequently references concepts in braces when the model uses emoji-like formatting or when a prompt injection causes the model to emit format-like syntax.

The `DEFAULT_TEMPLATE` itself has `{caption}` commented out (line 359), but the `caption` keyword argument is still passed to `str.format()`, meaning any `{...}` token in `text` will be treated as a format key lookup. If `caption` were ever uncommented and the horoscope text contained `{something}`, the entire `text` value including `{something}` would cause a crash.

**Affected Users**: Pipeline operators; indirect impact — all 12 horoscope videos fail silently when the crash occurs mid-batch.

**Recommendation**:  
Sanitise curly braces in the LLM output **before** calling `str.format()`, or switch to a safe string interpolation approach that does not treat the content as a format template:

```python
# Option 1: sanitise before format (also fixes ASS injection above)
safe_text = text.replace('{', '{{').replace('}', '}}')
prompt = template.format(..., caption=safe_text, ...)

# Option 2: use Template from string module (safer for untrusted input)
from string import Template
prompt = Template(template).safe_substitute(aspect=..., caption=text, ...)
```

---

### [HIGH] — No Content Safety Filter on LLM-Generated Horoscope Text

- **Severity**: High
- **SDL Phase**: Design
- **Category**: AI Content Safety / Content Moderation
- **File**: `src/write/horoscope_writer.py`, lines 75–96

**Description**:

The OpenAI API response is consumed and persisted without any content validation:

```python
# horoscope_writer.py lines 88–90
resp = client.chat.completions.create(**api_params)
text = (resp.choices[0].message.content or "").strip()
results[sign] = text  # ← saved directly with no safety check
```

The system prompt (`SYSTEM_STYLE`) instructs the model to "Keep it PG, approachable, and a little playful," but this is a stylistic request, not a technical safety control. Relying on a model instruction alone is insufficient because:

1. **Model safety filters can be bypassed or fail** — GPT models with temperature 0.8 are more creative and more likely to produce edge-case outputs.
2. **No rejection of harmful content types** is implemented: the pipeline has no check for predictions about health, death, financial decisions, self-harm, relationship abuse, or other psychologically harmful themes that a horoscope framing could invoke.
3. **OpenAI's Moderation API** (`openai.moderations.create`) is available and free, but is not called.
4. The pipeline **saves every output to disk** (`save_horoscopes`) and **burns it into video subtitles** — meaning harmful content propagates through the full pipeline to a distributable artefact.

**Affected Users**: Downstream social media viewers; particularly vulnerable individuals (e.g., those in crisis, susceptible to superstition or false authority).

**Regulatory Reference**: EU AI Act Art. 52 (transparency obligations for AI-generated content); UK Online Safety Act (harmful content provisions).

**Recommendation**:  
Add an OpenAI Moderation API call after each horoscope is generated:

```python
from openai import OpenAI

def is_safe_content(client: OpenAI, text: str) -> bool:
    response = client.moderations.create(input=text)
    return not response.results[0].flagged

# In generate_daily_horoscopes(), after line 89:
text = (resp.choices[0].message.content or "").strip()
if not is_safe_content(client, text):
    print(f"⚠️  Content flagged for {sign}, using safe fallback")
    text = f"Today brings gentle energy for {sign}. Focus on what brings you peace."
results[sign] = text
```

Additionally, add keyword-based rejection for high-risk phrases (medical advice, financial predictions with certainty, references to death or harm).

---

### [HIGH] — Generated Videos Carry No AI-Generated Content Disclosure

- **Severity**: High
- **SDL Phase**: Design
- **Category**: Transparency / Deceptive Design
- **File**: Design-level; `src/render/veo_horoscope_pipeline.py`, `src/render/subtitle_utils.py`

**Description**:

Videos produced by this pipeline contain no indication that they are AI-generated. Specifically:

1. **No on-screen watermark or label** — the subtitle system renders only the horoscope text. There is no persistent "AI-generated" overlay, disclaimer badge, or watermark.
2. **No MP4 metadata** — no `comment`, `description`, or custom metadata tags are written to the output files indicating AI provenance.
3. **No disclaimer text in the horoscope content** — the `SYSTEM_STYLE` prompt does not instruct the model to include "for entertainment only" language, and the pipeline adds none during post-processing.
4. **The `DEFAULT_TEMPLATE` prompt description actively constructs a realistic aesthetic** ("Cinematic shot," "cozy neon-lit studio," "slow push-in, shallow depth of field") that gives AI-generated videos the appearance of professionally produced human content.

Notably, the `DEFAULT_TEMPLATE` (line 359) has the `{caption}` subtitle line **commented out**, meaning the AI-generated horoscope text is not even visible in the Veo video generation prompt. This could result in Veo producing videos that have no horoscope content visible at all, with the horoscope text being added only via the subtitle overlay — giving the impression it was a human-voiced or human-captioned video.

When distributed to platforms like TikTok, YouTube, or Instagram, viewers have no way to know the content was generated by an AI pipeline, which:
- Violates the **EU AI Act Art. 50** requirement to label AI-generated synthetic media
- Contradicts the **UK Online Safety Act** duty of care for user clarity about content authenticity
- May violate individual platform policies that require AI-content labeling

**Affected Users**: All downstream social media viewers; particularly consumers of astrology content who may ascribe personal meaning to predictions.

**Recommendation**:  
Implement a three-layer disclosure strategy:

1. **On-screen label**: Add a persistent small-text overlay ("AI-generated for entertainment only") via a second ASS subtitle dialogue event.
2. **MP4 metadata**: Write `comment` and `description` tags via FFmpeg:
   ```
   -metadata comment="AI-generated content" -metadata description="Created by AI Video Agent"
   ```
3. **Horoscope text disclaimer**: Add to `SYSTEM_STYLE`: *"Always end with a brief note that this is for entertainment purposes only."* Or append a fixed disclaimer during `save_horoscopes()`.

---

### [HIGH] — Silent Placeholder Substitution Presents Error Content as Genuine Output

- **Severity**: High
- **SDL Phase**: Implementation
- **Category**: AI Content Safety / Error Handling / Misinformation
- **File**: `src/write/horoscope_writer.py`, line 96

**Description**:

When the OpenAI API call fails for any reason (quota exhaustion, network error, model unavailability), the pipeline silently substitutes a hardcoded placeholder:

```python
# horoscope_writer.py line 96
except Exception as e:
    print(f"✗")
    print(f"       ERROR: {str(e)}")
    results[sign] = f"({sign} placeholder horoscope: Today is a lucky day! 🌟)"
```

This placeholder:
1. **Is saved to disk** as a `.txt` file and inside `horoscopes.json`, appearing indistinguishable from a real generated horoscope.
2. **Is burned into a video** as a subtitle caption via `add_caption_to_video()`.
3. **Contains no indication it is error fallback content** — it looks like a horoscope to any downstream consumer.
4. The pipeline **prints the error to stdout** but **does not propagate the failure state** — the manifest `status` for this job will still show `"done"` if the video generation succeeds.
5. The error is only printed to the console; there is no structured error flag in the JSON output that would allow automated detection of degraded content.

A video published to social media with placeholder horoscope text ("Today is a lucky day! 🌟") as a subtitle would be factually misleading about its origin and quality.

**Affected Users**: Pipeline operators (who may not review every generated file); downstream viewers who receive a non-personalised placeholder as a horoscope prediction.

**Recommendation**:  
On API failure, either abort that sign's processing and skip video generation, or use a content-tagged fallback that is clearly marked:

```python
except Exception as e:
    print(f"✗ ERROR: {str(e)}")
    results[sign] = None   # or raise, or use a flag
    # Do NOT silently substitute placeholder content

# In the pipeline, skip signs with None content:
scripts = {k: v for k, v in scripts.items() if v is not None}
```

If a fallback is required, ensure it is tagged in the manifest as `"content_source": "fallback"` so operators can identify and suppress distribution.

---

### [MEDIUM] — System Prompt Requests "Concrete Predictions" Without Entertainment Framing

- **Severity**: Medium
- **SDL Phase**: Design
- **Category**: Misinformation / User Wellbeing
- **File**: `src/write/horoscope_writer.py`, lines 27–31 and 70–73

**Description**:

The `SYSTEM_STYLE` prompt is:
> *"You are a witty, positive horoscope writer. Write concise, 2–3 sentence daily horoscopes with a clear prediction or action. Keep it PG, approachable, and a little playful."*

The `user_prompt` additionally instructs:
> *"Include a concrete prediction or recommended action for today."*

The word **"concrete prediction"** in the user prompt actively encourages the model to produce authoritative-sounding, specific claims about the future. Reviewing the generated sample content in `data/horoscopes/2025-12-23/horoscopes.json`, this produces outputs like:

> *"Dive into a creative project or tackle that task you've been putting off"* (Aries)  
> *"Consider hosting a cozy gathering with friends tonight"* (Libra)  
> *"Tackle a lingering project or have that heart-to-heart you've been putting off"* (Scorpio)

These are presented as personal, daily, sign-specific instructions. Without any disclaimer, a viewer unfamiliar with AI horoscopes — or one who holds genuine beliefs in astrology — may act on these instructions as if they were authoritative personal guidance.

There is no "for entertainment only" framing in the system prompt, the generated text, or the video output. This is a **misinformation risk** and a **user wellbeing risk**, particularly for:
- Individuals who take astrological content seriously
- Vulnerable users who are seeking guidance during difficult life periods
- Younger viewers on social media platforms

**Affected Users**: General social media audience; disproportionate risk for vulnerable adults and credulous younger audiences.

**Regulatory Reference**: EU DSA Art. 27 (algorithmic recommender systems and vulnerable users); UK Online Safety Act (priority harmful content definitions).

**Recommendation**:  
Modify the system prompt to include explicit entertainment framing:

```python
SYSTEM_STYLE = (
    "You are a witty, creative horoscope writer producing entertainment content. "
    "Write concise, 2–3 sentence daily horoscopes that are fun and uplifting. "
    "Avoid giving specific life advice or predictions that could be taken as authoritative guidance. "
    "Keep it PG, approachable, playful, and clearly fictional. "
    "Always frame content as lighthearted entertainment, not predictions."
)
```

And change the user prompt's "Include a concrete prediction" to "Include a fun, lighthearted suggestion."

---

### [MEDIUM] — No Safety Review Gate on Veo Video Generation Prompts

- **Severity**: Medium
- **SDL Phase**: Design
- **Category**: AI Content Safety / Content Moderation
- **File**: `src/render/veo_horoscope_pipeline.py`, lines 420–428

**Description**:

The video generation prompt sent to the Veo API is built from a template and transformers without any safety review:

```python
# veo_horoscope_pipeline.py lines 420–428
for i, scene in enumerate(scenes, 1):
    for t in self.transformers:
        scene.prompt = t(scene.prompt, scene)
    (out_dir / "prompts" / f"{scene.sign}.txt").write_text(scene.prompt, encoding="utf-8")
```

The `PromptTransformer` protocol (lines 363–365) allows arbitrary callables to modify the prompt. The `CyberpunkPunchup` transformer (lines 371–374) appends visual style instructions. There is no validation that the final prompt:
1. Does not contain harmful visual requests (violence, sexual content, dangerous scenarios)
2. Does not include content that violates Veo's usage policies
3. Has not been inadvertently corrupted by LLM output that bled into the prompt template

While Google Vertex AI Veo has its own built-in content filters, relying solely on provider-side filtering means:
- Policy violations are discovered only after API submission (costing API credits and time)
- No local audit record of what was submitted
- No opportunity to review borderline content before it is sent

**Affected Users**: Pipeline operators (API quota consumption, policy violations); downstream viewers.

**Recommendation**:  
Add a prompt validation step before API submission:

```python
def validate_veo_prompt(prompt: str) -> tuple[bool, str]:
    """Basic safety check on video generation prompt before submission."""
    blocked_terms = ["violence", "blood", "sexual", "nude", "harm", "weapon", ...]
    lower = prompt.lower()
    for term in blocked_terms:
        if term in lower:
            return False, f"Blocked term detected: {term}"
    if len(prompt) > 2000:
        return False, "Prompt exceeds safe length limit"
    return True, "OK"
```

Additionally, log all submitted prompts with timestamps to a local audit file for review.

---

### [MEDIUM] — No Rate Limiting, Quota Guard, or Abuse Prevention on Bulk Generation

- **Severity**: Medium
- **SDL Phase**: Design
- **Category**: Rate Limiting / Abuse Prevention
- **File**: `src/write/horoscope_writer.py`, line 98; `src/render/veo_horoscope_pipeline.py`, lines 473–480

**Description**:

The pipeline can be run without any constraints on:
- **How many times per day** it can be executed
- **How many signs** are processed (the `--signs` flag allows any subset, but there is no upper bound guard)
- **Total API spending** per run or per day
- **Bulk/automated abuse**: the CLI can be scripted to generate content continuously

The only throttle present is a cosmetic `time.sleep(0.3)` between OpenAI API calls (line 98), which is explicitly described as "Tiny delay to be polite on rate limits." This is not an abuse prevention control.

If this pipeline or its configuration is accessible to more than one operator, or if credentials are shared or compromised, there is no circuit breaker to prevent runaway spending or bulk harmful-content generation attempts.

The `--signs` flag also accepts arbitrary strings from the command line that are passed directly to the OpenAI API as sign names — if an operator passes unusual values, this could result in unexpected prompt content (e.g., `--signs "Ignore previous instructions"`).

**Affected Users**: Pipeline operators (API cost exposure); platform audiences (bulk low-quality content).

**Recommendation**:  
1. Add a daily run counter (persisted to disk) that limits runs to a configurable maximum (e.g., 3 per day by default).
2. Validate `--signs` against the known `ZODIAC_SIGNS` list before use (already partially done, but the list is not enforced as a strict allowlist in the CLI).
3. Add a `--dry-run` cost estimator that shows projected API costs before execution.
4. Consider adding a `MAX_SIGNS_PER_RUN` guard.

---

### [LOW] — No Structured Audit Log of AI-Generated Content

- **Severity**: Low
- **SDL Phase**: Requirements
- **Category**: AI Content Safety / Transparency
- **File**: Design-level

**Description**:

The pipeline produces no persistent, structured audit log recording:
- Which model generated each horoscope
- The exact prompt used
- The timestamp of generation
- Whether content was a live API response or a fallback placeholder
- Whether any errors occurred during generation

The `manifest.json` (line 488–506) records job metadata for the video generation step, but there is no equivalent record for the horoscope writing step. The `save_horoscopes()` function saves the text outputs but not the model, prompt parameters, or generation metadata.

This matters because:
1. If a harmful piece of content is generated and distributed, there is no audit trail to determine which model version and parameters produced it.
2. Fallback placeholder content is indistinguishable from real generated content in the saved files.
3. Accountability for AI-generated content is increasingly required by regulation.

**Regulatory Reference**: EU AI Act Art. 12 (record-keeping for high-risk AI systems); EU DSA Art. 17 (complaints and redress mechanisms requiring traceability).

**Recommendation**:  
Extend `save_horoscopes()` to write a generation metadata record alongside the content:

```json
{
  "generated_at": "2025-07-14T10:30:00Z",
  "model": "gpt-5-nano",
  "temperature": 0.8,
  "date": "2025-07-14",
  "signs": {
    "Aries": { "text": "...", "source": "api", "flagged": false },
    "Taurus": { "text": "...", "source": "fallback", "flagged": false }
  }
}
```

---

### [LOW] — Google API Key Transmitted as URL Query Parameter

- **Severity**: Low
- **SDL Phase**: Implementation
- **Category**: AI Content Safety / Abuse Prevention (secondary)
- **File**: `src/render/veo_horoscope_pipeline.py`, lines 51–54

**Description**:

The Google API key is appended as a query parameter to the Veo API URL:

```python
url = (f"https://{self.region}-aiplatform.googleapis.com/v1/"
       f"projects/{self.project_id}/locations/{self.region}/"
       f"publishers/google/models/{self.model_id}:predictLongRunning"
       f"?key={self.api_key}")   # ← API key in URL
```

While HTTPS encrypts the transport, URL query parameters are routinely captured in:
- Web server access logs
- Proxy logs
- Browser history (if ever called from a web context)
- Network monitoring tools
- Error messages that include the full URL

API key exposure enables the pipeline to be abused by a third party to generate content at the operator's expense — including potentially harmful content — which is a digital safety risk in addition to a security risk.

**Recommendation**:  
Move the API key to the `Authorization` HTTP header instead of the URL:

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {self.api_key}"
}
r = requests.post(url_without_key, json=payload, headers=headers, timeout=60)
```

Alternatively, use Google Application Default Credentials (ADC) via `google-auth` library, which is already listed in `requirements.txt`.

---

## Positive Safety Controls Already Present

The following responsible design elements were identified and are acknowledged positively:

| Control | Location | Notes |
|---|---|---|
| PG content instruction in system prompt | `horoscope_writer.py` line 31 | Basic content tone control |
| API key loaded from `.env`, not hardcoded | `horoscope_writer.py` line 35–37 | Prevents credential exposure in source |
| Exception handling on API calls | `horoscope_writer.py` lines 92–96 | Pipeline does not crash on API failure |
| Test mode (`--test-mode` flag) | `veo_horoscope_pipeline.py` line 431 | Allows review before submitting to Veo |
| Prompt files saved to disk | `veo_horoscope_pipeline.py` line 427 | Creates a partial audit trail of prompts |
| Rate-limit courtesy sleep | `horoscope_writer.py` line 98 | Reduces accidental API hammering |
| `--signs` flag for subset testing | `veo_horoscope_pipeline.py` line 588 | Allows staged/partial runs |
| FFmpeg `-map 0:a?` optional audio | `subtitle_utils.py` line 214 | Graceful handling of missing audio streams |
| Output directory created safely | `veo_horoscope_pipeline.py` line 407 | `mkdir(parents=True, exist_ok=True)` |

---

## Summary Table

| # | Severity | Category | File | SDL Phase |
|---|---|---|---|---|
| 1 | **Critical** | AI Content Safety / Prompt Injection | `subtitle_utils.py:115`, `veo_horoscope_pipeline.py:188` | Implementation |
| 2 | **Critical** | AI Content Safety / Error Handling | `veo_horoscope_pipeline.py:385` | Implementation |
| 3 | **High** | AI Content Safety / Content Moderation | `horoscope_writer.py:88–96` | Design |
| 4 | **High** | Transparency / Deceptive Design | Design-level | Design |
| 5 | **High** | Misinformation / Error Handling | `horoscope_writer.py:96` | Implementation |
| 6 | **Medium** | Misinformation / User Wellbeing | `horoscope_writer.py:27–31, 70–73` | Design |
| 7 | **Medium** | AI Content Safety / Content Moderation | `veo_horoscope_pipeline.py:420–428` | Design |
| 8 | **Medium** | Rate Limiting / Abuse Prevention | `horoscope_writer.py:98`, pipeline CLI | Design |
| 9 | **Low** | AI Content Safety / Transparency | Design-level | Requirements |
| 10 | **Low** | Abuse Prevention | `veo_horoscope_pipeline.py:51–54` | Implementation |

---

## Recommended Remediation Priority

### Immediate (before any social media distribution)

1. **Fix ASS injection** — sanitise `{` and `}` from LLM output before writing subtitle files (Finding #1)
2. **Fix `str.format()` crash** — escape curly braces in LLM output before template interpolation (Finding #2)
3. **Add AI disclosure** — on-screen label + MP4 metadata (Finding #4)
4. **Fix silent placeholder** — do not silently substitute error content as real horoscope text (Finding #5)

### Short-term (before production deployment)

5. **Add OpenAI Moderation API call** on all generated text (Finding #3)
6. **Revise system prompt** to remove "concrete prediction" language and add entertainment disclaimer (Finding #6)
7. **Add structured generation audit log** (Finding #9)

### Medium-term (operational hardening)

8. **Add Veo prompt safety validation** before API submission (Finding #7)
9. **Add rate limiting and abuse prevention controls** (Finding #8)
10. **Move API key from URL to Authorization header** (Finding #10)

---

## Conclusion

The `ai-video-agent` pipeline is a capable and well-structured developer tool, but it was designed primarily for functional correctness rather than content safety. As it stands, the pipeline **should not be used to generate content for public distribution** without addressing at minimum the two Critical findings and the AI disclosure gap.

The most urgent architectural gap is the **absence of any human or automated review gate** between LLM output and the final distributed video. Every piece of generated content flows from GPT → disk → subtitle → Veo → MP4 → ready-to-publish, with no point at which safety, accuracy, or disclosure is verified. Adding an OpenAI Moderation API step and a mandatory `--review` flag before distribution would substantially reduce the content safety risk profile.

The speculative nature of horoscope content makes transparency especially important. Viewers who encounter a polished, cinematic, AI-generated video with a specific personal prediction have no reason to treat it as anything other than authoritative content — and the current pipeline design reinforces rather than counters that misperception.

---

*This report was generated by a digital safety audit system aligned with the Microsoft Security Development Lifecycle (SDL). Findings are for remediation purposes and do not constitute legal advice. Regulatory references are indicative and should be validated against current legislation applicable to your jurisdiction and distribution channels.*
