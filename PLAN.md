# TokLog — Product Plan

> `htop` for your LLM spend. Local-first. Proxy-only. Zero config.

---

## What TokLog Is

A local HTTP proxy that intercepts LLM API traffic, logs usage to JSONL, attributes cost by model/provider/program/tag, and surfaces actionable waste reports. No account. No hosted backend. No prompt egress.

```
pip install toklog && tl proxy start
```

Works with any client that speaks HTTP to OpenAI/Anthropic/Gemini — Cursor, Cline, Claude Code, custom scripts, anything.

---

## What We Have Today

- Proxy server with provider adapters (OpenAI, Anthropic, Gemini)
- JSONL logging under `~/.toklog/logs/`
- Process attribution via `/proc`
- Waste detectors: context stuffing, model mismatch, cache misses, retry storms
- Context drivers: classify what structural content drives token cost
- CLI: `tl report`, `tl gain`, `tl doctor`, `tl share`
- Pricing via LiteLLM auto-refresh
- Config-driven, local-first
- 733 tests passing

---

## Market Reality

### The Pain (validated across Reddit, HN, Twitter, forums)

1. **Bill shock** — $50 to $50K surprise bills. Universal, emotional, #1 driver of tool adoption.
2. **Zero attribution** — "I know my total bill but not which feature/agent/user burned it."
3. **Runaway agents** — coding assistants and agent loops burning $50-400 before anyone notices.
4. **Multi-provider fragmentation** — no single pane of glass across OpenAI/Anthropic/Gemini.
5. **Context stuffing** — apps sending 14K tokens when 3K would do. Nobody knows until they look.
6. **Wrong model for the task** — GPT-4o for classification tasks that GPT-4o-mini handles at 1/15th the cost.
7. **Privacy anxiety** — hosted tools (Helicone, LangSmith) route prompts through third-party servers.

### The Competition

| Tool | Model | TokLog's Edge |
|---|---|---|
| Helicone | Hosted proxy, freemium | Acquired/stagnating. MITM concerns. No local option. |
| Langfuse | Open-source, OTel-based | Heavy setup, pulling upmarket toward enterprise. Requires instrumentation. |
| LangSmith | LangChain-coupled | Lock-in. Cloud-only. Hated by non-LangChain users. |
| Provider dashboards | Free, built-in | Single-provider only. No waste detection. No incentive to tell you "use a cheaper model." |
| DIY (20% of teams) | Custom logging | TokLog's real competitor. Basic logging is a weekend project; waste intelligence is not. |

### TokLog's Wedge

The only local-first, proxy-based, cross-provider cost tool that requires zero code changes. Privacy by architecture, not by policy.

---

## Target Users

**Primary — Agent power users (highest pain, fastest conversion):**
- Claude Code / Cursor / Cline / Aider users with direct API keys
- Solo devs and small teams (2-15 engineers) past prototype stage
- Anyone who's had a surprise LLM bill

**Secondary — Privacy-conscious ML engineers:**
- Won't touch hosted observability due to proprietary data / GDPR
- Startups with compliance concerns but no enterprise budget

---

## What to Build

### Phase 1 — Ship in 2 weeks (the install triggers)

| Feature | What | Why |
|---|---|---|
| **Budget enforcement / kill switch** | `tl proxy start --budget 5.00` — proxy returns 429 when spend exceeds limit. Per-session and daily budgets via `~/.toklog/budgets.yaml`. Loop detection (N requests/minute → auto-block). | #1 requested feature across all research. Prevents real money loss. The reason people install TokLog. |
| **Live terminal dashboard** | `tl watch` — top-style TUI showing requests flowing, cost accumulating, waste flags in real time. | The reason people keep TokLog running. The screenshot that sells the product. |
| **Session grouping** | Auto-group requests by PID + time window. "This coding session cost $14.20" without manual tagging. | Zero-friction attribution for the most common question. |

### Phase 2 — Ship in 30 days (the value deepeners)

| Feature | What | Why |
|---|---|---|
| **Actionable waste fixes** | Don't just flag "context stuffing detected" — output: "Your /chat endpoint stuffs 14.2K tokens avg. Completions reference last 2.8K tokens 89% of the time. Estimated savings: $18.40/day if you trim context." | The insight that justifies paying for TokLog. |
| **Model-fit scoring** | Per-endpoint analysis: "42 requests used claude-opus for simple Q&A. Switching to gpt-4o-mini would save $12.40/week." | Quantifiable savings recommendation. |
| **Minimal web dashboard** | `tl dashboard` → localhost:7745. Single HTML page, no build step. Tabs: Today / Trends / Waste / Sessions. | Shareability. Screenshots. "Wow factor." How TokLog spreads inside orgs. |
| **Webhook alerts** | Slack/webhook ping: "Your /agent endpoint just exceeded $5 in the last hour." | Bridges local-first and team visibility. |

### Phase 3 — Ship in 90 days (the moat builders)

| Feature | What | Why |
|---|---|---|
| **Routing recommendations** | `tl route-analysis` — analyze which requests overpay for model tier. Recommend optimal model per endpoint. Analysis only, no request mutation. | Step toward optimization without owning outcomes. |
| **OTel export** | Emit JSONL as OTel spans. Feed into existing Grafana/Datadog stacks. | Enterprise/team adoption. Low effort, high signal. |
| **CI integration** | `tl ci-check --max-cost-per-test 0.50` — fail the build if test suite LLM costs exceed threshold. | Stickiness. Set-and-forget guardrail. |
| **Gain tracking v2** | Before/after: "You trimmed context on /chat on March 5. Since then you've saved $412." | Addictive feedback loop. ROI proof for purchase justification. |

### Not Building (yet)

| Feature | Why not |
|---|---|
| **Smart routing (request mutation)** | Turns TokLog from observer to actor. If routing fails, users blame us. Different trust model. Revisit after routing recommendations prove demand. |
| **Semantic caching** | High complexity (embeddings, cache store), moderate payoff. 2-3 week build for one developer. Not worth it until user base exists. |
| **Team sync** | The moment we add a server, we're building Helicone. Stay single-player until growth demands it. |

---

## Monetization

### Free tier (forever)
- Logging, basic cost tracking, CLI reports
- Budget kill switch + loop detection
- `tl watch` live dashboard
- Session grouping, process attribution
- Basic waste flags (binary: detected / not detected)
- Web dashboard

### Pro tier ($19-29/mo)
- Actionable waste analysis with fix suggestions + estimated savings
- Model-fit scoring
- Routing recommendations
- Gain tracking (ROI proof)
- CI integration
- Priority waste pattern updates

**Why this split:** Free tier is genuinely useful — the `htop` that shows what's happening. Pro tier answers "now what do I do about it?" Classic problem → solution upgrade. At $29/mo, the tool pays for itself if it saves $1/day. Most users save 10-50x.

**No usage-based pricing.** Ironic to charge per-request for a cost optimization tool. Flat rate, simple, predictable.

---

## Launch Plan

### Pre-launch (Week 3)
- [ ] README is the landing page — hook → 3-line install → screenshot → comparison table → FAQ
- [ ] Record 45-second terminal demo GIF (install → proxy start → API calls → report)
- [ ] Write Show HN post (personal pain story, not corporate launch)
- [ ] Write Reddit posts tailored to r/ClaudeAI, r/cursor, r/LocalLLaMA
- [ ] Seed 10-15 GitHub stars from network
- [ ] Prepare Machine Learns newsletter issue

### Launch week (Week 4)
- [ ] **Day 1:** Show HN (Tuesday/Wednesday, 8-9am ET)
- [ ] **Day 1:** Tweet launch with demo GIF from @erogol
- [ ] **Day 2:** r/ClaudeAI + r/cursor posts
- [ ] **Day 3:** r/LocalLLaMA + r/ChatGPTCoding
- [ ] **Day 4-5:** Machine Learns newsletter
- [ ] **Day 6-7:** Blog post: "I tracked my Claude Code spending for a week. Here's what I found."

### Week 5-6 (ride the wave)
- [ ] Engage in existing Reddit/HN threads about LLM costs
- [ ] Join Discord communities (Claude Code, Cursor, LangChain, Latent Space)
- [ ] DM 10 power users who starred the repo
- [ ] Publish comparison page: TokLog vs Helicone vs Langfuse vs LiteLLM
- [ ] Collect testimonials

### Week 7-8 (compound)
- [ ] Blog: "The 5 Most Common Ways Developers Waste Money on LLM APIs"
- [ ] Twitter thread: "I analyzed 10,000 LLM API calls. Here's what most people get wrong."
- [ ] Outreach to AI newsletters (TLDR, Ben's Bites, Latent Space)
- [ ] Product Hunt launch (if 200+ stars)

### Target Metrics

| Metric | Week 4 | Week 8 |
|---|---|---|
| GitHub stars | 200 | 500+ |
| PyPI installs | 500 | 2,000+ |
| Active users | 50 | 300+ |

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Provider native dashboards get good enough** | HIGH | Ship cross-provider + waste intelligence fast. Providers won't tell users "use a cheaper model." |
| **DIY inertia** | HIGH | Ship waste detection patterns fast. "TokLog detects 23 waste patterns" changes the DIY calculus. |
| **Framework-level cost tracking absorbs it** | MEDIUM | Stay framework-agnostic. LiteLLM/LangChain adding basic cost ≠ waste intelligence. |
| **Nobody pays** | MEDIUM | Free tier must be genuinely viral. Pro tier must show quantifiable ROI. |
| **Retention craters** | MEDIUM | Budget kill switch = always-on value. `tl watch` = habit formation. |
| **Local-first limits team adoption** | LOW (for now) | Solve single-player first. Team sync is a Phase 4 problem. |

---

## The Honest Assessment

TokLog's **best case**: privacy-first LLM cost tool that 5-10K developers use, 2-5% convert to Pro = $2-15K/mo recurring. Side project revenue with <5 hrs/week maintenance.

TokLog's **worst case**: well-regarded OSS tool with 500 regular users and no paying customers. A resume project.

The gap between these outcomes is **waste intelligence quality + launch execution**. The basic proxy is commoditizable. The detection patterns, context drivers, and actionable fix suggestions are not.

**The moat is opinions about waste, not the proxy itself.**

---

## The Flywheel

```
Show HN / Reddit launch
        ↓
Users install, see their waste → "holy shit" moment
        ↓
Share screenshots / reports → organic growth
        ↓
Power users hit free tier ceiling → Pro conversion
        ↓
Pro revenue funds more detection patterns
        ↓
More patterns → more waste found → stronger "holy shit" → more sharing
```

---

*Last updated: 2026-03-27*
