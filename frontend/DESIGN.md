# Console design plan

**Subject.** A support workbench for Open University module teams: tutors, student-support advisors and module leads.
**Primary job.** Decide who to contact this week, understand why, and send the right message — with evidence.

## Concept: the course clock
Everything in OULAD is anchored to a day in a course presentation, and every prediction is only valid "as of" a point
in the course. The top band of every page is a course timeline (weeks 0–38) with assessment deadlines as ticks and the
built checkpoints as stops; the selected stop is marked like a highlighter on a printed timetable. Moving the stop
re-scopes every view. That ribbon is the one memorable element; everything else stays quiet.

## Tokens
| Role | Value |
|---|---|
| Ink (text, rules on dark) | `#172036` |
| Paper (app background, cool grey-green, not cream) | `#F5F6F3` |
| Surface | `#FFFFFF` |
| Rule | `#DCE0E4` |
| Marker (selected checkpoint, focus) | `#F4C53D` |
| Tiers: High / Medium / Low | `#B23A2B` / `#C9821A` / `#2E7466` |
| Generated text (LLM) | `#4C4BC2` left rule + "Drafted by AI" label |

Colour is functional: tier colours only mean tier, indigo only means "generated — review before use", yellow only
means "the point in the course you are looking at". Computed facts are plain ink.

**Type.** Atkinson Hyperlegible (designed for legibility; fitting for education and accessibility) for everything, with
tabular numerals. Scale 32 / 24 / 18 / 15 / 13. No uppercase labels, no eyebrows, no middle-dot meta strings.

**Layout.** Narrow left rail for places; course clock across the top; content left-aligned, max 1280px. Sections are
separated by whitespace and hairline rules rather than identical cards. Primary panels 6px radius, chips and inputs 4px.

## Principles
1. Every number says what it is and as of when.
2. Generated text is always labelled and never the only source of a fact.
3. Empty and error states say what to do next.
4. Keyboard and screen-reader friendly; reduced motion respected; works down to tablet width.
