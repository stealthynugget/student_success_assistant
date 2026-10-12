# Console design plan

**Subject.** A support workbench for Open University module teams: tutors, student-support advisors and module leads.
**Primary job.** Decide who to contact this week, understand why, and send the right message — with evidence.

## Concept: the course clock
Everything in OULAD is anchored to a day in a course presentation, and every prediction is only valid "as of" a point
in the course. The top band of every page is a course timeline (weeks 0–38) with assessment deadlines as ticks and the
built checkpoints as stops; the selected stop is marked like a highlighter on a printed timetable. Moving the stop
re-scopes every view. That ribbon is the one memorable element; everything else stays quiet.

## Visual language
Dark, monochrome and technical: hairline frames instead of cards, sharp 2px corners everywhere, a grotesk sans with a
mono for numbers and labels. Dark is the default theme; light and "system" are one click away (theme switch, Ctrl+K).

## Tokens (dark default; light values in `src/styles.css`)
| Role | Dark | Light |
|---|---|---|
| Ink (text, primary buttons) | `#ececea` | `#131313` |
| Paper (app background) | `#0e0e0f` | `#f2f2f0` |
| Surface / raised | `#141415` / `#1a1a1c` | `#fbfbfa` / `#f6f6f4` |
| Rule (hairlines) | `#2b2b2e` | `#d9d9d5` |
| Marker (selected checkpoint, focus ring, hover rule) | `#f2c94c` | `#f2c94c` |
| Tiers: High / Medium / Low | `#e8735f` / `#e4a74b` / `#5cb9a5` | `#b23a2b` / `#c9821a` / `#2e7466` |
| Generated text (LLM) | `#a8a6ff` left rule + "Drafted by AI" label | `#4c4bc2` |

Colour is functional: tier colours only mean tier, indigo only means "generated, review before use", yellow only means
"the point in the course you are looking at" (and keyboard focus). Computed facts are plain ink. One radius: 2px.

**Type.** Geist (UI and headings, weight 400-500, tight tracking) and Geist Mono (numbers, table headers, labels, chips),
self-hosted via Fontsource. Scale 34 / 24 / 18 / 15 / 13. No uppercase eyebrows, no middle-dot meta strings, no em-dashes.

**Layout.** Narrow left rail for places; course clock across the top (frosted, sticky); content left-aligned, max 1320px.
Sections are separated by whitespace and hairlines. Stats sit in a hairline grid, not in cards.

**Motion.** Meaningful only: the clock marker slides between checkpoints, agent steps appear as they run, numbers count
when the checkpoint changes, sections rise in as they scroll into view (CSS scroll-driven animation). Hover: the rail and
table rows show a marker rule, buttons press 1px. Everything is static under `prefers-reduced-motion`.

## Landing page (`/`)
Framed column with hairline rails (after the reference), eight sections, each a different layout: hero with a dithered
dot-field dome whose waves are real weekly activity of students who passed and withdrew (`src/landing/DitherField.tsx`,
canvas, pointer lens, static under reduced motion) and a working ask box that opens the copilot; a built-with logo strip
(Simple Icons); live numbers; a scroll-pinned course clock story; a bento product tour with real console screenshots
(`public/landing/`, regenerate after UI changes); the agent flow; measured evaluation results; closing call to action.
Every number is read live from the API; nothing is typed by hand.

## Principles
1. Every number says what it is and as of when.
2. Generated text is always labelled and never the only source of a fact.
3. Empty and error states say what to do next.
4. Keyboard and screen-reader friendly; reduced motion respected; works down to tablet width.
