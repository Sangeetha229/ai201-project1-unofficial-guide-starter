# The Unofficial Guide — Project 1
### Pre-Med & Public Health Professor Reviews | Texas A&M University

---

## Domain

This system covers student reviews of pre-med and public health course professors at Texas A&M University. The domain covers Rate My Professors reviews, Reddit threads, Discord messages, and official TAMU advising resources for professors in the School of Public Health, BIMS program, and Chemistry department.

This knowledge is valuable because pre-med students make GPA-critical decisions about which professors to take for Organic Chemistry, BIMS, and Public Health courses. Choosing the wrong professor for a prerequisite course can tank a GPA and hurt medical school applications. Official sources — the course catalog, faculty directory, and advisors — never tell you whether a professor curves, whether their exams align with lectures, or whether office hours are worth attending. The TAMU SPH course catalog tells you what HLTH 320 covers, not whether Prof DeLeon posts her slides the night before class and tests content she barely mentioned in lecture. Student reviews fill that gap with lived, peer-to-peer knowledge that is more actionable than anything published officially.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Rate My Professors — TAMU | Professor ratings + written reviews | https://www.ratemyprofessors.com/search/professors/1003?q=* |
| 2 | Reddit r/aggies + r/premed | Student discussion threads | https://www.reddit.com/r/aggies |
| 3 | Niche — TAMU Academics | Verified student academic reviews | https://www.niche.com/colleges/texas-a-and-m-university/academics/ |
| 4 | Coursicle — TAMU Professors | Reviews tied to specific course sections | https://www.coursicle.com/tamu/professors/ |
| 5 | TAMU Discord servers | Real-time student channel exports | https://disboard.org/servers/tag/tamu |
| 6 | Uloop TAMU Professor Ratings | TAMU-specific professor ratings | https://tamu.uloop.com/professors |
| 7 | Professors.directory — TAMU | Cross-check ratings for sparse professors | https://www.professors.directory/school/tx-texas_a_m_university_college_station/g/ |
| 8 | TAMU SPH Course Catalog | Official faculty + course descriptions | https://catalog.tamu.edu/undergraduate/public-health/ |
| 9 | TAMU Pre-Medical Society | Pre-med org guidance + recommendations | https://getinvolved.tamu.edu/org/tamupremed |
| 10 | TAMU OPSA | Official pre-med advising office resources | https://opsa.tamu.edu |

---

## Chunking Strategy

**Chunk size:** 200 tokens (~800 characters)

**Overlap:** 50 tokens (~200 characters)

**Why these choices fit my documents:**

Most documents are short student opinions of 2–5 sentences (60–180 tokens). Chunk size 200 was chosen because it matches the average length of one RMP or Niche review, keeping one complete opinion — teaching quality, exam style, and recommendation — inside a single vector. Most reviews fit in a single chunk, so each embedded vector represents one coherent student opinion rather than a fragment.

Overlap of 50 tokens carries roughly 2–3 sentences of prior context into the next chunk. This handles the minority of reviews that exceed 200 tokens, ensuring the boundary chunk still contains enough context to be understood standalone. An overlap of 100+ tokens would create near-duplicate embeddings and inflate the index; zero overlap would cut reviews mid-thought with no recovery.

Two structural features reinforce the strategy. First, a `[REVIEW_END]` separator is inserted between reviews before splitting, so the splitter respects review boundaries rather than fusing two professors' opinions into one chunk. Second, every chunk is prefixed with `[Professor: Name | Course: Code | Dept: Department]` so even a boundary chunk always carries identifying context in the embedded text itself, not just in metadata.

**Preprocessing steps:** HTML tags stripped, HTML entities decoded, boilerplate phrases removed, reviews under 20 words discarded, professor name variants normalized (e.g. "dr barry" → "Adam Barry"), and duplicates removed by comparing leading 80 characters.

**Final chunk count:** ~100–120 chunks from 57 documents across all 10 sources (sample data). With live-scraped data the target is 600–2,000 chunks.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`

Chosen for its 256-token context window that fits the 200-token chunk size exactly, strong performance on short conversational review text, zero cost local inference (no API key, no rate limits), and 384-dimensional vectors that capture semantic similarity between student opinions even when worded differently.

**Production tradeoff reflection:**

In a real production deployment serving TAMU pre-med and public health students, the most impactful upgrades would involve weighing several factors:

- **Context length:** `all-MiniLM-L6-v2` maxes out at 256 tokens. Upgrading to `text-embedding-3-large` (OpenAI, 8,191 tokens) or `bge-large-en-v1.5` would allow embedding longer Reddit threads or Discord conversations without splitting, reducing boundary loss.
- **Accuracy on domain-specific text:** General-purpose models handle informal academic language ("curved the exam", "reads off slides") reasonably well, but a model fine-tuned on academic review text would better capture course-specific terminology like "BIMS 301", "SPH", and "MCAT prep".
- **Multilingual support:** TAMU has a large international student population. `paraphrase-multilingual-MiniLM-L12-v2` supports 50+ languages and would allow non-English queries to retrieve English-language chunks correctly — `all-MiniLM-L6-v2` is English-only.
- **Latency:** `all-MiniLM-L6-v2` runs locally under 100ms. API-based models like `text-embedding-3-large` add 200–500ms of network latency per query — acceptable but noticeable for a real-time student tool.
- **Cost:** Local models cost nothing after setup. OpenAI embeddings cost ~$0.00013 per 1K tokens — at 10,000 daily queries the cost is significant. For a student-facing free tool, local inference is the right default.

The production upgrade path would be: `all-MiniLM-L6-v2` (v1, current) → `bge-large-en-v1.5` (better accuracy, still local, free) → `text-embedding-3-large` (best accuracy, API, multilingual) depending on scale and budget.

---

## Grounded Generation

**System prompt grounding instruction:**

The system prompt instructs the LLM as follows:

You are the Unofficial Guide assistant, a RAG system that answers questions
about Texas A&M pre-med and public health professors using ONLY the student
review documents provided to you.

STRICT GROUNDING RULES:
1. Answer ONLY using information from the numbered source chunks below.
   Do NOT use your general training knowledge about TAMU, professors, or courses.
2. If the provided chunks do not contain enough information to answer the question,
   respond with: "The available student reviews don't have enough information to
   answer this question."
3. Cite every factual claim with its source number in brackets: [1], [2], etc.
4. Never invent professor names, ratings, course numbers, or student opinions
   not present in the source chunks.
5. End every answer with a Sources section listing each cited source.

GROUNDING TEST: Before writing your answer, ask yourself:
"Can I point to a specific chunk that supports every claim I'm making?"
If the answer is no for any claim — remove that claim.

**How source attribution is surfaced in the response:**

Source attribution is enforced in two ways. First, the system prompt instructs the LLM to place inline citation numbers `[1]`, `[2]` after every factual claim. Second, the prompt template formats each retrieved chunk with a numbered header — `[1] Professor: X | Source: Y | Date: Z` — so the LLM can reference them by number. The response always ends with a `Sources:` section listing each cited source by number. The `Retrieved sources` panel in the Gradio UI also displays all retrieved chunks with their similarity scores independently of the LLM answer, so users can verify grounding manually.

---

## Retrieval Test Examples

Three queries tested against the vector index, showing top returned chunks and relevance explanation.

**Query 1:** "What do students say about Prof Adam Barry teaching quality in Public Health?"

Top returned chunks:

[1] Professor: Adam Barry | Source: Uloop | similarity:0.740
   
Review: Barry for intro to public health was my first college course and it set the bar for what a good professor looks like. He explains the why behind every concept, not just the what. He is generous with his time and treats students as intelligent adults capable of engaging with complex material. The course introduced me to epidemiology, health behavior theory, health policy, and environmental health in a way that actually made me want to learn more about each. Excellent foundation course.

[2] Professor: Adam Barry | Source: Rate My Professors | similarity:0.721
    Review: Prof Barry is genuinely one of the best professors in the School of Public Health at TAMU. His lectures connect real-world health policy to actual coursework — he brings in guest speakers from the Texas DSHS and local nonprofits. Exams are application-based, not memorization. Everything comes from lecture so attendance is mandatory. His office hours are packed but worth every minute. Would take him for every HLTH course if I could.

[3] Professor: Adam Barry | Source:Professors.directory | similarity:0.720
    Review: Adam Barry is one of the highest-rated professors in the TAMU School of Public Health across multiple review platforms. His courses are consistently well-organized, his exams are fair, and he is accessible and responsive to students. He combines theoretical frameworks with applied public health examples that make the material memorable. His genuine investment in student success sets him apart from other faculty. Students in his courses reliably report that the learning translates to professional and academic settings after graduation.

**Why these chunks are relevant:** All three chunks directly discuss Adam Barry's teaching quality in the School of Public Health — the exact subject of the query. The cosine similarity scores (0.620-0.740) are little higher  because the query uses Barry's full name and "teaching quality", which appear in the chunk text and the `[Professor: Adam Barry]` prefix. The metadata pre-filter also applied `department = Public Health`, narrowing search to relevant chunks before cosine comparison.

**Query 2:** "What do TAMU pre-med students  say about which chemistry professor to take for Orgo 1 (CHEM 227)?"

Top returned chunks:

[1] Professor:Joanna Goodey | Course: CHEM 227| Source: Rate My Professors | similarity: 0.715
    "Goodey for Orgo 1 is the professor every pre-med should take at TAMU. She explains reaction mechanisms step by step and never assumes you already understand something. Exams are hard but she provides practice exams two weeks in advance that mirror the real thing almost exactly. The curve at semester end saved my GPA — class average was 67 and she curved to 76. Do not skip her office hours, she drops hints constantly. Attendance is mandatory and she notices."

[2] Professor: Jim Herman | Course: CHEM 227 | Source:Uloop  | similarity: 0.672
    "Goodey for CHEM 227 is the professor recommendation every returning student gives to every incoming pre-med. Her teaching method works: systematic coverage of mechanisms, practice exams that match the real thing, office hours that are genuinely helpful not performative. She curved the class by 9 points last semester. Her TAs are well-trained and hold regular review sessions. If you are pre-med at TAMU and you do not take Goodey for Orgo 1 you are making your life harder than it needs to be."

[3] Professor: Joanna Goodey| Course: CHEM 227 | Source: Coursicle | similarity: 0.645
    " Goodey for CHEM 227 — the professor to take for Orgo 1 at TAMU. Her practice exams are essentially the real exam with different numbers on the problems. She provides them two weeks before each test. Office hours before exams are standing room only so go early in the semester to establish a connection. She curves at the end of the semester based on class performance — last year the curve was 9 points. Take the afternoon section, the 8am section moves at a pace that is hard to keep up with before coffee."

**Why these chunks are relevant:** The query contains "CHEM 227" which the course code detector maps to `department = CHEM 227`, triggering a metadata pre-filter. All returned chunks contain `[Professor: Joanna Goodey | Course: CHEM 227]` in their prefix, ensuring exact course match. The semantic similarity is high because "recommend" and "specific professor" in the query align closely with opinionated review language in the chunks.

**Query 3:** "What do Reddit students say about CHEM 227 professor recommendations??"

Top returned chunks:
No sources retrieved.

**Why these chunks are relevant:**It is relevant as there are no recommendations for professor for CHEM 227

---

## Grounded Generation — Example Responses

**Example 1 — Grounded response with citations:**

Query: *"How do students rate Prof Adam Barry teaching quality in TAMU School of Public Health?"*

System response:

Students highly rate Prof Adam Barry's teaching quality in the TAMU School of Public Health, considering him one of the best professors in the department [1]. He is praised for making courses well-organized, exams fair, and being accessible and responsive to students [1]. His ability to combine theoretical frameworks with applied public health examples makes the material memorable, and his investment in student success sets him apart [1]. Students appreciate his teaching style, which includes bringing in guest speakers and making health behavior theory feel relevant, with exams that are challenging but fair [2, 3, 4].

Sources:
[1] Professor: Adam Barry | Dept: Public Health | Source: Professors.directory | Date: 2025-01-05
[2] Professor: Adam Barry | Dept: Public Health | Course: HLTH 320 | Source: Rate My Professors | Date: 2025-01-15
[3] Professor: Adam Barry | Dept: Public Health | Course: HLTH 320 | Source: Uloop | Date: 2024-10-15
[4] Professor: Adam Barry | Dept: Public Health | Course: HLTH 320 | Source: Rate My Professors | Date: 2024-11-03

This demonstrates grounding working correctly as it returns correct result.

**Example 2 — Out-of-scope query with refusal:**

Query: *"What is the parking situation near the TAMU SPH building?"*

System response:

No relevant student reviews found for that question.

Try including a professor name (Barry, DeLeon, Goodey, Clendenin, Herman) or a course code (HLTH 320, BIMS 301, CHEM 227, EPIB 301).

This demonstrates grounding working correctly. The system does not guess or draw on general knowledge about TAMU parking — it explicitly declines because none of the retrieved chunks contain parking information.

---

## Query Interface

**Interface description:**

The system runs as a Gradio web application at `http://127.0.0.1:7860`. The interface has the following fields:

How to launch
powershellpython app.py
Then open your browser and go to http://127.0.0.1:7860

Input fields
--------------
Question input box
-------------------
A text box at the center
Type your question about any TAMU pre-med or public health professor
Press Enter or click the Ask → button to submit
Supports multi-line input for longer questions

Ask → button
-------------
Sends your question through the full RAG pipeline
Triggers: retrieve chunks → generate grounded answer → display response

Clear chat button
-------------------
Resets the entire conversation history
Clears all panels including sources and debug

Example questions (clickable)
----------------------------------
Pre-written sample questions shown below the input box
Click any example to auto-fill the input box
Useful for quickly testing the system

Output fields
----------------
Conversation panel (chat window)

Shows the full back-and-forth conversation history

System answers appear on the left side as assistant bubbles
Every system answer includes inline citation numbers like [1] and [2]
History is preserved across multiple questions in the same session

Retrieved sources panel (right column)
----------------------------------------
Shows which documents were retrieved for the most recent question
Each source line shows:

Citation number [1]
Professor name
Platform (Rate My Professors, Reddit, Niche, etc.)
Similarity score (higher = more relevant, e.g. 0.87)
Stale warning if the review is over 2 years old

Allows you to manually verify that the answer is grounded in real sources

Debug panel (expandable accordion at the bottom)
-----------------------------------------------
Hidden by default — click to expand
Shows the raw text of every retrieved chunk
Each chunk shows: source, professor name, distance score, and first 300 characters of text
Used to verify that every claim in the answer traces back to a specific chunk
If the answer says something not present in any chunk — that is a grounding failure


Full pipeline triggered by one question
You type a question
        │
        ▼
retrieve.py — embeds your question as a 384-dim vector
        │       detects professor/course keywords
        │       searches ChromaDB for top-5 similar chunks
        ▼
generate.py — formats chunks as numbered context [1][2][3]
        │       sends to Groq llama-3.3-70b-versatile
        │       model answers ONLY from provided chunks
        ▼
app.py — displays answer in chat bubble (left side)
         displays sources in right panel
         appends to conversation history

What makes it grounded?
The system is designed so every answer is traceable to your documents:

The system prompt instructs the LLM to answer only from the numbered chunks provided — never from general training knowledge
The Retrieved sources panel shows exactly which documents were used
The Debug panel shows the raw chunk text so you can verify each claim
If you ask something not covered by your documents the system responds: "The available student reviews don't have enough information to answer this question" — it does not guess

**Sample interaction transcript:**

User:Do students recommend BIMS 301 with a specific professor at TAMU?

System:  Based on the available student review, it seems that students highly recommend taking BIMS 301 with Professor Jim Herman at TAMU, especially for those committed to medical school preparation [1]. His course is considered challenging, but valuable, and his teaching style helps students develop a genuine understanding of the material [1]. Additionally, students who complete his course tend to see an improvement in their performance on MCAT biological sciences content [1]. However, it's noted that his course is not suitable for students looking for an easy GPA boost [1].

Sources:
[1] Professor: Jim Herman | Dept: BIMS | Course: BIMS 301 | Source: Professors.directory | Date: 2025-01-28

User:    What about Goodey for Orgo 1?

System:  For Orgo 1, Professor Joanna Goodey is highly recommended, with students noting that her teaching method is effective, including systematic coverage of mechanisms and practice exams that match the real thing [2]. Her office hours are also genuinely helpful, and her TAs are well-trained and hold regular review sessions [2]. Additionally, she has a history of curving the class, with a 9-point curve last semester [2]. If you're pre-med at TAMU, taking Goodey for Orgo 1 is considered a wise decision, as it can make your life easier [2].

Sources:
[1] Professor: Joanna Goodey | Dept: Chemistry | Course: CHEM 237 | Source: Rate My Professors | Date: 2025-02-15
[2] Professor: Joanna Goodey | Dept: Chemistry | Course: CHEM 227 | Source: Uloop | Date: 2024-11-05

---
## Evaluation Report

 # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |


| 1 | What do students say about Prof Joy DeLeon's exam difficulty in the Public Health department at TAMU? | Concept-heavy exams, fast lectures, dense slides, inconsistent grading, office hours hard to access | Students say that Prof Joy DeLeon's exams in the Public Health department at TAMU are difficult because they sometimes test content she mentioned only in passing [1], and the exam questions do not match the lecture emphasis [2]. According to students, her lectures are dense and fast-paced, making it hard to take detailed notes, and the textbook is more useful for learning the material [1], [4]. Additionally, students have reported that her exam questions can be concept-heavy, requiring synthesis across multiple lectures, and it's essential to start studying at least a week before [3]. Overall, students have found her exams to be challenging and recommend forming a study group or self-teaching from the textbook to prepare [1], [4].

Sources:
[1] Professor: Joy DeLeon | Dept: Public Health | Course: HLTH 215 | Source: Coursicle | Date: 2024-08-25
[2] Professor: Joy DeLeon | Dept: Public Health | Course: HLTH 215 | Source: Rate My Professors | Date: 2025-02-10
[3] Professor: Joy DeLeon | Dept: Public Health | Course: HLTH 310 | Source: Rate My Professors | Date: 2024-08-20
[4] Professor: Joy DeLeon | Dept: Public Health | Course: HLTH 215 | Source: Uloop | Date: 2024-09-01
[5] Professor: Joy DeLeon | Dept: Public Health | Course: HLTH 215 | Source: Rate My Professors | Date: 2024-10-12| Good| Accurate |

| 2 | Do students recommend taking BIMS 301 with a specific professor at TAMU, and what reasons do they give? | Names Jim Herman, cites application-heavy exams, office hours value, MCAT prep, 8-point curve | Students highly recommend taking BIMS 301 with Professor Jim Herman at TAMU, citing that his course is "widely regarded as one of the most valuable" in the pre-med pathway [1]. They appreciate his application-focused exam style, which promotes genuine understanding over surface memorization, and note that his course consistently improves performance on MCAT biological sciences content [1]. Additionally, students praise Professor Herman's informative office hours, where he provides targeted guidance on areas to focus on [2]. However, they caution that his course is "not the right choice for students looking for an easy GPA boost," but rather for those committed to medical school preparation [1].
Sources:
[1] Professor: Jim Herman | Dept: BIMS | Course: BIMS 301 | Source: Professors.directory | Date: 2025-01-28
[2] Professor: Jim Herman | Dept: BIMS | Course: BIMS 301 | Source: Rate My Professors | Date: 2025-02-10. | Good | Accurate |

| 3 | What do TAMU pre-med students on Reddit say about which chemistry professor to take for Orgo 1 (CHEM 227)? | References Rate My Professors, Uloop,Coursicle names Goodey, cites curve policy and practice exam match | If you're a pre-med student at TAMU looking to take Orgo 1 (CHEM 227), Professor Joanna Goodey is highly recommended by students [1], [2], [3]. She explains reaction mechanisms step by step, provides practice exams that mirror the real thing, and offers helpful office hours where she drops hints [1]. Her teaching method is systematic, and she curves the class at the end of the semester, which can be a big help - last year the curve was 9 points [2]. If you're a pre-med student, taking Goodey for Orgo 1 is considered to make your life easier [2].
Sources:
[1] Professor: Joanna Goodey | Dept: Chemistry | Course: CHEM 227 | Source: Rate My Professors | Date: 2024-12-01
[2] Professor: Joanna Goodey | Dept: Chemistry | Course: CHEM 227 | Source: Uloop | Date: 2024-11-05
[3] Professor: Joanna Goodey | Dept: Chemistry | Course: CHEM 227 | Source: Coursicle | Date: 2024-09-30 | Good | Accurate |

| 4 | How do students rate Prof Adam Barry's teaching quality in TAMU School of Public Health, and would they take him again? | Rating 4.0+, would take again, specific comments on teaching and course content | Students rate Prof Adam Barry's teaching quality highly in the TAMU School of Public Health, praising his well-organized courses, fair exams, and accessibility [1]. He is considered one of the best professors in the School of Public Health, with students appreciating how he combines theoretical frameworks with applied public health examples [2]. Students would definitely take him again, with one student stating they would take him for every HLTH course if possible [2]. His ability to make complex material feel relevant and his generosity with his time have set a high standard for what a good professor looks like [4].
Sources:
[1] Professor: Adam Barry | Dept: Public Health | Source: Professors.directory | Date: 2025-01-05
[2] Professor: Adam Barry | Dept: Public Health | Course: HLTH 320 | Source: Rate My Professors | Date: 2025-01-15
[3] Professor: Adam Barry | Dept: Public Health | Course: HLTH 320 | Source: Uloop | Date: 2024-10-15
[4] Professor: Adam Barry | Dept: Public Health | Course: HLTH 215 | Source: Uloop | Date: 2024-08-22
[5] Professor: Adam Barry | Dept: Public Health | Course: HLTH 320 | Source: Rate My Professors | Date: 2024-11-03 | Good | Accurate |

| 5 | What do Reddit students say about CHEM 227 professor recommendations? |No relevant student reviews found for that question as no mention for professor recommendation for CHEM227 in reddit reviews  | No relevant student reviews found for that question.

Try including a professor name (Barry, DeLeon, Goodey, Clendenin, Herman) or a course code (HLTH 320, BIMS 301, CHEM 227, EPIB 301).| Good | Accurate |

---

## Failure Case Analysis

**Question that failed:**

Do students recommend taking BIMS 301 with a specific professor at TAMU?

**What the system returned:**

## Failure Case Analysis

*Question that failed:**

Do students recommend taking BIMS 301 with a specific professor at TAMU?

**What the system returned:**
"No relevant student reviews found for that question.
Try a professor name (Barry, DeLeon, Goodey, Clendenin, Herman)
or a course code (HLTH 320, BIMS 301, CHEM 227, EPIB 301)."

**Root cause (tied to a specific pipeline stage):**
The failure occurred across two pipeline stages — clean.py (Stage 2)
and retrieve.py (Stage 5).

In clean.py, each document is tagged with the first matching department
keyword found in its text. Several Reddit and Discord chunks that discuss
BIMS 301 also mention "public health" or "pre-med" earlier in the same
message. For example a Reddit comment that begins "finishing my junior
year as pre-med at TAMU" and later says "Herman for BIMS 301" gets
tagged department = Public Health instead of BIMS because "public health"
appears first in the keyword scan.

In retrieve.py, the query "BIMS 301" triggered a strict metadata
pre-filter of department == "BIMS" using the $eq operator. This filter
correctly matched chunks tagged as BIMS but silently excluded all the
Reddit and Discord chunks that were mis-tagged as Public Health during
cleaning. With those chunks blocked, not enough results passed the
similarity threshold and retrieve_chunks() returned an empty list [].
The pipeline then triggered the "No relevant student reviews found"
fallback message.

**What was changed to fix it:**
In retrieve.py, the department $eq filter was replaced with a professor
name filter — professor_name is stored consistently from the metadata
headers in the .txt files and does not depend on keyword inference.
A guaranteed unfiltered fallback was also added: if the professor filter
returns zero results, the query retries with no metadata filter and relies
purely on cosine similarity. This ensures a valid question about a
professor who exists in the corpus never returns empty results due to a
metadata tagging mismatch in the cleaning stage.

---
## Spec Reflection

**One way the spec helped during implementation:**

The planning.md chunking strategy section — specifying 200 tokens, 50 overlap, LangChain `RecursiveCharacterTextSplitter`, the `[REVIEW_END]` separator, and the professor prefix requirement — made the Claude prompt for generating `chunk.py` extremely precise. Instead of getting a generic text splitter, the generated code included the separator logic, the prefix builder, and the boundary chunk re-prefixing. The spec's requirement to "prepend professor name and course code to every chunk" directly produced the `build_prefix()` function that now ensures every chunk carries identifying context in its embedded text. Without this level of spec detail the generated code would have needed multiple iterations.

**One way implementation diverged from the spec, and why:**

The spec specified a minimum similarity threshold of 0.70, but during testing this threshold was too strict — it filtered out valid results that scored between 0.50 and 0.69, causing the "No relevant results found" error for valid queries like "Do students recommend BIMS 301 with a specific professor?" The threshold was lowered to 0.50 because the corpus is small (sample data) and the `all-MiniLM-L6-v2` model produces lower absolute similarity scores on short informal text than academic benchmarks suggest. In a production system with a full corpus the 0.70 threshold would be more appropriate.

---
## AI Usage

**Instance 1**

- *What I gave the AI:* The full Document Ingestion section of planning.md describing all 10 sources, the required metadata fields (professor_name, department, course, source, rating, review_date), the cleaning rules (strip HTML, normalize professor names, filter under 20 words), and the chunking strategy section specifying fixed-size 200 tokens, 50 overlap, LangChain RecursiveCharacterTextSplitter, [REVIEW_END] separator, and professor prefix requirement.
- *What it produced:* `ingest.py` with one function per source, `clean.py` with normalize and deduplicate functions, and `chunk.py` with the `build_prefix()` and `chunk_document()` functions.
- *What I changed or overrode:* The generated `ingest.py` used live web scraping as the primary approach, which failed for most sources due to JavaScript rendering and bot blocking. I directed Claude to rebuild it using plain `.txt` files in a `documents/` folder as the primary data source, with live scraping as an optional fallback. I also directed it to add the `parse_txt_file()` function with `[REVIEW_END]` block parsing, which was not in the original generated code.

**Instance 2**

- *What I gave the AI:* The Retrieval Approach section of planning.md specifying `all-MiniLM-L6-v2`, ChromaDB with collection name `rateprofessors`, top-K=5, cosine similarity, minimum threshold 0.70, and the metadata filtering requirement for department keywords. I also gave it the 5 evaluation questions to use as test queries.
- *What it produced:* `embed.py` with `embed_and_store()` and `verify_index()`, and `retrieve.py` with `retrieve_chunks()` including department keyword detection and metadata pre-filter.
- *What I changed or overrode:* The generated `retrieve.py` only detected broad department keywords like "public health" or "chemistry" but missed course codes like "BIMS 301" or "CHEM 227". I directed Claude to add a `COURSE_CODE_DEPT_MAP` that maps specific course codes to departments. I also overrode the similarity threshold from 0.70 to 0.50 after testing showed valid queries were being filtered out, and I added a fallback that retries without the metadata filter when the filtered query returns zero results.

---
## Pipeline

Student question
      │
      ▼
Stage 1 — ingest.py
  10 sources from documents/ → data/raw/*.jsonl
  Tool: plain .txt parser + optional PRAW for Reddit
      │
      ▼
Stage 2 — clean.py
  Strip HTML · normalize names · filter <20 word reviews · deduplicate
      │
      ▼
Stage 3 — chunk.py
  200 tokens · 50 overlap · [Professor:] prefix · [REVIEW_END] separator
  LangChain RecursiveCharacterTextSplitter
      │
      ▼
Stage 4 — embed.py
  all-MiniLM-L6-v2 → 384-dim vectors → ChromaDB (local, no API key)
      │
      ▼
Stage 5 — retrieve.py
  Embed query → course code + dept keyword filter → top-5 cosine similarity ≥ 0.50
      │
      ▼
Stage 6 — generate.py
  Groq llama-3.3-70b-versatile → grounded answer → cited sources
      │
      ▼
app.py → Gradio chat UI at http://127.0.0.1:7860

---

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key to .env
# Edit .env and set: GROQ_API_KEY=your_key_here
# Get a free key at: https://console.groq.com

# 4. Build the index (run once)
python main.py --build

# 5. Launch web UI
python app.py
# Open http://127.0.0.1:7860
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Vector index not found` | Run `python main.py --build` |
| `GROQ_API_KEY not set` | Add key to `.env` file |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `langchain.text_splitter` error | `pip install langchain-text-splitters` |
| Site can't be reached at 0.0.0.0 | Open `http://127.0.0.1:7860` instead |
| No results returned | Run `python main.py --build --overwrite` to rebuild index |
