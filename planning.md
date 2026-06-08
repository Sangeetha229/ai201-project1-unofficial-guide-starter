# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

Pre-med and Public Health are high-stakes academic paths — students pursuing medicine, dentistry, or public health careers need to make smart course decisions early. Choosing the wrong professor for Organic Chemistry, Biochemistry, or BIMS courses can tank a GPA and hurt medical school applications. Student reviews tell you things that genuinely matter: which professor's exams are concept-heavy vs. memorization-based, who gives useful feedback on assignments, whose office hours are actually worth attending, and which Public Health electives are "easy A's" vs. unexpectedly brutal.

The TAMU course catalog tells you what a class covers, not whether the professor reads directly off slides for 75 minutes. The School of Public Health faculty directory lists credentials and research interests — not whether Dr. X grades harshly or curves generously. Advisors give politically safe recommendations. Department websites have zero opinion. None of these official sources will tell you that a particular BIMS professor's exams have a 40% class average, or that a Public Health instructor is known for being unresponsive during finals week.
Student-generated knowledge fills that gap. It's earned through lived experience, passed peer-to-peer, and often more actionable than anything published officially — which is exactly what makes it the right kind of content for an Unofficial Guide RAG system.

---
## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Rate My Professors — TAMU College Station|The primary source for individual professor ratings at Texas A&M. Filter by department (Biology, Chemistry, Public Health, BIMS). Includes quality scores, difficulty ratings, and written student reviews |https://www.ratemyprofessors.com/search/professors/1003?q=*|
| 2 |r/aggies (Reddit) |The main TAMU subreddit where students post threads asking for professor recommendations, share course experiences, and give honest takes on pre-med prerequisites like Orgo, Biochem, and BIMS courses. Search "professor" or course name. |https://www.reddit.com/r/aggies |
| 3 |Niche — TAMU Academics & Professor Reviews |Aggregates thousands of verified student reviews specifically about TAMU academics, professors, and course workload. Students comment on pre-med and STEM course rigor, professor quality, and overall teaching support. Includes an academics grade and searchable review filters. |https://www.niche.com/colleges/texas-a-and-m-university/academics/|
| 4 |Coursicle — TAMU Professors | Lists professors at Texas A&M with student reviews tied directly to specific courses and sections, including what they're teaching each semester. Useful for pre-med and public health students planning their schedule since reviews are linked to real course offerings.|https://www.coursicle.com/tamu/professors/|
| 5 |TAMU Discord Servers |Unofficial student Discord servers (search "TAMU" or "Aggies" on Disboard.org) contain channels where students discuss professors for BIMS, BIOL, CHEM, and Public Health courses in real time. |https://disboard.org/servers/tag/tamu|
| 6 |Uloop TAMU Professor Ratings | A student-ranking site that collects professor ratings specifically from Texas A&M students. Good secondary source if RMP has sparse reviews.  |https://tamu.uloop.com/professors |
| 7 |Professors.directory — TAMU | Provides professor ratings and reviews for Texas A&M University, College Station. Useful for cross-checking professors with few RMP reviews.| https://www.professors.directory/school/tx-texas_a_m_university_college_station/g/|
| 8 |TAMU School of Public Health Course Catalog |Official faculty directory listing all Public Health professors by department (Health Behavior, Epidemiology & Biostatistics, Environmental & Occupational Health, Health Policy & Management). Use this to get exact professor names, then look them up on RMP |https://catalog.tamu.edu/undergraduate/public-health/|
| 9 |TAMU Pre-Medical Society |Founded in 1975, this is the official pre-med student org that works closely with the Office of Professional School Advising. Their Discord or GroupMe often contains candid professor recommendations passed between members.|https://getinvolved.tamu.edu/org/tamupremed|
| 10 |TAMU Professional School Advising (OPSA) |Official advising office for pre-med/pre-dental students. Their listservs and portal carry course advice from advisors who know which professors are most supportive of pre-health applicants. |https://opsa.tamu.edu |

---

## Chunking Strategy

 <!--How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

Chunk size: 200 - this matches the average length of one RMP/Niche review (2–5 sentences ≈ 60–180 tokens) closely enough that most reviews land in a single chunk naturally.

Overlap: 50 A typical RMP or Niche review sentence runs 15–25 tokens. So 50 tokens carries roughly 2–3 sentences of context into the next chunk — enough to preserve meaning at a split boundary without excessive redundancy.

**Reasoning:**

Chunk size 200 was chosen because it matches the average length of one student review (60–180 tokens), keeping one complete opinion — teaching quality, exam style, and recommendation — inside a single vector. Overlap of 50 tokens was chosen because it carries 2–3 sentences of prior context into the next chunk, ensuring that reviews slightly over 200 tokens remain coherent when split, without inflating the index with near-duplicate embeddings that degrade retrieval precision. Together, 200/50 respects the natural unit of knowledge in a review-heavy corpus while staying simple enough to implement reliably as a fixed-size strategy. 

---
## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**all-MiniLM-L6-v2  via sentence-transformers — chosen for its 256-token context window that fits the 200-token chunk size, strong performance on short conversational review text, zero cost local inference, and 384-dimensional vectors that capture semantic similarity between student opinions even when worded differently.

**Top-k:** 5 - retrieves 5 chunks per query to capture diverse student perspectives across sources (RMP, Reddit, Niche, Coursicle, Discord) covering teaching style, exam difficulty, grading, and office hours, while keeping retrieved context to ~1,000 tokens — well within Claude's context window and leaving headroom for the system prompt and generated answer.

**Production tradeoff reflection:**
In a real production deployment serving TAMU pre-med and public health students, the most impactful upgrades would be: replacing fixed-size chunking with review-aware chunking to prevent professor review merging, upgrading the embedding model from all-MiniLM-L6-v2 to bge-large-en-v1.5 for better accuracy on informal student language, implementing dynamic K with a similarity score threshold to prevent noise chunks from degrading answer quality, migrating from ChromaDB to pgvector for scalable metadata-filtered retrieval, routing simple queries to Claude Haiku and complex comparisons to Claude Sonnet to balance cost and quality, and adding weekly re-indexing with recency decay weighting to ensure reviews about retired or changed professors do not mislead students making real GPA-affecting course decisions. Every tradeoff ultimately reduces to the same question: how much engineering complexity is justified to improve the accuracy of information that students will act on.

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. -->

| # | Question | Expected answer |
|---|----------|-----------------|

| 1 |What do students say about Prof Joy DeLeon's exam difficulty in the Public Health department at TAMU? |Reviews should mention that her exams are concept-heavy, lectures move fast, slides are dense, and that her difficulty rating on RMP is 2.8. Answer should cite at least 2 student reviews from RMP or Coursiclee and mention her department  specifically. A vague answer like "she is hard" with no supporting student quotes or source citations is a failure. |

| 2 |Do students recommend taking BIMS 301 with a specific professor at TAMU, and what reasons do they give? |Answer should name at least one specific professor who teaches BIMS 301, include student reasoning (teaching clarity, exam style, grading policy, or office hours usefulness), and cite the source (RMP or  Professors.directory). An answer that says "BIMS 301 is a hard course" without naming a professor or citing student opinions is a failure. |

| 3 |What do TAMU pre-med students on Reddit say about which chemistry professor to take for Orgo 1 (CHEM 227)? |Answer should reference (RMP,Uloop,Couricle) as the source, name at least one professor recommended by students for CHEM 227, and include specific student reasoning such as exam structure, curve policy, or teaching style. An answer pulled only from RMP without mentioning Uloop is a partial failure — the question specifically asks about Reddit. |

| 4 |How do students rate Prof Adam Barry's teaching quality in the TAMU School of Public Health, and would they take him again? |Answer should include his quality rating (expected high, 4.0+), his "would take again" percentage, at least 2 specific student comments about his teaching style or course content, and name his department. Answer must be grounded in cited review chunks — a generic positive answer without specifics or citations is a failure. |

| 5 |What do Reddit students say about CHEM 227 professor recommendations?|It should return that it couldn't find relevant reviews otherwise it is a failure|

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1.Two Different Professors' Reviews Merged Into One Chunk:
Fixed-size chunking does not recognize where one review ends and the next begins
A 200-token cut can land mid-boundary — merging Prof Barry's positive review with Prof DeLeon's negative review into one chunk
Retrieval returns a blended chunk — Claude wrongly attributes one professor's exam style to another
Pre-med students make course decisions based on incorrectly attributed reviews
Fix: Insert [REVIEW_END] separator between reviews during scraping + store professor name as mandatory metadata on every chunk

2.Outdated Reviews Returned as Current Fact:
RMP and Reddit reviews  span over 10+ years — all indexed with equal retrieval weight
A 2019 review about a professor who retired in 2022 is retrieved with the same confidence as a 2025 review
Claude generates a confident answer citing "multiple student reviews" that are all years out of date
Student enrolls expecting one experience and finds a completely different professor or course structure
Fix: Store review_date metadata on every chunk + apply recency decay weighting + instruct Claude to flag reviews older than 2 years

3.Off-Topic Chunks Retrieved Due to Shared Vocabulary
Words like "curved," "office hours," "hard exams," and "recommend" appear in reviews across every TAMU department
An engineering or chemistry professor review embeds nearly identically to a public health review
Top-5 retrieved chunks include professors from wrong departments — Claude names a Chemistry professor as an epidemiology recommendation
Student registers for the completely wrong course based on a confident-sounding but off-topic answer
Fix: Tag every chunk with department metadata + filter vector search by department before running cosine similarity

4.Key Information Split Across Chunk Boundaries:
Reviews follow a setup-first, key-opinion-second structure — chunking splits this mid-thought
Second chunk starts without professor name, course number, or recommendation context
50-token overlap carries back partial context but not enough to identify the professor or course
Claude reads "Her exams are straightforward if you go to lecture" with no clear attribution — answer becomes vague or wrongly attributed
Fix: Prepend professor name and course code to every chunk text during indexing so boundary chunks always carry identifying context

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->


Student question---->Stage 1 — ingest.py(Document Ingestion)
  10 sources from documents/ → data/raw/*.jsonl
 ( Tool: plain .txt parser + optional PRAW for Reddit)---->Stage 2 — clean.py
  (Strip HTML · normalize names · filter <20 word reviews · deduplicate)---->Stage 3 — chunk.py(Chunking)  (200 tokens · 50 overlap · [Professor:] prefix · [REVIEW_END] separator
  LangChain RecursiveCharacterTextSplitter)----->Stage 4 — embed.py(Embedding + Vector Store)
  all-MiniLM-L6-v2 → 384-dim vectors → ChromaDB (local, no API key)---->Stage 5 — retrieve.py(Retrieval) (Embed query → course code + dept keyword filter → top-5 cosine similarity ≥ 0.50)---->Stage 6 — generate.py(Generation)
  (Groq llama-3.3-70b-versatile → grounded answer → cited sources---->app.py → Gradio chat UI at http://127.0.0.1:7860)
  

## AI Tool Plan

**Milestone 3 — Ingestion and chunking:**

AI tool: Claude (claude.ai chat)

What I gave it as input:
- The Document Ingestion section of this planning.md describing all 10 sources
  (RMP, Reddit, Niche, Coursicle, Discord, Uloop, Professors.directory,
  TAMU SPH Catalog, Pre-Med Society, OPSA)
- The metadata fields required on every document: professor_name, department,
  course, source, rating, review_date
- The cleaning rules: strip HTML, remove reviews under 20 words, normalize
  professor names, skip deleted Reddit comments
- The Chunking Strategy section specifying: fixed-size, 200 tokens, 50 overlap,
  LangChain RecursiveCharacterTextSplitter
- The requirement to prepend [Professor: {name} | Course: {course}] to every chunk
- The requirement to insert [REVIEW_END] separator between reviews before splitting

What I expected it to produce:
- ingest.py with one loading function per source using plain .txt files
  in a documents/ folder as the primary data source
- clean.py with a clean_text() function that strips HTML, normalizes
  whitespace, normalizes professor name variants, and filters short reviews
- chunk.py with a chunk_documents() function using
  RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
- Output files saved to data/raw/*.jsonl (one per source) and
  data/chunks/chunks.jsonl

How I verified it:
- Ran ingest.py and confirmed all 10 .jsonl files appeared in data/raw/
- Spot-checked 5 records per source to confirm metadata fields were populated
  and professor names were normalized
- Loaded chunks.jsonl and confirmed professor name prefix appeared on every chunk
- Ran chunk.py and inspected 5 sample chunks to confirm each was self-contained
  and readable as a standalone unit

What I changed or overrode:
- The generated ingest.py used live web scraping as the primary approach which
  failed for most sources due to JavaScript rendering and bot blocking. I
  directed Claude to rebuild it using plain .txt files in a documents/ folder
  as the primary data source with live scraping as an optional fallback
- I directed Claude to add the parse_txt_file() function with [REVIEW_END]
  block parsing which was not in the original generated code


**Milestone 4 — Embedding and retrieval:**

AI tool: Claude (claude.ai chat)

What I gave it as input:
- The Retrieval Approach section of this planning.md specifying
  all-MiniLM-L6-v2 via sentence-transformers, 384-dimensional vectors
- The Vector Store section specifying ChromaDB (local), collection
  name tamu_reviews
- The metadata fields to store alongside every vector: professor_name,
  department, source, review_date, course
- The Retrieval section specifying top-K=5, cosine similarity, minimum
  threshold of 0.70
- The metadata filtering requirement: detect department keywords in query
  and pre-filter before cosine search
- All 5 evaluation plan questions to use as test queries after implementation

What I expected it to produce:
- embed.py with an embed_and_store() function that loads chunks.jsonl,
  encodes each chunk using SentenceTransformer('all-MiniLM-L6-v2'),
  and stores vectors plus metadata in ChromaDB
- embed.py with a verify_index() function that prints total vector count
  and runs one test query
- retrieve.py with a retrieve_chunks(query, k=5) function that embeds
  the query, applies department metadata pre-filter if keyword detected,
  queries ChromaDB, and drops results below similarity 0.70
- Output: top-5 chunk dictionaries with text, metadata, and
  similarity_score fields

**Milestone 5 — Generation and interface:**

AI tool: Claude (claude.ai chat)

What I gave it as input:
- The Generation section of this planning.md specifying
  llama-3.3-70b-versatile via Groq, max_tokens=1000
- The system prompt requirements: answer only from retrieved chunks,
  cite sources by number, flag reviews older than 2 years, never
  fabricate professor names
- The output format requirement: prose answer with numbered citations
  showing professor name and source
- The 5 evaluation plan questions with their expected answers and
  3-point scoring rubric
- The retrieve_chunks() function from Milestone 4 as context

What I expected it to produce:
- generate.py with a generate_answer(query, chunks) function that builds
  the system prompt, formats top-5 chunks as numbered context blocks,
  and calls the Groq API
- app.py with a Gradio web interface showing both user and system
  messages in a chat window with a Retrieved sources panel
- main.py that wires all stages together into one callable pipeline


