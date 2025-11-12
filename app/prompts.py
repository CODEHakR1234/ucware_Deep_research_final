# app/prompts.py
"""prompts.py
LLM 파이프라인에서 사용하는 주요 프롬프트 템플릿 정의 모듈.

Jinja2 템플릿을 사용하며, 웹 여부 판단 / 정보 평가 / 응답 생성 / 검증 /
리파인 / 번역 등 LangGraph 노드별 작업에 대응하는 프롬프트를 제공한다.
"""

from jinja2 import Template

# ─────────────────────────────────────────────────────────────
# 1-a. 웹 정보 필요 여부 판단 (RAG_router)
# ─────────────────────────────────────────────────────────────
PROMPT_DETERMINE_WEB = Template("""
You are an intelligent assistant tasked with determining whether the given query requires additional, up-to-date, or broader information from the web, beyond what has been retrieved from a local database (vectorDB).

Consider the following:
- If the summary from vectorDB sufficiently and specifically answers the query with relevant and reasonably current information, respond with `false`.
- If the query concerns the structural components of the document (e.g., headings, conclusions, format), and the summary appears to contain that structure, respond with `false`.
- If the summary is missing key information, is outdated, overly generic, or irrelevant to the query, respond with `true`.
- If the query involves recent events, real-time data, current prices, news, or trending topics, respond with `true`.

You may only respond with a single word: either `true` or `false`.

Query: {{ query }}
Retrieved Summary: {{ summary }}
""")

# ─────────────────────────────────────────────────────────────
# 1-b. 구조적 RAG(ColPali) 필요 여부 판단 (RAG_router)
# ─────────────────────────────────────────────────────────────
PROMPT_DETERMINE_STRUCT_RAG = Template("""
You are a routing assistant that decides whether the question requires STRUCTURAL, image-grounded retrieval (figures, tables, diagrams, page layout) rather than plain text retrieval.

Return ONLY one word:
- 'colpali' if the user asks about figures, tables, images, diagrams, screenshots, equations as visual objects, page numbers/layout, or content that is better answered from page images.
- 'text' otherwise.

Query: {{ query }}
Vector Summary (for reference): {{ summary }}
""")


# ─────────────────────────────────────────────────────────────
# 2. 검색 조각(chunks) 유효성 점수 (grade)
# ─────────────────────────────────────────────────────────────
PROMPT_GRADE = Template("""
You are a relevance grader evaluating whether a retrieved document chunk is topically and semantically related to a user question.

Instructions:
- Your job is to determine if the retrieved chunk is genuinely helpful in answering the query, based on topic, semantics, and context.
- Surface-level keyword overlap is not enough — the chunk must provide meaningful or contextually appropriate information related to the query.
- However, minor differences in phrasing or partial answers are acceptable as long as the document is on-topic.
- If the chunk is off-topic, unrelated, or misleading, return 'no'.
- If it is relevant and contextually appropriate, return 'yes'.

You MUST return only one word: 'yes' or 'no'. Do not include any explanation.

Query: {{ query }}
Retrieved Chunk: {{ chunk }}
Vector Summary (Optional): {{ summary }}
""")


# ─────────────────────────────────────────────────────────────
# 3. 최종 답변 생성 (generate)
# ─────────────────────────────────────────────────────────────
PROMPT_GENERATE = Template("""
You are a helpful assistant that can generate a answer of the query in English.
Use the retrieved information to generate the answer.
YOU MUST RETURN ONLY THE ANSWER, NOTHING ELSE.
Query: {{ query }}
Retrieved: {{ retrieved }}
""")

# ─────────────────────────────────────────────────────────────
# 4. 답변 품질 검증 (verify)
# ─────────────────────────────────────────────────────────────
PROMPT_VERIFY = Template("""
You are a helpful assistant that can verify the quality of the generated answer.
Please evaluate the answer based on the following five criteria:

1. Does the answer directly address the query?
2. Is the answer based on the retrieved information?
3. Is the answer logically consistent?
4. Is the answer complete and specific?
5. Does the answer avoid hallucinations or unsupported claims?

Notes:
- Even if the query is short, polite, or conversational in nature (e.g., greetings, thanks, confirmations), the answer must still be grounded in the retrieved information to be considered good.
- If the answer does not reference or rely on the retrieved content in a meaningful way, mark it as bad.
- Do not infer user intent beyond the given query and content.

Query: {{ query }}
Summary: {{ summary }}
Retrieved Information: {{ retrieved }}
Generated Answer: {{ answer }}

Return only one word: good or bad.
""")

# ─────────────────────────────────────────────────────────────
# 5. 쿼리 리파인 또는 사과문 (refine)
# ─────────────────────────────────────────────────────────────
PROMPT_REFINE = Template("""
You are a helpful assistant that can do two things:
1. If the query is not related to the document summary, return ONLY this sentence: "I'm sorry, I can't find the answer to your question even though I read all the documents. Please ask a question about the document's content."
2. If the query is related, refine the query to get more relevant and accurate information based on the document summary and retrieved information. Return ONLY the refined query, nothing else.

Document Summary: {{ summary }}
Original Query: {{ query }}
Retrieved Information: {{ retrieved }}
Generated Answer: {{ answer }}
""")

# ─────────────────────────────────────────────────────────────
# 6. 번역 (translate)
# ─────────────────────────────────────────────────────────────
PROMPT_TRANSLATE = Template("""
You are a helpful assistant that can translate the answer to User language.
EN is English, KR is Korean.
ONLY RETURN THE TRANSLATED SEQUENCE, NOTHING ELSE.
User language: {{ lang }}
Answer: {{ text }}
""")

# ─────────────────────────────────────────────────────────────
# 7. Tutorial 번역 (tutorial_translate)
# ─────────────────────────────────────────────────────────────
PROMPT_TUTORIAL_TRANSLATE = Template("""
You are a professional translator specializing in educational content and technical documentation.
Your task is to translate the following tutorial guide to the specified language while preserving ALL content and structure.

CRITICAL REQUIREMENTS:
1. **Preserve ALL content**: Do not summarize, condense, or omit any information
2. **Maintain exact structure**: Keep all sections, subsections, and their order
3. **Preserve ALL formatting**: Headers (# ## ###), lists (- *), bold (**), italic (*), code blocks, etc.
4. **STRICT IMAGE REFERENCE RULE**: 
   - ONLY keep image references that EXACTLY match the original text
   - DO NOT create, add, or modify any image references
   - Copy EXACT image IDs from original (e.g., [IMG_3_1], [IMG_4_1], [IMG_13_1])
   - DO NOT change image IDs (e.g., don't change [IMG_3_1] to [IMG_1_1])
   - DO NOT create new image references that don't exist in original
   - If original has no images, output should have no images
5. **Preserve ALL links**: Keep all URLs and references unchanged
6. **Maintain educational tone**: Keep the tutor's notes and explanations intact
7. **Preserve technical accuracy**: Maintain all technical terms and concepts
8. **Keep ALL bullet points**: Do not reduce or combine bullet points
9. **Preserve ALL examples**: Keep all examples and their explanations
10. **Maintain ALL key takeaways**: Keep the complete "Key takeaways" section

ABSOLUTE FORBIDDEN ACTIONS:
- ❌ DO NOT create new image references like [IMG_X_Y]
- ❌ DO NOT add image references where none existed
- ❌ DO NOT change existing image IDs
- ❌ DO NOT mention images that don't exist in the original

Target language: {{ lang }}
Tutorial content to translate:
{{ text }}

IMPORTANT: 
- Translate word-for-word while maintaining the exact same structure and completeness
- Do not add, remove, or modify any content except for the language translation itself
- Image references must be EXACTLY as they appear in the original text
- If you are unsure about an image reference, DO NOT include it
""")

# ─────────────────────────────────────────
# 섹션별 번역 프롬프트 (이미지 체크 포함)
# ─────────────────────────────────────────
PROMPT_TUTORIAL_TRANSLATE_WITH_IMAGES = Template("""
📋 This section contains {{ image_count }} image reference(s): {{ available_image_ids }}

🚨 CRITICAL TRANSLATION REQUIREMENTS - READ CAREFULLY:

1. **ABSOLUTE COMPLETENESS - NO EXCEPTIONS**:
   ❌ DO NOT summarize or condense
   ❌ DO NOT skip or omit ANY sentence
   ❌ DO NOT shorten explanations
   ✅ TRANSLATE EVERY SINGLE WORD
   ✅ Your translation should be SIMILAR IN LENGTH to the original

2. **STRICT IMAGE REFERENCE RULE**:
   - Preserve EXACTLY {{ image_count }} image reference(s)
   - Each [IMG_X_Y] must appear the SAME number of times as in the original
   - DO NOT add, remove, or modify any image references

3. **EXACT STRUCTURE PRESERVATION**:
   - Keep ALL sections, subsections, and their exact order
   - Maintain ALL formatting: Headers (# ## ###), lists (- *), bold (**), italic (*), code blocks
   - Preserve ALL examples, explanations, and details

---

You are a professional translator specializing in educational content and technical documentation.
Your task is to translate the following tutorial guide to the specified language while preserving EVERY SINGLE DETAIL.

Target language: {{ lang }}
Tutorial content to translate:
{{ text }}

⚠️ FINAL CHECK BEFORE SUBMITTING:
- Is your translation similar in length to the original?
- Did you translate EVERY sentence without skipping?
- Do you have exactly {{ image_count }} image reference(s)?
- Is the structure IDENTICAL to the original?
""")

PROMPT_TUTORIAL_TRANSLATE_NO_IMAGES = Template("""
⚠️ This section has NO images. Your translation should also have NO image references.

---

🚨 CRITICAL TRANSLATION REQUIREMENTS - READ CAREFULLY:

1. **ABSOLUTE COMPLETENESS - NO EXCEPTIONS**:
   ❌ DO NOT summarize or condense
   ❌ DO NOT skip or omit ANY sentence
   ❌ DO NOT shorten explanations
   ✅ TRANSLATE EVERY SINGLE WORD
   ✅ Your translation should be SIMILAR IN LENGTH to the original

2. **NO IMAGES**: 
   - This section has no images
   - DO NOT add any image references

3. **EXACT STRUCTURE PRESERVATION**:
   - Keep ALL sections, subsections, and their exact order
   - Maintain ALL formatting: Headers (# ## ###), lists (- *), bold (**), italic (*), code blocks
   - Preserve ALL examples, explanations, and details

---

You are a professional translator specializing in educational content and technical documentation.
Your task is to translate the following tutorial guide to the specified language while preserving EVERY SINGLE DETAIL.

Target language: {{ lang }}
Tutorial content to translate:
{{ text }}

⚠️ FINAL CHECK BEFORE SUBMITTING:
- Is your translation similar in length to the original?
- Did you translate EVERY sentence without skipping?
- Are there NO image references in your translation?
- Is the structure IDENTICAL to the original?
""")

# ─────────────────────────────────────────
# 새 멀티모달 자습서용 프롬프트
# ─────────────────────────────────────────
PROMPT_TUTORIAL = Template("""
You are an expert tutor.
Using the *semantic chunks* below, write a self-study guide
for a learner.

Rules
-----
• Output in Markdown (H1~H3 headings).
• **STRICT IMAGE REFERENCE RULE**: 
  - The chunks below contain image placeholders like [IMG_3_1], [IMG_4_1], [IMG_13_1], etc.
  - ONLY use image IDs that EXACTLY appear in the chunks below
  - Copy the EXACT image ID from chunks when referencing images
  - **Each image should be referenced ONLY ONCE** - do not repeat the same image multiple times
  - Place the image reference where it is MOST relevant in your explanation
  - DO NOT create or imagine new image IDs
  - If no [IMG_X_Y] pattern exists in chunks, do not add any images
• After the FIRST reference to each Figure/Table, add "**Tutor's note:** …" line explaining it.
• If you need to mention the same concept again, just reference it verbally (e.g., "as shown in the figure above")
• Keep each section ≤ 200 words if possible.
• End with "Key takeaways" bulleted list.

ABSOLUTE FORBIDDEN ACTIONS:
- ❌ DO NOT create new image IDs that don't exist in chunks
- ❌ DO NOT modify existing image IDs (e.g., changing [IMG_3_1] to [IMG_1_1])
- ❌ DO NOT add image references where none existed in chunks
- ❌ DO NOT mention images without using their exact [IMG_X_Y] format from chunks

Chunks:
{{ chunks }}
""")

# ─────────────────────────────────────────
# 섹션별 이미지 제한 프롬프트
# ─────────────────────────────────────────
PROMPT_TUTORIAL_SECTION_WITH_IMAGES = Template("""
📋 Available image IDs for this section ONLY: {{ available_image_ids }}

🚨 STRICT IMAGE POLICY:
1. Use ONLY the {{ image_count }} image(s) listed above
2. Each image should appear EXACTLY ONCE in your explanation
3. The chunks below already contain placeholders [IMG_X_Y] showing where images appear in the original document
4. You can either:
   - Keep the placeholder where it is (if it fits your explanation)
   - Move it to a better location (if it makes more sense)
   - BUT: Use each image ID only ONCE

✅ GOOD example:
   "The transformer architecture [IMG_3_1] uses attention..."
   (image used once, naturally placed)

❌ BAD example:
   "The architecture [IMG_3_1] processes... Later, [IMG_3_1] shows..."
   (same image used twice - FORBIDDEN!)

---

You are an expert tutor.
Using the *semantic chunks* below, write a self-study guide for a learner.

Rules
-----
• Output in Markdown (H1~H3 headings).
• Use ONLY the {{ image_count }} image(s) listed at the top: {{ available_image_ids }}
• Each image should be referenced ONLY ONCE in your explanation
• After the FIRST reference to each Figure/Table, add "**Tutor's note:** …" line explaining it.
• If you need to mention the same concept again, just reference it verbally (e.g., "as shown in the figure above")
• Keep each section ≤ 200 words if possible.
• End with "Key takeaways" bulleted list.

Chunks:
{{ chunks }}
""")

PROMPT_TUTORIAL_SECTION_NO_IMAGES = Template("""
⚠️ This section has NO images available. 
DO NOT reference ANY images. Write text-only explanation.

---

You are an expert tutor.
Using the *semantic chunks* below, write a self-study guide for a learner.

Rules
-----
• Output in Markdown (H1~H3 headings).
• NO images available - write text-only explanation
• Keep each section ≤ 200 words if possible.
• End with "Key takeaways" bulleted list.

Chunks:
{{ chunks }}
""")
