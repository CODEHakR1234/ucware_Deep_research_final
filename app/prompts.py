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
# 7. Tutorial 번역 (tutorial_translate) - 간소화
# ─────────────────────────────────────────────────────────────
PROMPT_TUTORIAL_TRANSLATE = Template("""
You are a professional translator.

Task: Translate this tutorial to {{ lang }}.

RULES:
1. Translate EVERYTHING - don't skip or summarize
2. Keep [IMG_X_Y] tokens EXACTLY as they are
3. Preserve all structure (headings, lists, formatting)
4. Maintain the same length and detail

Target language: {{ lang }}

Content to translate:
{{ text }}
""")

# ─────────────────────────────────────────
# 섹션별 번역 프롬프트 (이미지 체크 포함) - 간소화
# ─────────────────────────────────────────
PROMPT_TUTORIAL_TRANSLATE_WITH_IMAGES = Template("""
You are a professional translator.

Task: Translate the tutorial section below to {{ lang }}.

CRITICAL RULES:

**1. Translate EVERYTHING**
- Every sentence, paragraph, and list item
- Keep the same length and detail level
- Don't skip or summarize anything

**2. Image References ({{ image_count }} in this section: {{ available_image_ids }})**
- Keep [IMG_X_Y] tokens EXACTLY as they are
- [IMG_4_1] stays as [IMG_4_1] - DO NOT translate or modify
- Must have exactly {{ image_count }} image reference(s) in translation

Example:
- Original: "The architecture [IMG_4_1] shows..."
- ✅ Correct: "아키텍처 [IMG_4_1]는 보여줍니다..."
- ❌ Wrong: "아키텍처는 보여줍니다..." (missing image)

**3. Preserve Structure**
- Keep all headings (# ## ###)
- Keep all formatting (**bold**, *italic*, lists)
- Keep all line breaks

---

Content to translate:
{{ text }}

---

Check before submitting:
- Translated everything? Same length?
- {{ image_count }} [IMG_*] tokens present and unchanged?
- Structure identical?
""")

PROMPT_TUTORIAL_TRANSLATE_NO_IMAGES = Template("""
You are a professional translator.

Task: Translate the tutorial section below to {{ lang }}.

⚠️ This section has NO images.

CRITICAL RULES:

**1. Translate EVERYTHING**
- Every sentence, paragraph, and list item
- Keep the same length and detail level
- Don't skip or summarize anything

**2. NO Images**
- This section has no [IMG_*] tokens
- Don't add any image references

**3. Preserve Structure**
- Keep all headings (# ## ###)
- Keep all formatting (**bold**, *italic*, lists)
- Keep all line breaks

---

Content to translate:
{{ text }}

---

Check before submitting:
- Translated everything? Same length?
- NO [IMG_*] tokens in translation?
- Structure identical?
""")

# ─────────────────────────────────────────
# 새 멀티모달 자습서용 프롬프트 (간소화)
# ─────────────────────────────────────────
PROMPT_TUTORIAL = Template("""
You are an expert tutor creating a comprehensive self-study guide.

Your task: Transform the chunks below into a complete tutorial that helps learners understand the document.

## Guidelines:

**1. Structure**
- Create a title (# H1) and table of contents
- Organize into clear sections (## H2) and sub-topics (### H3)
- Follow a logical learning flow

**2. Content**
- Explain concepts clearly and naturally
- Use examples when helpful
- Make complex ideas accessible

**3. Images**
The chunks contain image placeholders like [IMG_3_1], [IMG_4_1], etc.
- Use ONLY image IDs that appear in chunks
- Use each image EXACTLY ONCE
- Place [IMG_X_Y] where it helps understanding
- After each image, briefly explain what it shows (1-2 sentences)
- DO NOT create new image IDs

Example:
```
The transformer uses encoder and decoder stacks.

[IMG_4_1]

The diagram shows the encoder processing input and decoder generating output.
```

**4. Key Takeaways**
- End each major section with "### Key Points" (3-5 bullets)
- Include final "## Key Takeaways" section at the end

---

Source chunks:
{{ chunks }}

---

Remember:
- Use all available images exactly once
- Keep explanations clear and natural
- Make it easy to learn from
""")

# ─────────────────────────────────────────
# 섹션별 이미지 제한 프롬프트 (간소화)
# ─────────────────────────────────────────
PROMPT_TUTORIAL_SECTION_WITH_IMAGES = Template("""
You are an expert tutor creating a self-study guide based on a specific document.

Your task: Write a tutorial section that explains ONLY the concepts and information from the chunks below.

🚨 CRITICAL RULES:
- ONLY use information that appears in the chunks below
- DO NOT invent concepts, examples, or information not in the chunks
- DO NOT add generic knowledge or examples unrelated to the chunks
- Base your explanation STRICTLY on what is written in the chunks

## Guidelines:

**1. Topic & Structure**
- Identify the main topic from the chunks (not generic topics)
- Use Markdown headings (## for main topic, ### for sub-topics)
- Use actual terms, names, and concepts from the chunks

**2. Explanation**
- Explain ONLY what is described in the chunks
- Use specific examples, formulas, or details from the chunks
- If the chunks mention specific methods, algorithms, or techniques, explain those
- Keep language accessible but accurate to the source material

**3. Images ({{ image_count }} available: {{ available_image_ids }})**
- Use each image EXACTLY ONCE where it helps understand the chunk content
- Place image reference like: [IMG_4_1]
- After the image, explain what it shows based on the chunk context (1-2 sentences)
- DO NOT create new image IDs - only use: {{ available_image_ids }}

---

Source chunks (use ONLY this information):
{{ chunks }}

---

Remember:
- Extract and explain ONLY what is in the chunks above
- Use all {{ image_count }} image(s) exactly once
- Do not add generic programming or learning examples unless they appear in the chunks
- Be specific to the document content, not generic
""")

PROMPT_TUTORIAL_SECTION_NO_IMAGES = Template("""
You are an expert tutor creating a self-study guide based on a specific document.

Your task: Write a tutorial section that explains ONLY the concepts and information from the chunks below.

🚨 CRITICAL RULES:
- ONLY use information that appears in the chunks below
- DO NOT invent concepts, examples, or information not in the chunks
- DO NOT add generic knowledge or examples unrelated to the chunks
- Base your explanation STRICTLY on what is written in the chunks

⚠️ Note: This section has NO images - focus on clear text explanations.

## Guidelines:

**1. Topic & Structure**
- Identify the main topic from the chunks (not generic topics)
- Use Markdown headings (## for main topic, ### for sub-topics)
- Use actual terms, names, and concepts from the chunks

**2. Explanation**
- Explain ONLY what is described in the chunks
- Use specific examples, formulas, or details from the chunks
- If the chunks mention specific methods, algorithms, or techniques, explain those
- Break down complex ideas step-by-step based on chunk content
- Keep language accessible but accurate to the source material

---

Source chunks (use ONLY this information):
{{ chunks }}

---

Remember:
- Extract and explain ONLY what is in the chunks above
- NO images available - explain everything with text
- Do not add generic programming or learning examples unless they appear in the chunks
- Be specific to the document content, not generic
""")
