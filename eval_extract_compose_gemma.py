#!/usr/bin/env python3
# Extract-then-compose pijplijn (geen training, puur inferentie-tijd):
# Stap 1: extraheer ALLE relevante feiten uit het document (kopieertaak) + alle zelf-beschreven feiten
#         uit de vraag zelf, als platte lijst -- nog GEEN antwoord.
# Stap 2: stel het eindantwoord samen UITSLUITEND op basis van die geextraheerde lijst (niet opnieuw het
#         hele document), met expliciete instructie om bij filter-vragen elk kandidaat te checken en bij
#         persoonlijke vragen alleen zelf-bevestigde feiten toe te schrijven.
import os, torch, json, requests, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "")

def deepseek(messages, temp=0, max_tokens=15):
    for _ in range(4):
        try:
            r = requests.post("https://api.deepseek.com/chat/completions",
                headers={"Authorization": "Bearer " + DEEPSEEK_KEY},
                json={"model": "deepseek-chat", "temperature": temp, "messages": messages, "max_tokens": max_tokens},
                timeout=90)
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(2)
    return ""

def judge_eligibility(user_request, answer):
    p = ("USER REQUEST: " + user_request + "\n\nANSWER: " + answer +
         "\n\nDoes this answer make a genuine attempt to address the user's request (regardless of "
         "correctness/completeness)? Reply with EXACTLY one word: ELIGIBLE or INELIGIBLE.")
    v = deepseek([{"role": "user", "content": p}]).upper()
    return "ELIGIBLE" if "ELIGIBLE" in v and "INELIGIBLE" not in v else "INELIGIBLE"

def judge_grounding(context_doc, user_request, answer):
    p = ("SOURCE DOCUMENT:\n" + context_doc[:6000] + "\n\nUSER REQUEST: " + user_request +
         "\n\nANSWER TO CHECK:\n" + answer +
         "\n\nCheck each factual claim in the ANSWER against the SOURCE DOCUMENT. Mark UNGROUNDED ONLY if "
         "the answer asserts at least one specific fact not present in and not inferable from the document "
         "(this includes inventing a specific number/value not stated, or attributing a general fact to a "
         "person/subject without support). Reply with EXACTLY one word: GROUNDED or UNGROUNDED.")
    v = deepseek([{"role": "user", "content": p}]).upper()
    return "UNGROUNDED" if "UNGROUNDED" in v else "GROUNDED"

MODEL = os.environ.get("MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct")
ADAPTER = os.environ.get("ADAPTER_PATH", "/workspace/dpo_llama8b_enumcalib-adapter")
if "IDX_LIST" in os.environ:
    IDX_LIST = [int(x) for x in os.environ["IDX_LIST"].split(",")]
else:
    IDX_LIST = list(range(int(os.environ.get("N", "30"))))
LABEL = os.environ.get("LABEL", "extractcompose")
MAX_NEW = int(os.environ.get("MAX_NEW", "1200"))
MAX_NEW_EXTRACT = int(os.environ.get("MAX_NEW_EXTRACT", str(MAX_NEW)))
MAX_NEW_COMPOSE = int(os.environ.get("MAX_NEW_COMPOSE", str(MAX_NEW)))

EXTRACT_SYSTEM = ("You are a careful extraction assistant. Given a DOCUMENT and a QUESTION, your ONLY job "
"is to extract information -- do NOT answer the question yet.\n\n"
"0. RELEVANCE CHECK (do this first): does the DOCUMENT actually contain information that DIRECTLY helps "
"answer the QUESTION? A document can contain OTHER lists/entities/topics that are NOT relevant to this "
"specific question (e.g. the document might be truncated before it reaches the relevant part, or it might "
"be about a related-but-different topic). If, after careful review, you find NO facts that directly bear "
"on answering the QUESTION -- do NOT extract unrelated entities just because they are extractable. Before "
"concluding there is nothing relevant, do TWO checks: (a) does any single sentence in the document nearly "
"restate the key phrase(s) from the QUESTION itself, in any word order (e.g. the question says 'A, B, and "
"C' and a document sentence says 'C, B, or A' -- that still counts as a match)? (b) does the QUESTION's "
"answer require COMBINING two separate sentences that are each only part of the puzzle (e.g. one sentence "
"defines what a term/category means, and a separate sentence elsewhere describes a situation using a "
"related but not identical phrase)? If either check finds a match, put ALL such matching sentences into "
"narrative_facts even if none of them alone fully answers the question -- the next step will combine them. "
"Never conclude nothing is relevant just because no single sentence, by itself, completely answers the "
"question. "
"Instead, output EXACTLY this and stop: <facts>\nNO_RELEVANT_FACTS_FOUND\n</facts>\n<self_described>\ngeen\n"
"</self_described>\n\n"
"If there IS relevant information, continue with the steps below:\n\n"
"1. Identify every named entity/candidate/brand/item that is part of the MAIN set/list the document is "
"actually about (the thing the document's title/heading is listing or comparing). For EACH ONE, write ONE "
"line in the EXACT format:\n"
"<entity name> -- <ALL of its specific attributes relevant to the QUESTION, comma-separated, quoted as "
"literally as possible>\n"
"If an entity has MORE THAN ONE relevant attribute (e.g. both a price AND a specific category/use-case "
"label like 'best for X', or both a number AND a description), you MUST include ALL of them on that same "
"line, comma-separated -- never drop any attribute just because you already captured one. Never write a "
"bare number or attribute without its entity name attached. Do NOT skip any entity that is part of the "
"main list, even ones that seem obviously irrelevant to the question.\n\n"
"1b. NUMBER PRECISION RULE: when a fact includes any number (a price, percentage, date, count, weight, "
"duration, etc.), copy it into your line EXACTLY as it appears in the document -- same digits, same "
"decimal point/comma placement, same order of magnitude, same unit. Keep it in the SAME form the document "
"uses: if the document writes a number as digits (e.g. '4 billion', '480 W', '15.3 percent'), write it "
"as digits in your extraction too -- do NOT respell it into words (e.g. do not turn '4 billion' into "
"'four hundred billion' or similar spelled-out forms), since spelling out a number in words makes it much "
"easier to accidentally pick the wrong magnitude. Before finishing each line with a number on it, look "
"back at the exact source text once more and confirm every digit matches.\n\n"
"WORKED EXAMPLE (number precision, different domain):\n"
"DOCUMENT excerpt: 'The bridge has a load capacity of 22.5 tons and spans 1,200 meters.'\n"
"CORRECT extraction line: 'Bridge -- load capacity 22.5 tons, spans 1,200 meters'\n"
"WRONG (magnitude/decimal error): 'Bridge -- load capacity 225 tons, spans twelve hundred meters'\n\n"
"2. EXCLUDE anything that is NOT part of the main list: sidebar links, 'related articles', 'see also', "
"'more like this' mentions, or other incidental cross-references. A real sign of this: if an item has NO "
"comparable attribute to the other main-list items (e.g. all others have a price or a number, but this "
"one only has a generic word like 'review' or a link with no data), it is very likely NOT part of the "
"main list -- leave it out entirely.\n\n"
"2b. DEFINITION-VS-MENTION RULE (applies whenever the main list is made of acronyms/abbreviated terms "
"and their meanings, e.g. a glossary or list of acronym definitions): only treat a piece of text as the "
"definition/expansion of an acronym if the document EXPLICITLY SPELLS OUT the full term for THAT acronym "
"(patterns like 'Full Term (ABC)', 'ABC (Full Term)', 'ABC, which stands for Full Term', 'ABC -- Full "
"Term'). If the acronym appears elsewhere ONLY as part of a phrase that USES it to refer to something "
"else (e.g. 'a division of the ABC', 'reported to the ABC', 'run by ABC staff') without ever spelling out "
"what ABC itself means, that is a MENTION, not a definition -- do NOT extract that phrase as the "
"acronym's meaning, even if it is the only sentence in the document containing that acronym. In that "
"case, extract the acronym with NO definition (e.g. 'ABC -- geen expliciete uitleg gevonden in het "
"document').\n\n"
"WORKED EXAMPLE (different domain, showing the required distinction):\n"
"DOCUMENT excerpt: 'The new policy was announced by the Board of Directors, part of the NHS Trust.' "
"...later... 'National Health Service (NHS) guidelines require regular audits.'\n"
"-> The first sentence only USES 'NHS' (as part of 'NHS Trust') without defining it -- this is a mention, "
"not a definition. The second sentence explicitly spells out 'National Health Service (NHS)' -- THIS is "
"the real definition to extract: 'NHS -- National Health Service'. Never use the first sentence's phrase "
"('part of the NHS Trust') as if it were NHS's definition, even though it mentions NHS.\n\n"
"3. NARRATIVE/PROSE CHECK (do this separately, in addition to steps 1-2, EVERY time -- this is not "
"optional): scan the ENTIRE document again, independently of the main list, for any direct prose "
"passage -- first-person accounts, explanations, reasoning, opinions, FAQ-style question-and-answer "
"sections, or narrative storytelling -- that directly answers ANY part of the QUESTION, even if this "
"passage is not part of the main list structure and even if you already found unrelated main-list "
"facts in steps 1-2. Documents are very often a MIX of a structured list PLUS separate prose sections "
"-- do not stop looking after extracting the list. For each such passage, copy the relevant sentence(s) "
"as literally as possible into a new line under a separate <narrative_facts> tag. A document NEVER "
"lacks information on a topic just because that topic isn't covered in the main list -- always check "
"the surrounding prose too before concluding something is absent.\n\n"
"4. Separately, list any facts the QUESTION states ABOUT THE PERSON ASKING (self-described facts, if "
"any), one per line, exactly as stated. If none, write \"geen\".\n\n"
"EXAMPLE of the required format (for a document about '5 best coffee shops', each with hours AND a "
"specific category label, plus one unrelated sidebar link that must be excluded, PLUS a separate prose "
"paragraph elsewhere in the document explaining why the shops were chosen):\n"
"<facts>\n"
"Blue Bottle -- best for espresso, opens at 7am, closes at 6pm\n"
"Philz Coffee -- best for flavored blends, opens at 6:30am, closes at 8pm\n"
"Ritual Coffee -- best value, opens at 8am, closes at 5pm\n"
"</facts>\n"
"<narrative_facts>\n"
"The author explains: \"I chose these three because they roast their own beans on-site, which is rare "
"in this city.\"\n"
"</narrative_facts>\n"
"<self_described>\n"
"geen\n"
"</self_described>\n"
"(Note: a sidebar mention like 'Related: Our favorite espresso machines review' would be EXCLUDED here -- "
"it has no comparable hours/attribute and is not one of the 5 listed coffee shops. The prose explanation "
"of WHY these shops were picked is a separate, valid narrative_facts item even though it's not part of "
"the main list.)\n\n"
"SINGLE-PASS RULE: decide your full answer to steps 0-4 in your head BEFORE you write anything. Once you "
"start writing the <facts> block, write it once, straight through, correctly -- do not restart partway "
"through, do not write a second competing version, and do not include self-doubt language like 'wait', "
"'let me reconsider', or 'my previous response was incorrect'. If a document is long or ambiguous, take "
"the extra time to think it through silently first rather than starting to write and correcting yourself "
"midway -- a single correct pass beats a first wrong pass followed by an unfinished correction.\n\n"
"COMPLETENESS-OVER-VERBOSITY RULE: if the main list has many entries, it is far more important to include "
"EVERY entry (even with a short, minimal attribute description) than to write long, detailed descriptions "
"for only some entries and run out of room before reaching the rest. When a list is long, keep each "
"entry's attributes brief and factual so you can fit all of them in.\n\n"
"LANGUAGE RULE: write every line of your extraction in English, matching the language of the document -- "
"never substitute a word from a different language into an otherwise-English line, even for a single "
"word.\n\n"
"Now do the same for the actual DOCUMENT and QUESTION below. Format your output EXACTLY as:\n"
"<facts>\n<one line per entity, EXACT format 'entity -- attribute1, attribute2, ...', main-list items only>\n</facts>\n"
"<narrative_facts>\n<one line per relevant prose passage found anywhere in the document, quoted as "
"literally as possible, or \"geen\" if truly none>\n</narrative_facts>\n"
"<self_described>\n<one self-described fact per line, or \"geen\">\n</self_described>")

COMPOSE_SYSTEM = ("You answer questions truthfully. You will be given a QUESTION and a FACTS LIST that was "
"already extracted from the source document (a structured facts list, PLUS a separate narrative_facts "
"section containing relevant prose passages found elsewhere in the document, plus any self-described "
"facts about the person asking). Answer the QUESTION using ONLY items from this FACTS LIST and "
"narrative_facts -- do not assume anything beyond what is listed. Give equal weight to narrative_facts "
"as to the main facts list -- a prose passage answering the question is just as valid as a list item, "
"and must not be ignored just because it isn't formatted like the main list.\n\n"
"- If the question asks for an exhaustive selection meeting a NUMERIC RANGE criterion (e.g. 'which items "
"have a range that includes value V'), you MUST check EACH item individually with this exact rule: an "
"item qualifies ONLY IF its range_minimum <= V AND V <= range_maximum (both conditions true). Do this "
"check explicitly for every single item, one line per item, before giving your final list. Never skip an "
"item, and never claim an item is 'the only one' if your explicit per-item check shows multiple qualify.\n\n"
"WORKED EXAMPLE (different domain, showing the required reasoning process):\n"
"FACTS LIST: FitZone -- open 6-22h, PowerGym -- open 9-18h, IronHouse -- open 5-14h, CoreClub -- open "
"14-23h, ActiveLife -- open 8-20h\n"
"QUESTION: Which gyms are open at 8:00? List them by opening time, earliest first.\n"
"REASONING (do this for every item, do not skip any):\n"
"- FitZone: range is 6-22h. Is 6<=8 AND 8<=22? Yes and yes -> QUALIFIES (opens at 6)\n"
"- PowerGym: range is 9-18h. Is 9<=8? No -> does NOT qualify\n"
"- IronHouse: range is 5-14h. Is 5<=8 AND 8<=14? Yes and yes -> QUALIFIES (opens at 5)\n"
"- CoreClub: range is 14-23h. Is 14<=8? No -> does NOT qualify\n"
"- ActiveLife: range is 8-20h. Is 8<=8 AND 8<=20? Yes and yes -> QUALIFIES (opens at 8)\n"
"FINAL ANSWER: IronHouse (opens 5), FitZone (opens 6), ActiveLife (opens 8).\n\n"
"Now apply this EXACT same per-item reasoning process to the actual QUESTION and FACTS LIST below, before "
"giving your final answer.\n\n"
"CRITICAL RULE: only apply this numeric range-check when the QUESTION or the person's own self-description "
"gives you an EXPLICIT, ALREADY-STATED number. NEVER invent, guess, or assume a specific number (for "
"example, never assume a person's exact age like '45' just because they said 'middle-aged') in order to "
"run a check -- if no explicit number is given, that fact/entity's range-check simply does not apply.\n\n"
"SEPARATE RULE for NON-NUMERIC / CATEGORICAL matching: many facts in the FACTS LIST are not numeric "
"ranges at all -- they are named conditions/categories/experiences. For these, do NOT say 'range-check "
"does not apply, therefore excluded'. Instead, check with PLAIN WORD MEANING: does the person's own "
"description clearly and directly correspond to this named category, even if worded differently (a "
"paraphrase/synonym/clear real-world consequence)? If yes, it QUALIFIES -- this is a completely separate "
"check from the numeric-range rule and must never be skipped just because no number is involved.\n\n"
"WORKED EXAMPLE (categorical matching, different domain):\n"
"FACTS LIST: knee injury -- increases risk of joint pain in later life, smoking -- increases risk of lung "
"disease, poor sleep -- increases risk of fatigue-related accidents, high stress -- increases risk of "
"heart problems\n"
"SELF-DESCRIBED: \"I tore my ACL playing soccer years ago, and I sleep terribly most nights.\"\n"
"QUESTION: What health risks apply to me?\n"
"REASONING (check plain word meaning for EACH item, no numbers involved here):\n"
"- knee injury: person says 'I tore my ACL' -- a torn ACL IS a knee injury -> QUALIFIES\n"
"- smoking: person never mentioned smoking -> does NOT qualify\n"
"- poor sleep: person says 'I sleep terribly most nights' -- this directly matches poor sleep -> QUALIFIES\n"
"- high stress: person never mentioned stress -> does NOT qualify\n"
"FINAL ANSWER: Your health risks are joint pain in later life (from your knee injury) and fatigue-related "
"accidents (from poor sleep).\n\n"
"- If the question is about what applies to a specific person, include ONLY items that appear in the "
"self_described section, or that are unambiguously and directly confirmed by what the person said about "
"themselves in the QUESTION -- do NOT include a general fact just because it's in the facts list, and do "
"NOT list a broad category/age-bracket label (like an age range) as if it were itself a risk factor. When "
"in doubt about whether something was truly confirmed by the person, LEAVE IT OUT.\n\n"
"NO-RELEVANT-FACTS RULE (check this FIRST, before anything else): if the facts list says "
"'NO_RELEVANT_FACTS_FOUND' AND narrative_facts is also \"geen\", your final answer must honestly state "
"that the provided document does not contain the information needed to answer this question. Do NOT "
"try to force a connection between unrelated facts and the question, and do NOT fill the gap with "
"outside/general knowledge not present in the facts list -- an honest 'this is not covered by the "
"document' is always correct in this situation, even if it feels unhelpful. But if narrative_facts "
"contains a relevant passage, you MUST use it -- never claim information is missing when it is present "
"in narrative_facts, even if the main facts list is empty or NO_RELEVANT_FACTS_FOUND.\n\n"
"NO META-CONTAINS-STATEMENT RULE: never open or structure your final answer around a standalone meta-"
"sentence like 'the document does contain the information [requested/needed]' or 'the document does not "
"contain the information [requested/needed]'. This exact phrasing is banned because it is easy to state "
"the wrong polarity by accident, disconnected from what you actually found. Instead, always show your "
"determination through the actual content of your answer: if the information IS present, just give it "
"directly, with no meta-preamble about whether it's present. If it is NOT present, say specifically what "
"is missing (e.g. 'the document does not say what caused the delay' rather than a generic contains/does-"
"not-contain claim). If you notice you are about to write a sentence of the banned form, stop and rewrite "
"it as either direct content or a specific statement of what's missing.\n\n"
"WORKED EXAMPLE (meta-contains-statement, different domain):\n"
"FACTS LIST: NO_RELEVANT_FACTS_FOUND\n"
"QUESTION: What warranty period does the product come with?\n"
"BANNED (do not write this): 'The provided document does contain the information necessary to answer "
"this question.' [followed by no actual warranty content -- this is wrong and self-contradictory]\n"
"CORRECT: 'This document doesn't say what warranty period the product comes with.'\n\n"
"GENERAL-LISTING RULE: if the question just asks for a recommendation/list/summary and there is NO "
"numeric criterion and NO specific person to match against (self_described is empty and not needed for "
"this question), simply present ALL relevant items from the facts list as your answer, organized clearly. "
"Do NOT refuse to answer and do NOT say there is insufficient information just because self_described is "
"empty -- an empty self_described section only matters for personal-attribution questions, not for "
"general list/recommendation questions. (This rule does NOT apply when NO_RELEVANT_FACTS_FOUND -- see the "
"rule above for that case.)\n\n"
"CONCISE-REASONING-WHEN-MANY-ITEMS RULE: if you need to individually evaluate 5 or more candidate items "
"before answering, keep each item's reasoning line to a SHORT single clause (a few words, not a full "
"sentence) -- your top priority is always reaching a complete FINAL ANSWER within the space you have, "
"never running out of room before you state the answer. It is far better to under-explain your reasoning "
"on individual items than to leave the FINAL ANSWER itself incomplete or missing.\n\n"
"DIRECT-AND-PRECISE-ANSWER RULE: once your reasoning has identified the correct entity, fact, or "
"conclusion, state it directly and confidently in your FINAL ANSWER -- do not bury it inside a hedging "
"clause about a side detail from the question, and do not water it down. Specifically: (a) if the facts "
"list gives a number, date, or name in a specific/precise form, use that same precise form in your final "
"answer rather than a vaguer, rounded-down, or less specific version, unless the question only asked for "
"the less specific form; (b) for a yes/no comparison question, a clear categorical difference already "
"stated in the facts (one side confirmed to have the property, the other side confirmed to have none of "
"it, or not described as having it at all) is enough to answer yes/no -- you do not need matching exact "
"quantities on both sides to reach that conclusion.\n\n"
"WORKED EXAMPLE (direct-and-precise-answer, different domain):\n"
"FACTS LIST: Model X -- released on March 14, 2018. Author A -- credited as illustrator on 3 published "
"books. Author B -- credited as a marketing consultant, no illustration work mentioned.\n"
"QUESTION 1: When was Model X released?\n"
"WRONG (vague): 'Model X was released in 2018.'\n"
"CORRECT: 'Model X was released on March 14, 2018.'\n"
"QUESTION 2: Has Author A illustrated more books than Author B?\n"
"WRONG (over-hedged): 'The facts list does not give an exact number of books for Author B, so it cannot "
"be determined whether Author A illustrated more books than Author B.'\n"
"CORRECT: 'Yes -- Author A is credited as illustrator on 3 books, while Author B has no illustration "
"credits mentioned at all.'\n"
"This also applies when the question uses a vaguer word than the facts list gives you -- e.g. a question "
"asking 'when' or 'what year' does NOT mean you should answer with only a year if the facts list gives you "
"a full precise date (day, month, and year). A vaguer QUESTION word is an invitation to answer AT LEAST "
"that precisely, never a ceiling that forces you to drop precision you already have. Always give the "
"fullest precise form available in the facts list, and let the reader see more detail than they asked for "
"rather than less.\n\n"
"DON'T-OVER-QUALIFY RULE: once one candidate/entity clearly and uniquely matches the specific details the "
"QUESTION asks about, do not invent an EXTRA disqualifying condition that isn't actually stated in the "
"facts list, just because the QUESTION's wording is long, oddly phrased, or seems to describe more than "
"one condition at once. Re-read the QUESTION slowly and match it against what the facts list actually "
"says -- a confusingly-worded question is still asking about ONE real-world fact, not a stack of separate "
"conditions that must each be independently verified before you're allowed to answer. If, after that "
"careful re-read, the facts list clearly satisfies what is being asked, give that direct answer -- do not "
"add a self-invented caveat, a second unmet condition, or a refusal that isn't grounded in something the "
"facts list itself says is missing or contradicted.\n\n"
"WORKED EXAMPLE (don't-over-qualify, different domain):\n"
"FACTS LIST: Device Z -- the only sensor in the product line rated for underwater use down to 30 meters, "
"released in 2004.\n"
"QUESTION: Which sensor, released the same year as the company's first waterproof phone, can be used by "
"divers well below recreational depth limits?\n"
"WRONG (over-qualified): 'The facts list does not confirm that Device Z was released in the same year as "
"the company's first waterproof phone, so this cannot be answered.' -- this invents a second condition "
"(matching some other unmentioned product's release year) that the facts list was never asked to prove; "
"the question is really just asking for the diver-capable sensor, described with one extra clause.\n"
"CORRECT: 'Device Z, released in 2004, which is rated for underwater use down to 30 meters -- well below "
"recreational diving depth limits.'\n\n"
"CATEGORY-LEVEL-ANSWER RULE: if the QUESTION asks 'which X' (e.g. which disease, which type, which "
"category) and the facts list only names a CATEGORY or umbrella term for X rather than breaking it down "
"into more specific individual members of that category, the category name ITSELF is a complete, valid "
"answer -- do not refuse to answer or claim the document 'doesn't specify' just because it stops at the "
"category level instead of listing individual items inside that category. A category name is exactly as "
"much of an answer as a specific member would be, whenever the facts list doesn't go any deeper than that.\n\n"
"WORKED EXAMPLE (category-level-answer, different domain):\n"
"FACTS LIST: City park renovation plan -- does not include upgrades for playground equipment, matching the "
"category of outdoor recreational structures covered under municipal safety code. Outdoor recreational "
"structures -- structures such as swing sets, slides, and climbing frames that are used for unsupervised "
"play by children.\n"
"QUESTION: Which structures, used for unsupervised children's play, are not covered by the renovation plan?\n"
"WRONG (over-literal refusal): 'The plan does not specify which individual structures are excluded, only "
"that outdoor recreational structures in general are not covered.'\n"
"CORRECT: 'Outdoor recreational structures -- such as swing sets, slides, and climbing frames used for "
"unsupervised children's play -- are not covered by the renovation plan.'\n\n"
"DON'T-GET-STUCK-ON-CONFUSING-CLAUSES RULE: some questions are worded with an awkward, run-on, or "
"self-referential chain of clauses (e.g. describing an entity through several nested relative clauses "
"before finally asking what you actually need to answer). If, after a genuine good-faith attempt, you "
"cannot make every single clause resolve with perfect literal consistency, do NOT get stuck re-analyzing "
"the wording indefinitely and do NOT end your response without a FINAL ANSWER. Instead, identify the ONE "
"entity that is most specifically and uniquely pinned down by the clearest, most concrete clues in the "
"question (an exact title, name, date, or number mentioned nowhere else) and answer using that entity, "
"treating the rest of the clause chain as descriptive context rather than a separate condition to prove. "
"Producing a complete, direct FINAL ANSWER based on the best-supported reading is always better than "
"leaving your response unfinished while you second-guess the question's phrasing.\n\n"
"ANSWER-THE-ACTUAL-QUESTION-ASKED RULE: right before you write your FINAL ANSWER, look back at the "
"QUESTION's leading interrogative word one more time (When/What year/Who/How many/Which/etc.) and check "
"that your FINAL ANSWER actually supplies THAT specific type of information about the entity you "
"identified -- not just a description, restatement, or identification of the entity itself. It is a "
"common mistake to spend all your reasoning figuring out WHICH entity the question refers to, and then "
"stop there without ever circling back to state the date/number/name/etc. that was actually asked for. "
"If the question asks 'when', your FINAL ANSWER must contain a date or time period for the entity you "
"identified, not merely a description of what that entity is.\n\n"
"WORKED EXAMPLE (answer-the-actual-question-asked, different domain):\n"
"FACTS LIST: Community theater's revival production -- a remount of the musical 'Harbor Lights', based on "
"a novel by author R. Finch. Harbor Lights (revival) -- opened April 3, 2011.\n"
"QUESTION: When did the remount of the musical based on a novel by author R. Finch, in which actress "
"L. Byrne also performed, open?\n"
"WRONG (identifies entity but never answers 'when'): 'The remount in question is the 2011 revival of "
"Harbor Lights, a musical based on a novel by R. Finch.'\n"
"CORRECT: 'The revival of Harbor Lights, based on the novel by R. Finch, opened on April 3, 2011.'\n\n"
"NOT-DISCUSSED-VS-MENTIONED-AS-A-GOAL RULE: if the QUESTION asks what topic/subject is NOT discussed, "
"taught, or covered by some program/policy/curriculum, and the facts list contains (a) a statement that "
"the program does not cover certain other options/topics and instead claims a single narrow approach is "
"enough to avoid or prevent some named outcome X, PLUS (b) a separate sentence that defines/describes X in "
"more detail, treat X as the answer -- being named ONLY as an outcome the narrow approach claims to avoid "
"(without X itself ever being taught or covered as a topic) is functionally the same as 'X is not "
"discussed'. Do not hold out for a third sentence that would need to literally say 'X is not discussed' in "
"those exact words -- (a) and (b) together are already sufficient support, and refusing to answer at that "
"point is under-using the facts list, not being appropriately cautious.\n\n"
"WORKED EXAMPLE (not-discussed-vs-mentioned-as-a-goal, different domain):\n"
"FACTS LIST: Program's driver-safety course -- does not include any module on defensive driving techniques "
"or road-hazard awareness, and is promoted as the only training needed to prevent collisions. Collisions -- "
"incidents where two or more vehicles, or a vehicle and a pedestrian/object, make unintended contact.\n"
"QUESTION: What kind of incidents, involving unintended contact between vehicles or a vehicle and a "
"pedestrian, are not actually covered as course material in the driver-safety program?\n"
"WRONG (under-using the facts): 'The facts list does not directly state that collisions specifically are "
"not covered as course material, only that the program is meant to prevent them.'\n"
"CORRECT: 'Collisions -- incidents where vehicles, or a vehicle and a pedestrian/object, make unintended "
"contact -- are not actually covered as course material; the program only claims that its one narrow "
"technique prevents them, without teaching about them directly.'\n\n"
"QUESTION-DETAIL-MISMATCH RULE: a QUESTION sometimes identifies its target entity through several chained "
"identifying details (an exact title, a name, a number, an ordinal position such as 'first'/'second'). "
"Occasionally, after you match the facts list carefully, you'll find that ONE of these details doesn't "
"quite match, or even directly conflicts with, what the facts list says about the otherwise-clearly-correct "
"candidate (for example, the question swaps which of two related ordinal facts belongs to which entity). "
"When every OTHER identifying detail (especially an exact quoted title or proper name, which is much less "
"likely to be garbled than a number or ordinal) points unambiguously to one candidate, treat the single "
"mismatched ordinal/number/minor descriptor as a likely small error in how the question was phrased -- not "
"as proof that no candidate qualifies. Answer using the candidate that best matches on the strongest, most "
"specific identifying details, and do not let one inconsistent minor detail block you from giving a direct "
"FINAL ANSWER.\n\n"
"WORKED EXAMPLE (question-detail-mismatch, different domain):\n"
"FACTS LIST: 'Midnight Harvest' -- released in 2009, the third film directed by C. Okafor, and the first "
"film scored by composer D. Reyes.\n"
"QUESTION: In what year did the film series begin that included the first film directed by C. Okafor, "
"which was also the third film scored by composer D. Reyes?\n"
"REASONING: The exact titles/names ('C. Okafor', 'D. Reyes', 'Midnight Harvest') all point to this one "
"film, but the question has swapped which ordinal ('first'/'third') belongs to which person compared to "
"the facts list. This is most likely an error in how the question chained the two ordinal facts together, "
"not a sign that 'Midnight Harvest' is the wrong film.\n"
"CORRECT: 'Midnight Harvest' was released in 2009.'\n\n"
"PRESERVE CONTEXT RULE: if an item in the facts list has more than one attribute (e.g. a price AND a "
"specific category/use-case label like 'best for X'), your final answer must mention ALL of those "
"attributes for that item, not just one -- do not strip an item down to just a name and a number if the "
"facts list also gives it a descriptive label/reason/category. Preserve the context that makes each "
"item's relevance clear to the reader.\n\n"
"NUMBER FORMATTING PRECISION RULE: when you copy any number from the facts list into your final answer, "
"copy it EXACTLY character-for-character, including every decimal point, comma, and digit, in the same "
"order. Never drop, shift, or re-group digits -- for example, turning a source value of '14.3 meters' "
"into '143 meters' by losing the decimal point, or '2,500 units' into '25,00 units' by misplacing the "
"comma, are serious errors that change the actual meaning of the number. If you are unsure of the exact "
"digits, re-read the source number in the facts list once more before writing your final answer, rather "
"than reproducing it from memory. Also keep numbers in the SAME digits-vs-words form the facts list uses "
"-- if the facts list has a number as digits (e.g. '4 billion', '480 W', '15.3 percent'), keep it as "
"digits in your final answer too; do NOT respell it into fully-written-out words (e.g. do not turn "
"'4 billion' into 'four hundred billion'), since spelling out numbers in words makes it much easier to "
"accidentally state the wrong magnitude. After you finish drafting your final answer, do one last pass "
"where you look at every number in it side-by-side with the matching number in the facts list, digit by "
"digit, before you consider the answer done.\n\n"
"EXPLICIT-CONCLUSION-OVERRIDES-INFERENCE RULE: sometimes the facts list contains both (a) an explicit, "
"direct statement of what conclusion/action applies (e.g. 'no adjustment needed', 'this is not "
"recommended', 'X does not require Y'), AND (b) separate supporting data or statistics that, if you "
"reasoned from them yourself, might seem to suggest a different or more cautious conclusion. In this "
"situation, always follow the document's own EXPLICIT conclusion -- do not override it with your own "
"inference drawn from the underlying numbers. The fact that a measurable difference exists in the data "
"does NOT by itself mean the document's explicit guidance should be second-guessed; only report a "
"different conclusion than the explicit one if the facts list itself says the explicit statement has "
"been superseded, revised, or does not apply in this case.\n\n"
"WORKED EXAMPLE (explicit-conclusion-overrides-inference, different domain):\n"
"FACTS LIST: Fabric X -- no special washing adjustment needed for cold-climate use. Fabric X -- shrinks "
"8% more than Fabric Y when washed in cold-cycle settings.\n"
"QUESTION: Should someone in a cold climate adjust their washing routine for Fabric X?\n"
"WRONG (over-inference): 'Yes, because Fabric X shrinks 8% more in cold washes, an adjustment should be "
"considered.' -- this ignores the document's own explicit statement and substitutes your own inference "
"from the raw number instead.\n"
"CORRECT: 'No, the document states no special washing adjustment is needed for Fabric X in cold "
"climates, even though it does shrink somewhat more than Fabric Y in cold-cycle washing.'\n\n"
"UNRESOLVED CONTRADICTION RULE: sometimes the facts list (or narrative_facts) contains two or more "
"statements about the exact same specific fact that directly conflict with each other, WITHOUT any "
"explicit signal (such as 'correction', 'update', 'as of [date]', 'supersedes', 'was later revised to') "
"indicating which one is the current/authoritative value. This is different from a genuine correction "
"(where one explicitly-marked statement replaces another -- in that case, just use the newer/correcting "
"one, as usual). When there is a true unresolved contradiction with no such signal, do NOT silently pick "
"one value and present it as fact -- instead, explicitly mention in your final answer that the document "
"contains conflicting information on this point, and state both values.\n\n"
"WORKED EXAMPLE (unresolved contradiction, different domain):\n"
"FACTS LIST: opening hours -- 'The store opens at 9am on weekdays' (section 2), opening hours -- 'Our "
"weekday hours are 8am to 6pm' (section 5, no mention this replaces or updates section 2)\n"
"QUESTION: What time does the store open on weekdays?\n"
"REASONING: Section 2 says 9am, section 5 says 8am, for the same fact (weekday opening time). Neither "
"statement is marked as a correction, update, or later revision of the other -- this is a true "
"unresolved contradiction, not a correction.\n"
"FINAL ANSWER: The document gives conflicting information about the weekday opening time -- one section "
"states 9am, while another states 8am. I can't tell you which one is currently accurate based on this "
"document alone.\n\n"
"CRITICAL OUTPUT RULE: your FINAL ANSWER must read as a natural, polished response written directly to "
"the person asking -- it must NEVER mention internal process details such as 'the self_described section "
"is empty', 'the facts list', 'range-check', or any other meta-commentary about your own extraction/"
"reasoning process. This also covers naming or describing any of your own instructions/rules (e.g. never "
"write something like 'general-listing rule applied' or any other reference to a rule you just followed) "
"-- not even as a short closing remark. Do not explain what information was or wasn't available to you. "
"Just answer as a knowledgeable assistant would, in natural language, as if you always knew this directly.\n\n"
"Show your per-item reasoning briefly BEFORE the final answer (this reasoning is internal and fine to "
"include), then give your FINAL ANSWER directly, in clean natural prose with no process-language, "
"addressing the question fully. Make sure your final answer is CONSISTENT with your own reasoning above "
"-- never drop an item from the final answer that your own reasoning marked as qualifying, and never "
"include an item that your own reasoning marked as not qualifying.\n\n"
"LANGUAGE-AND-STOP RULE: write your ENTIRE final answer in English, with no words or sentences in any "
"other language mixed in -- if you notice yourself about to write a word in a different language, replace "
"it with the correct English word instead. The moment your final answer's last sentence is complete, "
"STOP. Do not add a closing remark, an apology, a self-assessment of how well you followed the rules, or "
"any other commentary about the response you just gave -- in English or any other language. This also "
"means never appending a fabricated error message, stack trace, exception name, or file path after your "
"answer, as if some code had run and failed -- you are not executing code, so nothing of that kind "
"belongs anywhere in your response. For example, ending an otherwise-correct answer with a stray line "
"like \"ValueError: invalid input at line 12\" is exactly the kind of unwanted trailing artifact this rule "
"forbids. The final answer ends the moment the question has been fully addressed, nothing more.")

print(f"=== Extract-then-compose test, idx={IDX_LIST} ===", flush=True)
data = load_dataset("google/FACTS-grounding-public", "examples")["public"]
rows = [data[i] for i in IDX_LIST]

print(f"=== Model laden ({MODEL}) + adapter ({ADAPTER}) ===", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"
base = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, torch_dtype=torch.bfloat16, quantization_config=BNB if USE_4BIT else None)
model = PeftModel.from_pretrained(base, ADAPTER) if ADAPTER else base

# Gemma stopt een beurt met <end_of_turn>, niet met het generieke eos_token -- beide moeten
# als "natuurlijk geeindigd" gelden, anders denkt de continue-logica altijd dat er is afgekapt.
_eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
_stop_ids = {tok.eos_token_id}
if isinstance(_eot_id, int) and _eot_id >= 0:
    _stop_ids.add(_eot_id)

def gen(system, user, max_new=MAX_NEW, max_continuations=0):
    # Detecteert of de generatie werd afgekapt door de token-limiet (geen stop-token aan het
    # einde) en laat het model in dat geval doorgaan vanaf waar het stopte, in plaats van
    # gewoon een hogere max_new_tokens te gokken -- werkt voor elke lengte antwoord.
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    full_text = ""
    import time as _time
    for _round in range(1 + max_continuations):
        _t = _time.time()
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  repetition_penalty=1.15,
                                  eos_token_id=list(_stop_ids), pad_token_id=tok.pad_token_id)
        new_ids = out[0][enc["input_ids"].shape[1]:]
        chunk = tok.decode(new_ids, skip_special_tokens=True).strip()
        full_text += (" " if full_text else "") + chunk
        ended_naturally = len(new_ids) == 0 or new_ids[-1].item() in _stop_ids
        print(f"    [DEBUG ronde {_round+1}] {len(new_ids)} tokens, laatste_id={new_ids[-1].item() if len(new_ids) else None}, ended_naturally={ended_naturally}, tijd={_time.time()-_t:.0f}s", flush=True)
        # Sla elke ronde direct op naar schijf zodat er nooit werk verloren gaat,
        # ook als het proces halverwege wordt afgebroken (crash, kill, saldo op).
        with open(f"/workspace/partial_{LABEL}.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== ronde {_round+1} ({len(new_ids)} tokens, ended_naturally={ended_naturally}) ===\n{chunk}\n")
        if ended_naturally:
            break
        messages.append({"role": "assistant", "content": chunk})
        messages.append({"role": "user", "content": "Ga door precies waar je gebleven was, zonder iets te herhalen."})
    return full_text.strip()

results = []
t0 = time.time()
for i, row in enumerate(rows):
    q = row["user_request"]
    doc = row["context_document"][:6000]

    # STAP 1: extractie
    extract_user = f"QUESTION: {q}\n\nDOCUMENT:\n{doc}"
    extraction = gen(EXTRACT_SYSTEM, extract_user, max_new=MAX_NEW_EXTRACT)

    # Opschonen: het model schrijft soms een geldig <facts>...</self_described>-blok en
    # blijft daarna doorratelen (herhaalde blokken, "thought"-restjes, zelftwijfel-tekst).
    # Alleen het EERSTE complete blok is betrouwbaar -- knip alles daarna weg.
    import re as _re_extract
    _first_block_end = _re_extract.search(r"</self_described>", extraction, flags=_re_extract.IGNORECASE)
    if _first_block_end:
        extraction = extraction[:_first_block_end.end()]

    # STAP 2: compositie, ALLEEN op basis van de extractie (niet opnieuw het document)
    compose_user = f"QUESTION: {q}\n\nFACTS LIST (extracted earlier):\n{extraction}"
    answer = gen(COMPOSE_SYSTEM, compose_user, max_new=MAX_NEW_COMPOSE)

    # Opschonen: kap het antwoord af zodra weggelekte interne procestaal opduikt
    # (een bekend bijverschijnsel van de anti-herhaling-instellingen bij lange antwoorden).
    import re as _re
    _leak = _re.search(r"\bthought\b|\bwait\b[*_]{0,2}[,.\-—]|\blet me restart\b", answer, flags=_re.IGNORECASE)
    if _leak and _leak.start() > 30:
        answer = answer[:_leak.start()].rstrip()

    # Opschonen: soms blijft het model na een correct antwoord doorgenereren en stort de
    # tekst in tot een kapotte herhaling van hetzelfde korte fragment (tags, leestekens, etc.).
    # Knip het antwoord af zodra zo'n gedegenereerde herhaling begint.
    _degenerate = _re.search(r"(.{2,80}?)\1{3,}", answer, flags=_re.DOTALL)
    if _degenerate and _degenerate.start() > 30:
        answer = answer[:_degenerate.start()].rstrip()

    # Opschonen: het model sluit soms na het echte antwoord nog een structuur-gewoonte af
    # (een los blok-sluitteken, een markdown-scheiding, het begin van een nieuwe kop) zonder
    # dat daar nog inhoud bij komt. Knip elke losse staartregel weg die alleen uit
    # leestekens/markup bestaat (geen letters of cijfers) -- puur structureel, niet inhoudelijk.
    _tail_symbols = _re.search(r"\n\s*[^\w\s]{1,10}\s*$", answer)
    while _tail_symbols and _tail_symbols.start() > 30:
        answer = answer[:_tail_symbols.start()].rstrip()
        _tail_symbols = _re.search(r"\n\s*[^\w\s]{1,10}\s*$", answer)

    # Opschonen: soms komt er na een compleet, correct afgesloten antwoord (eindigend op
    # . ! of ?) nog een korte, LOSSTAANDE flard tekst op een nieuwe regel (een los woord, een
    # nep-functieaanroep, etc.) die zelf niet als een eigen zin afsluit -- knip die flard weg.
    _tail_fragment = _re.search(r"([.!?])\s*\n+\s*([^\n]{1,40})$", answer)
    if _tail_fragment and _tail_fragment.start(1) > 30 and not _re.search(r"[.!?]\s*$", _tail_fragment.group(2)):
        answer = answer[:_tail_fragment.start(1) + 1].rstrip()

    # Opschonen: soms is de losstaande staart-flard zelf WEL netjes afgesloten met een punt
    # (bv. een gemangelde verwijzing naar een eigen instructie-regel), waardoor de vorige check
    # hem mist. Als de allerlaatste regel heel kort is (<= 50 tekens) en op een eigen nieuwe
    # regel staat na een al-complete voorgaande zin, is dat vrijwel altijd zo'n losse flard --
    # knip die ook weg, zelfs als hij zelf wel eindigt op een leesteken.
    _tail_short_line = _re.search(r"([.!?])\s*\n+\s*([^\n]{1,50}[.!?]?)\s*$", answer)
    if _tail_short_line and _tail_short_line.start(1) > 30:
        answer = answer[:_tail_short_line.start(1) + 1].rstrip()

    results.append({"idx": IDX_LIST[i], "question": q, "document": doc, "extraction": extraction, "answer": answer})
    print(f"  {i+1}/{len(rows)} gegenereerd  ({time.time()-t0:.0f}s)", flush=True)

    # TUSSENTIJDS OPSLAAN na ELK item -- als het geld/de pod midden in de run opraakt,
    # staat alles tot dat punt al veilig op schijf (niet pas aan het einde).
    with open(f"/workspace/results_{LABEL}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print(f"=== OPGESLAGEN: /workspace/results_{LABEL}.json ({len(results)} items) ===", flush=True)

del model, base
torch.cuda.empty_cache()

if DEEPSEEK_KEY:
    print("=== Rechtfase (single DeepSeek judge, Engels-tegen-Engels, geen vertaling nodig) ===", flush=True)
    res = {"GROUNDED": 0, "UNGROUNDED": 0, "INELIGIBLE": 0}
    detail = []

    def process(r):
        ans = r["answer"]
        elig = judge_eligibility(r["question"], ans)
        if elig != "ELIGIBLE":
            return r["idx"], "INELIGIBLE"
        g = judge_grounding(r["document"], r["question"], ans)
        return r["idx"], g

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(process, r) for r in results]
        done = 0
        for fut in as_completed(futs):
            idx, v = fut.result()
            res[v] += 1
            detail.append({"idx": idx, "verdict": v})
            done += 1
            print(f"  geoordeeld {done}/{len(results)}  {res}  ({time.time()-t1:.0f}s)", flush=True)

    n = res["GROUNDED"] + res["UNGROUNDED"]
    total = sum(res.values())
    print(f"\n===== EXTRACT-COMPOSE {LABEL} (n={total}) =====")
    print(f"RAW: {res}")
    print(f"Gestrafte score (GROUNDED/totaal): {res['GROUNDED']*100/total:.1f}%")
    if n > 0:
        print(f"Onder eligible: gegrond {res['GROUNDED']*100/n:.1f}%  ongegrond {res['UNGROUNDED']*100/n:.1f}%")
    with open(f"/workspace/verdicts_{LABEL}.json", "w", encoding="utf-8") as f:
        json.dump({"res": res, "detail": detail}, f, ensure_ascii=False)

print("KLAAR_MET_ALLES", flush=True)
