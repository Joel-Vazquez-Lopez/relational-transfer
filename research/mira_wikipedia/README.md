# Mira Wikipedia article-revision feasibility pilot — 16 October 2026

**Scientific status:** **SEVEN REVISION LOCATORS**, NOT seven vetted passages, verified event pairs, or independently human-annotated semantic-role labels. **No pretrained inference**. This is **OPEN DEVELOPMENT** with **high memorization risk**, not an untouched benchmark.

## Exact reproducible archive
- User-authorized ChatGPT File Library: `/Research Team Handoffs/MIRA_WIKIPEDIA_REVISION_PILOT_2026-10-16.zip`
- ZIP SHA-256: `416553219a629e2d0fb965440f326f1ee6ccd947ac6faf38470e34ca86783b62` (8,928 bytes). ZIP CRC/test passed in Mira workspace.
- Contains `candidate_articles.jsonl`, an exact-oldid Python source fetch/validation script, four CSV review templates, independent fixture tests, science gates and individual file hash manifest. Script's offline self-test **passed** (positive accepted and four negative controls rejected). An actual MediaWiki API fetch for all seven revisions **has NOT run** in this environment.
- The repo manifest records seven permalink oldid numbers **observed through each Wikipedia page's “Permanent link” control**. Only historical Marie Curie oldid `1379969707` rendered directly in the web viewer; the other six permanent revisions could not be retrieved for exact-text verification. Therefore the `source_exact_wikitext_sha256` and `source_exact_paragraph_anchor` fields are all `null` and MUST remain so until genuine fetch/readback.
- Page IDs are only included when separately verified from Wikipedia page information (Ada `974`, Marie Curie `20408`); other page IDs remain null. No guessed IDs, source content, snippets, annotation gold or privileged metadata are fed to any model.

## Why Wikipedia, and what remains unsafe
Most Wikipedia article text uses CC BY-SA 4.0, but **individual imported excerpts, quotations, images and exceptions may have different rights**. Reuse requires appropriate license/attribution to contributors (usually revision history), applicable ShareAlike and modification notices. See primary sources:
- https://foundation.wikimedia.org/wiki/Terms_of_Use
- https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse
- https://creativecommons.org/licenses/by-sa/4.0/

The seven candidates are not automatically cleared on an individual paragraph basis. The included `SOURCE_RIGHTS_BLANK.csv` requires page/revision and excerpt review. The candidate set is biased toward famous historical people/events and may have been memorized in pretrained models; DO NOT claim unseen-text generalization.

## Verification workflow
1. Retrieve exact immutable revision via MediaWiki API, checking `pageid/title/revid/timestamp/sha1`. The ZIP script `fetch_verify.py` provides the workflow, saves source wikitext **locally only**, and computes a SHA-256 of raw UTF-8 wikitext; user must supply a real identifying/contactable User-Agent when executing.
2. Have a reviewer select an explicit paragraph range and SHA; do not treat `section_hint` or `verb_search_hints` as verified passages. Screen imported quotes and privacy.
3. Record revision contributor history link, license link, allowed processing/redistribution and changes (even if none).
4. Obtain **two truly independent human semantic annotations**, preserve non-applicable/implicit arguments and disagreement; adjudicate separately.
5. Only then consider Nova's preregistered **two-different-pretrained-family** format/competence test and Comput's lexical/position adversaries. Astra retains field-order experimental ownership. No duplicate inference authorized.

## Relation to EWT
EWT [PR #26](https://github.com/Joel-Vazquez-Lopez/relational-transfer/pull/26) and suitability [PR #27](https://github.com/Joel-Vazquez-Lopez/relational-transfer/pull/27) remain development with 49 source-document original-text rights statuses unresolved; this branch neither removes nor supersedes their negative findings.

**Immediate reviewers:** coordinator for CC BY-SA/revision and individual excerpt approval; Comput for candidate construction and source-only shortcut attacks; Nova for source/prompt separation. **No EUREKA or mechanism claim.**
