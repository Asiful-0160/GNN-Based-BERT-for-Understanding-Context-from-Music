# GNN–BERT project report

The report is compiled and layout-checked in IEEEtran conference format, with a title/abstract page, two-column main text, and a readable single-column appendix. It is nine pages including all front matter and appendices. The authors are Mohammed Ali Hossain (21301453), Jarin Akter Mou (20301070), and Asiful Kanzan Auishik (19101628), BRAC University. Two requested figure assets remain pending, so this is not yet a submission-ready version.

## Files

- `main.tex`: complete LaTeX source.
- `references.bib`: verified references and explicit project-source citations.
- `final_report.pdf`: compiled review copy.
- `figures/`: all 14 supplied PNG figures, preserved without modification; the report selects seven source images for five combined/individual figures.
- `source_verification.json`: inventory of all 44 supplied files, hashes, local manifest comparisons, and source-reference checks.
- `verify_sources.py`: verifies sources without changing them and writes `source_verification.json`.

## Still needed

1. Course confirmed: CSE715. Moin Mostakim is credited only as the assignment-brief preparer, not assumed to be a report author or instructor.
2. Confirm any change from the defaults: IEEE Conference, 6–10 pages, condensed audit appendix. Provide a repository/demo URL if it should appear.
3. `figures/fig_tsne_task3.png`: the actual Task-3 embedding visualization, with its color-label and generation context. It must come from frozen embeddings; no coordinates have been fabricated.
4. `figures/fig_task4_qualitative_top3.png`: the actual ten fixed queries and seed-42 top-three retrieval panel. The underlying Script-74 CSV/JSON is preferable if the panel does not already exist. The supplied questionnaire gives captions but not the retrieved identities or ranking scores.
5. If available, final configuration/source records establishing optimizer settings, Phase-B trainable-layer scope, GPU/environment details, and Git commit would close documented reproducibility gaps. They are not inferred from generic defaults.

The ten questionnaire WAV files have now been supplied. A complete listening supplement is available in the project-level `human_evaluation/` directory and `Human_Evaluation_Supplement.zip`. Open that supplement's `index.html` to replay the clips. The original RESULTS export is preserved; use the supplement copy for working audio paths. The PDF documents the study and saved results without embedding audio.

Missing figures have conditional `includegraphics` calls, so the current review copy compiles with explicit textual notices. No substitute or fabricated image is generated. Captions should be finalized after the real missing figures are inspected.

## Compile

From this directory, use `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice. Overleaf: upload `main.tex`, `references.bib`, and `figures/`, select pdfLaTeX, and set `main.tex` as the main document. Dependencies: IEEEtran class/bibliography style; fontenc, inputenc, amsmath, amssymb, booktabs, graphicx, array, url, hyperref (standard TeX Live/MiKTeX packages).

## Evidence and interpretation

Sources read: `Project_Guideline.pdf`; the Task-1–4 independent audit exported on 12 September 2026 at 16:02; the master prompt; the RESULTS JSON/CSV export; and `index.html`. Source files are unchanged. Local verification finds no hash mismatches among the checked manifest entries and no unresolved LaTeX source references. Bibliographic claims were checked against ACL Anthology, NeurIPS proceedings, arXiv, PMLR, and Google's MusicCaps dataset card.

The report preserves the fusion negative result, caption/aspect circularity, qualified prototype readout, dataset deviation, partial source audit, and other audited caveats. Supervised uncertainty uses sample seed SD; Task-4 uncertainty uses population seed SD. Original plots retain legacy AP naming, clarified in the report captions. No model was retrained, no inference was run, and no experimental metric was retuned or recomputed.
