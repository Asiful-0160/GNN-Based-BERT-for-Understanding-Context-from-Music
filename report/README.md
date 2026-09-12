# GNN–BERT project report

The report is compiled and layout-checked in IEEEtran conference format, with a title/abstract page, two-column main text, and a readable single-column appendix. It is nine pages including all front matter and appendices. The authors are Mohammed Ali Hossain (21301453), Jarin Akter Mou (20301070), and Asiful Kanzan Auishik (19101628), BRAC University.

## Files

- `main.tex`: complete LaTeX source.
- `references.bib`: verified references and explicit project-source citations.
- `final_report.pdf`: compiled review copy.
- `figures/`: all 14 supplied PNG figures, preserved without modification; the report selects seven source images for five combined/individual figures.
- `source_verification.json`: inventory of all 44 supplied files, hashes, local manifest comparisons, and source-reference checks.
- `verify_sources.py`: verifies sources without changing them and writes `source_verification.json`.

## Scope

Course: CSE715. The t-SNE figure and top-three retrieval examples are excluded from the final report at the authors' request. Their placeholders and pending-asset notices have been removed. Scientific qualifications concerning the reported results remain intact.

The ten questionnaire WAV files have now been supplied. A complete listening supplement is available in the project-level `human_evaluation/` directory and `Human_Evaluation_Supplement.zip`. Open that supplement's `index.html` to replay the clips. The original RESULTS export is preserved; use the supplement copy for working audio paths. The PDF documents the study and saved results without embedding audio.

## Compile

From this directory, use `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice. Overleaf: upload `main.tex`, `references.bib`, and `figures/`, select pdfLaTeX, and set `main.tex` as the main document. Dependencies: IEEEtran class/bibliography style; fontenc, inputenc, amsmath, amssymb, booktabs, graphicx, array, url, hyperref (standard TeX Live/MiKTeX packages).

## Evidence and interpretation

Sources read: `Project_Guideline.pdf`; the Task-1–4 independent audit exported on 12 September 2026 at 16:02; the master prompt; the RESULTS JSON/CSV export; and `index.html`. Source files are unchanged. Local verification finds no hash mismatches among the checked manifest entries and no unresolved LaTeX source references. Bibliographic claims were checked against ACL Anthology, NeurIPS proceedings, arXiv, PMLR, and Google's MusicCaps dataset card.

The report preserves the fusion negative result, caption/aspect circularity, qualified prototype readout, dataset deviation, partial source audit, and other audited caveats. Supervised uncertainty uses sample seed SD; Task-4 uncertainty uses population seed SD. Original plots retain legacy AP naming, clarified in the report captions. No model was retrained, no inference was run, and no experimental metric was retuned or recomputed.
