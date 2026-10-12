# Dub quality scoring

The optional `POST /dub/qc/{job_id}` pass compares the generated text with a
second speech-to-text transcription. It reports drift, verification flags and
measured timing without replacing the original text.

Canonically equivalent Unicode spellings compare identically: a letter with a
separate combining accent matches its precomposed spelling. Real accent and
word differences still count, and compatibility ligatures are not folded.

This pass requires an installed speech-to-text model. Scoring-only tests with
supplied recognition segments do not measure synthesis or ASR accuracy.
