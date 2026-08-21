# Speech test set

48 stratified Bahasa Rojak utterances x 2 speakers = 96 clips.
Full method and rationale: `docs/whisper_step_by_step.md`.

## Recording

1. Read the `reference` column aloud at normal pace. Do NOT over-enunciate --
   you are measuring performance on realistic input.
2. Phone voice memo is fine. Quiet room.
3. Convert to the format Whisper wants (16 kHz mono WAV):

   ffmpeg -i raw/s01.m4a -ar 16000 -ac 1 eval/speech/audio/s01_ms_dom_yv.wav

4. Track progress in `recording_tracker.xlsx` (yellow cells only). `manifest.csv` holds
   the data and is what the eval script reads -- never edit sentences in Excel.

References were written BEFORE recording, deliberately. Transcribing your own audio
afterwards biases the reference toward whatever you happened to say.

## Stratification

| switch_type | n | tests |
|---|---|---|
| ms_dom    | 8 | Malay matrix, English insertion |
| en_dom    | 8 | English matrix, Malay particles |
| balanced  | 8 | true intra-sentential switching |
| particles | 8 | discourse particles (lah/lor/meh/kan/kot) |
| numeric   | 8 | prices, percentages, dates |
| entity    | 8 | Malaysian named entities |

Proposal S5.6 predicts `balanced` and `particles` score worst, because Whisper
commits to one language per segment. Confirming that with your own data is the
finding -- not a failure.

## Git

Commit `manifest.csv`, `prompt_examples.txt`, `results.csv`.
Do NOT commit `audio/` (see .gitignore). Keep audio in shared Drive.
