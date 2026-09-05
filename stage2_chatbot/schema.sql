-- Feedback log for Stage 4. See app/feedback.py.
--
-- This file is committed; feedback.db is not.
-- Recreate an empty database with:  python app/feedback.py --init

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT    NOT NULL,

    -- Kept apart on purpose. transcript is what Whisper heard and is NULL when
    -- the question was typed; user_text is what the model was actually given,
    -- after app/normalise.py. A wrong answer to a misheard question is an ASR
    -- failure, not an NLP failure, and the evaluation reports the two
    -- separately. Storing one column would lose that.
    transcript  TEXT,
    user_text   TEXT    NOT NULL,

    reply       TEXT    NOT NULL,
    verdict     TEXT    NOT NULL CHECK (verdict IN ('up', 'down')),

    -- The expected answer, supplied by the user on thumbs-down. This is the
    -- column that becomes a training pair in the next round.
    correction  TEXT,

    -- JSON array of the turns that preceded this one. The reply depended on
    -- them, so a correction cannot be judged without them.
    context     TEXT,

    -- Which path produced the reply, for reading latency complaints correctly.
    backend     TEXT,
    seconds     REAL
);

-- The review workflow is "show me what is still unread", so index the filter.
CREATE INDEX IF NOT EXISTS idx_feedback_verdict ON feedback (verdict, created_utc);