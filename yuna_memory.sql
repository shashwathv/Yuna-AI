CREATE TABLE conversations (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id BIGINT,
    role TEXT NOT NULL CHECK(role IN ('yuna', 'user')),
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conversations_user_time 
ON conversations(user_id, created_at DESC);

CREATE INDEX idx_conversations_session_time 
ON conversations(user_id, session_id DESC);