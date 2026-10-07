from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from environment variables (.env via docker compose).
    Every tuning knob has a sensible default, so .env only needs the secrets."""

    model_config = SettingsConfigDict(extra="ignore")

    # --- Providers: "gemini" (cloud) or "local" (llama.cpp containers) ---
    llm_provider: str = "gemini"
    embed_provider: str = "gemini"
    embed_dim: int | None = None                # None -> provider default (gemini 768, local 1024)

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-3.8-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    gemini_thinking_level: str = "low"          # low | medium | high (3.8 Flash rejects "minimal")

    # --- Local (llama.cpp server, OpenAI-compatible API) ---
    local_llm_url: str = "http://llm:8080"
    local_embed_url: str = "http://embed:8080"
    local_llm_file: str = "Qwen3.5-9B-UD-Q4_K_XL.gguf"
    local_embed_file: str = "Qwen3-Embedding-0.6B-Q8_0.gguf"
    local_temperature: float = 0.7              # Qwen3.5/3.6 non-thinking recommendation
    local_top_p: float = 0.8
    local_top_k: int = 20
    local_max_tokens: int = 8192
    local_enable_thinking: bool = False         # structured JSON tasks work best without thinking
    local_request_timeout_s: int = 900          # CPU-offloaded MoE can be slow on long outputs
    local_ready_timeout_s: int = 1800           # first start: download + load can take a while

    # --- Rate limiting (Gemini free tier friendly; local only uses llm_concurrency) ---
    llm_concurrency: int = 2                    # parallel generate calls (local: = llama.cpp slots)
    llm_rpm: int = 10                           # max generate requests / minute (gemini only)
    embed_rpm: int = 60                         # max embed requests / minute (gemini only)
    embed_batch_size: int = 100                 # texts per embed request (local uses min(this, 32))
    llm_max_retries: int = 6

    # --- Neo4j ---
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # --- Ingestion ---
    upload_dir: str = "/data/uploads"
    static_dir: str = "/app/static"
    max_upload_mb: int = 50
    chunk_size_chars: int = 4000                # ~1000 tokens
    chunk_overlap_chars: int = 400

    # Entity resolution (ANN candidates -> Gemini confirms)
    er_top_k: int = 5                           # neighbours fetched per new entity
    er_cosine_threshold: float | None = None    # candidate-pair cosine; None -> gemini 0.85, local 0.70
    er_max_group_size: int = 8
    er_groups_per_call: int = 15

    # Community detection + summaries
    leiden_max_levels: int = 10
    leiden_gamma: float = 1.0
    leiden_seed: int = 42
    summary_max_input_chars: int = 12000

    # Retrieval
    retrieval_threshold: float | None = None    # query vs summary cosine; None -> gemini 0.55, local 0.50
    retrieval_top_k: int = 6                    # max top-level communities to start from
    retrieval_min_top: int = 2                  # fallback: if nothing passes the threshold, start from the best N
    retrieval_fallback_margin: float = 0.05     # ...but only those within this margin of the best score
    retrieval_max_visits: int = 40              # BFS safety cap
    context_max_chars: int = 30000              # context budget (~7.5k tokens)
    leaf_max_entities: int = 8
    leaf_max_relationships: int = 12
    max_source_chunks: int = 6
    source_excerpt_chars: int = 1500

    @property
    def effective_embed_dim(self) -> int:
        if self.embed_dim:
            return self.embed_dim
        return 1024 if self.embed_provider == "local" else 768

    @property
    def embed_model_id(self) -> str:
        """Identifies the vector space: graphs built with a different id are incompatible."""
        model = self.local_embed_file if self.embed_provider == "local" else self.gemini_embed_model
        return f"{self.embed_provider}:{model}:{self.effective_embed_dim}"

    @property
    def llm_model_name(self) -> str:
        return self.local_llm_file if self.llm_provider == "local" else self.gemini_llm_model

    @property
    def gemini_key_set(self) -> bool:
        return bool(self.gemini_api_key) and self.gemini_api_key != "PUT_YOUR_KEY_HERE"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for name in ("llm_provider", "embed_provider"):
        v = getattr(s, name).strip().lower()
        if v not in ("gemini", "local"):
            raise ValueError(f"{name.upper()} must be 'gemini' or 'local', got {v!r}")
        setattr(s, name, v)
    # similarity scales differ per embedding model -> provider-specific defaults (measured, see README)
    local = s.embed_provider == "local"
    if s.er_cosine_threshold is None:
        s.er_cosine_threshold = 0.70 if local else 0.85
    if s.retrieval_threshold is None:
        s.retrieval_threshold = 0.50 if local else 0.55
    return s
