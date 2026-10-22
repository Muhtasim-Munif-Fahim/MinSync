//! Client-side context-length guard for remote embedders (`openai:`, `tei:`).
//!
//! The native embedder truncates at `max_length` with the model's own
//! tokenizer. A remote embedder has none here, so a chunk that is longer than
//! the model's context either fails with a 400 or is cut by the provider
//! without being counted. This module cuts such inputs below the limit before
//! they are sent, using an **upper bound** on the token count.

use std::borrow::Cow;

/// Share of the context an input may use. The estimate is already an upper
/// bound; this adds room for special tokens, prefixes and tokenizers that
/// split more finely than the ones the weights were measured on.
const BUDGET_NUMERATOR: usize = 9;
const BUDGET_DENOMINATOR: usize = 10;

/// Upper-bound weight of one char, in half tokens.
///
/// Worst single-passage tokens per char, measured on MIRACL passages in 18
/// languages with the tokenizers of the models in [`KNOWN_CONTEXT`] and of
/// common remote ones:
///
/// | chars        | Qwen3 | XLM-R / bge-m3 | OpenAI cl100k | BERT (gte, bge-en) |
/// |--------------|-------|----------------|---------------|--------------------|
/// | ASCII        | 0.34  | 0.34           | 0.32          | 0.32               |
/// | 2-byte (ar, ru, fa) | 0.69 | 0.50      | 0.83          | 0.90               |
/// | 3-byte (ja, zh, ko, hi, bn, te, th) | 1.51 | 1.14 | 1.78 | 2.06 |
///
/// ASCII is weighted 0.5 (natural-language text; a base64 or hex blob can
/// exceed it), 2-byte chars 1, and 3-byte chars 2, so the estimate stays an
/// upper bound for those tokenizers with the 90% budget applied. The price is
/// that CJK text on a 512-token model is cut well before the real limit.
fn weight(ch: char) -> usize {
    match ch as u32 {
        0..=0x7f => 1,
        0x80..=0x7ff => 2,
        _ => 4,
    }
}

/// Upper bound on the tokens of `text`.
pub fn estimate_tokens_upper(text: &str) -> usize {
    text.chars().map(weight).sum::<usize>().div_ceil(2)
}

/// Tokens an input may use for a model whose context is `max_length`.
pub fn budget(max_length: usize) -> usize {
    (max_length * BUDGET_NUMERATOR / BUDGET_DENOMINATOR).max(1)
}

/// `text` cut at a char boundary so that its estimate fits `budget` tokens.
pub fn truncate_to_budget(text: &str, budget: usize) -> Cow<'_, str> {
    if estimate_tokens_upper(text) <= budget {
        return Cow::Borrowed(text);
    }
    let limit = budget.saturating_mul(2);
    let mut used = 0;
    for (index, ch) in text.char_indices() {
        used += weight(ch);
        if used > limit {
            return Cow::Borrowed(&text[..index]);
        }
    }
    Cow::Borrowed(text)
}

/// Documented context length (tokens) of well-known embedding models, matched
/// on the model id without its provider scheme. The longest key that the id
/// contains wins. An unknown model has no entry and is never cut.
///
/// Values are the models' published context lengths (the lower figure where
/// two sources differ): OpenRouter's embeddings catalog for the hosted models
/// and the Hugging Face `config.json` for Qwen3-Embedding.
const KNOWN_CONTEXT: &[(&str, usize)] = &[
    ("text-embedding-3-small", 8191),
    ("text-embedding-3-large", 8191),
    ("text-embedding-ada-002", 8191),
    ("qwen3-embedding", 32768),
    ("pplx-embed", 32000),
    ("voyage-4", 32000),
    ("mistral-embed", 8192),
    ("bge-m3", 8192),
    ("multilingual-e5", 512),
    ("e5-large", 512),
    ("e5-base", 512),
    ("e5-small", 512),
    ("gte-large", 512),
    ("gte-base", 512),
    ("bge-large-en", 512),
    ("bge-base-en", 512),
    ("bge-small-en", 512),
];

/// Built-in context length for `model_id` (`openai:` / `tei:` prefix allowed).
pub fn default_max_length(model_id: &str) -> Option<usize> {
    let id = model_id
        .split_once(':')
        .map_or(model_id, |(_, rest)| rest)
        .to_ascii_lowercase();
    KNOWN_CONTEXT
        .iter()
        .filter(|(key, _)| id.contains(key))
        .max_by_key(|(key, _)| key.len())
        .map(|(_, length)| *length)
}

/// The context guard of one remote embedder: the model's `max_length` (explicit
/// or from the table) and the cut applied before each request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InputLimit {
    max_length: Option<usize>,
}

impl InputLimit {
    /// Limit for `model_id`: an explicit `max_length` wins over the table.
    pub fn for_model(model_id: &str, explicit: Option<usize>) -> Self {
        Self {
            max_length: explicit.or_else(|| default_max_length(model_id)),
        }
    }

    pub fn max_length(&self) -> Option<usize> {
        self.max_length
    }

    /// `text` (as it will be sent, prefix included) cut to the budget, or
    /// untouched when there is no limit or it already fits.
    pub fn apply<'a>(&self, text: &'a str) -> Cow<'a, str> {
        match self.max_length {
            Some(max_length) => truncate_to_budget(text, budget(max_length)),
            None => Cow::Borrowed(text),
        }
    }

    /// Whether `text` (as it will be sent) has to be cut.
    pub fn needs_cut(&self, text: &str) -> bool {
        self.max_length
            .is_some_and(|max_length| estimate_tokens_upper(text) > budget(max_length))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_models_resolve_with_or_without_the_provider_scheme() {
        assert_eq!(
            default_max_length("openai:perplexity/pplx-embed-v1-0.6b"),
            Some(32000)
        );
        assert_eq!(default_max_length("qwen/qwen3-embedding-4b"), Some(32768));
        assert_eq!(
            default_max_length("tei:intfloat/multilingual-e5-large"),
            Some(512)
        );
        assert_eq!(
            default_max_length("openai:text-embedding-3-small"),
            Some(8191)
        );
        assert_eq!(default_max_length("BAAI/BGE-M3"), Some(8192));
    }

    #[test]
    fn unknown_models_have_no_default() {
        assert_eq!(default_max_length("openai:acme/mystery-embed-v9"), None);
        assert_eq!(default_max_length(""), None);
    }

    #[test]
    fn the_longest_matching_key_wins() {
        // `bge-m3` (8192) and `e5-large` (512) both match; the longer key,
        // `e5-large`, decides.
        assert_eq!(default_max_length("acme/bge-m3-e5-large"), Some(512));
        assert_eq!(
            default_max_length("intfloat/multilingual-e5-large"),
            Some(512)
        );
    }

    #[test]
    fn text_within_the_budget_is_returned_untouched() {
        let text = "hello world";
        assert!(matches!(truncate_to_budget(text, 100), Cow::Borrowed(t) if t == text));
    }

    #[test]
    fn long_text_is_cut_to_a_prefix_inside_the_budget() {
        let text = "a".repeat(10_000);
        let cut = truncate_to_budget(&text, 100);
        assert!(text.starts_with(cut.as_ref()));
        assert!(
            estimate_tokens_upper(&cut) <= 100,
            "{}",
            estimate_tokens_upper(&cut)
        );
        assert!(
            cut.len() > 150,
            "it must keep what fits, not cut to nothing: {}",
            cut.len()
        );
    }

    #[test]
    fn the_cut_never_splits_a_multibyte_char() {
        let text = "한".repeat(5_000);
        let cut = truncate_to_budget(&text, 101);
        assert!(text.starts_with(cut.as_ref()));
        assert!(estimate_tokens_upper(&cut) <= 101);
        // 4 half-tokens per char: 101 tokens fit 50 chars.
        assert_eq!(cut.chars().count(), 50);
    }

    #[test]
    fn the_estimate_is_an_upper_bound_on_the_measured_worst_ratios() {
        // Worst single-passage tokens per char (see `weight`), over every
        // tokenizer measured; the budget keeps 10% in reserve.
        let samples: [(&str, char, f64); 9] = [
            ("en", 'e', 0.34),
            ("ru (BERT)", 'д', 0.88),
            ("fa (cl100k)", 'ب', 0.83),
            ("ko (Qwen3)", '한', 1.0),
            ("bn (cl100k)", 'ক', 1.32),
            ("te (Qwen3)", 'క', 1.51),
            ("te (cl100k)", 'క', 1.78),
            ("ja (cl100k)", 'あ', 1.54),
            ("ko (BERT)", '한', 2.06),
        ];
        for (name, ch, worst) in samples {
            let text: String = std::iter::repeat_n(ch, 1000).collect();
            let allowed_real = estimate_tokens_upper(&text) as f64 / 0.9;
            assert!(
                allowed_real >= worst * 1000.0,
                "{name}: budget-adjusted estimate {allowed_real} < real {}",
                worst * 1000.0
            );
        }
    }

    #[test]
    fn a_default_chunk_is_never_cut_on_a_32k_context_model() {
        // max_chunk_size defaults to 4096 chars: even all-Telugu fits 32000.
        let text: String = std::iter::repeat_n('క', 4096).collect();
        assert!(estimate_tokens_upper(&text) <= budget(32_000));
    }

    #[test]
    fn a_512_token_model_cuts_cjk_to_about_a_quarter_of_a_default_chunk() {
        let text: String = std::iter::repeat_n('한', 4096).collect();
        let cut = truncate_to_budget(&text, budget(512));
        assert_eq!(cut.chars().count(), 230);
    }
}
