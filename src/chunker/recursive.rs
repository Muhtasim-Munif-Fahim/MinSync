use crate::chunker::Chunker;
use crate::error::Result;
use crate::types::Chunk;
use chunk::{merge_splits, IncludeDelim, PatternSplitter};

pub struct RecursiveChunker {
    splitter: PatternSplitter,
    max_chunk_size: usize,
    schema_id: String,
}

impl RecursiveChunker {
    pub fn new(max_chunk_size: usize) -> Self {
        let patterns: &[&[u8]] = &[b"\n\n", b". ", b"? ", b"! ", b"\n"];

        Self {
            splitter: PatternSplitter::new(patterns),
            max_chunk_size,
            schema_id: "recursive".to_string(),
        }
    }

    pub fn from_config(options: &crate::config::ChunkerOptions) -> Self {
        Self::new(options.max_chunk_size)
    }
}

impl Chunker for RecursiveChunker {
    fn schema_id(&self) -> &str {
        &self.schema_id
    }

    fn chunk(&self, text: &str, _path: &str) -> Result<Vec<Chunk>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let bytes = text.as_bytes();
        let offsets = keep_table_rows_whole(
            text,
            self.splitter.split(bytes, IncludeDelim::Prev, 0),
            self.max_chunk_size,
        );
        let splits: Vec<&str> = offsets
            .into_iter()
            .filter_map(|(start, end)| std::str::from_utf8(&bytes[start..end]).ok())
            .collect();

        if splits.is_empty() {
            return Ok(vec![Chunk {
                text: text.to_string(),
                chunk_type: "chunk".to_string(),
                heading_path: String::new(),
            }]);
        }

        let token_counts: Vec<usize> = splits.iter().map(|split| split.chars().count()).collect();
        let merged = merge_splits(&splits, &token_counts, self.max_chunk_size).merged;

        let result = merged
            .into_iter()
            .map(|text| Chunk {
                text,
                chunk_type: "chunk".to_string(),
                heading_path: String::new(),
            })
            .filter(|c| !c.text.trim().is_empty())
            .collect();

        Ok(result)
    }
}

/// A markdown table row is a line whose first non-whitespace character is `|`.
fn is_table_row(line: &str) -> bool {
    line.trim_start().starts_with('|')
}

/// Join split ranges that cut through the inside of a markdown table row so the
/// row stays in a single piece. A row is only joined when the resulting piece
/// (including the delimiter attached to its end) fits in `max_chunk_size`;
/// larger rows keep falling back to sentence splitting, so this rule never
/// pushes a chunk past the embedder's context.
///
/// `offsets` must be sorted, contiguous byte ranges covering `text`.
fn keep_table_rows_whole(
    text: &str,
    offsets: Vec<(usize, usize)>,
    max_chunk_size: usize,
) -> Vec<(usize, usize)> {
    // join_prev[i]: piece i is a continuation of piece i - 1.
    let mut join_prev = vec![false; offsets.len()];
    let mut any_join = false;
    let mut line_start = 0;
    for line in text.split_inclusive('\n') {
        let row_start = line_start;
        line_start += line.len();
        if !is_table_row(line) {
            continue;
        }
        let row_end = row_start + line.trim_end_matches(['\n', '\r']).len();
        // First piece ending inside/after the row start, last piece starting before the row end.
        let first = offsets.partition_point(|&(_, end)| end <= row_start);
        let after = offsets.partition_point(|&(start, _)| start < row_end);
        if first >= offsets.len() || after <= first + 1 {
            continue;
        }
        let last = after - 1;
        let (span_start, span_end) = (offsets[first].0, offsets[last].1);
        if text[span_start..span_end].chars().count() <= max_chunk_size {
            join_prev[first + 1..=last].fill(true);
            any_join = true;
        }
    }
    if !any_join {
        return offsets;
    }

    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(offsets.len());
    for (piece, joined) in offsets.into_iter().zip(join_prev) {
        match merged.last_mut() {
            Some(last) if joined => last.1 = piece.1,
            _ => merged.push(piece),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        let chunker = RecursiveChunker::new(100);

        let chunks = chunker.chunk("", "f.md").expect("chunk empty text");

        assert!(chunks.is_empty());
    }

    #[test]
    fn test_single_short_line() {
        let chunker = RecursiveChunker::new(100);

        let chunks = chunker
            .chunk("Hello world", "f.md")
            .expect("chunk single line");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world");
    }

    #[test]
    fn test_multi_paragraph() {
        let chunker = RecursiveChunker::new(18);
        let text = "First paragraph\n\nSecond paragraph\n\nThird paragraph";

        let chunks = chunker.chunk(text, "f.md").expect("chunk paragraphs");

        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), text);
        assert!(chunks.iter().any(|chunk| chunk.text.ends_with("\n\n")));
    }

    #[test]
    fn test_merge_respects_max_chunk_size() {
        let max_chunk_size = 8;
        let chunker = RecursiveChunker::new(max_chunk_size);
        let text = "a. bb. ccc. dddd. unsplittablelongword";

        let chunks = chunker.chunk(text, "f.md").expect("chunk with max size");

        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), text);
        for chunk in chunks {
            let contains_boundary = chunk.text.matches(". ").count() > 1;
            if contains_boundary {
                assert!(chunk.text.chars().count() <= max_chunk_size);
            }
        }
    }

    #[test]
    fn test_schema_id() {
        let chunker = RecursiveChunker::new(100);

        assert_eq!(chunker.schema_id(), "recursive");
    }

    #[test]
    fn test_whitespace_only_filtered() {
        let chunker = RecursiveChunker::new(10);
        let text = "  \n\nreal\n  ";

        let chunks = chunker.chunk(text, "f.md").expect("chunk whitespace");

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text.trim(), "real");
    }

    #[test]
    fn test_from_config() {
        let options = crate::config::ChunkerOptions {
            max_chunk_size: 8,
            delimiters: "\n".to_string(),
        };
        let chunker = RecursiveChunker::from_config(&options);

        let chunks = chunker
            .chunk("alpha\nbeta\ngamma", "f.md")
            .expect("chunk from config");

        assert_eq!(chunker.schema_id(), "recursive");
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), "alpha\nbeta\ngamma");
    }

    fn table_row_with_sentences(key: &str, value: &str) -> String {
        let sentences: Vec<String> = (0..12).map(|i| format!("Sentence {i} here.")).collect();
        format!("| {key} | desc | {} | {value} |", sentences.join(" "))
    }

    fn filler_rows(count: usize) -> String {
        (0..count)
            .map(|i| format!("| F{i:03} | routine | one short note | 0 |\n"))
            .collect()
    }

    #[test]
    fn test_table_row_with_sentence_punctuation_is_never_split() {
        // Issue #68: the max_chunk_size boundary fell inside a row that contains ". ".
        let row = table_row_with_sentences("ZROWID9", "ZVAL9");
        let header = "| id | kind | description | amount |\n| --- | --- | --- | --- |\n";
        let text = format!("{header}{}{row}\n{}", filler_rows(8), filler_rows(8));
        let chunker = RecursiveChunker::new(500);

        let chunks = chunker.chunk(&text, "t.md").expect("chunk table");

        assert!(
            row.chars().count() < 500,
            "fixture row must fit in one chunk"
        );
        assert!(
            chunks.iter().any(|chunk| chunk.text.contains(&row)),
            "no chunk contains the complete row: {chunks:#?}"
        );
        assert_eq!(chunks.concat_text(), text);
    }

    #[test]
    fn test_every_table_row_is_whole_at_any_boundary_position() {
        // Slide the row across every possible offset relative to the size boundary.
        let row = table_row_with_sentences("ZROWID9", "ZVAL9");
        for filler in 0..40 {
            let text = format!("{}{row}\n{}", filler_rows(filler), filler_rows(8));
            let chunker = RecursiveChunker::new(500);

            let chunks = chunker.chunk(&text, "t.md").expect("chunk table");

            assert!(
                chunks.iter().any(|chunk| chunk.text.contains(&row)),
                "filler={filler}: row split across chunks: {chunks:#?}"
            );
            assert_eq!(chunks.concat_text(), text);
        }
    }

    #[test]
    fn test_table_row_without_trailing_newline_stays_whole() {
        let row = table_row_with_sentences("ZROWID9", "ZVAL9");
        let text = format!("{}{row}", filler_rows(10));
        let chunker = RecursiveChunker::new(300);

        let chunks = chunker.chunk(&text, "t.md").expect("chunk table");

        assert!(chunks.iter().any(|chunk| chunk.text.contains(&row)));
        assert_eq!(chunks.concat_text(), text);
    }

    #[test]
    fn test_protecting_a_row_never_pushes_a_chunk_past_max_chunk_size() {
        // Row content is exactly max_chunk_size, so with its trailing newline it
        // no longer fits in one chunk and must stay splittable.
        let max = 60;
        let body: String = "ab. ".repeat(max).chars().take(max - 4).collect();
        let row = format!("| {body} |");
        assert_eq!(row.chars().count(), max);
        let text = format!("{row}\n{row}\n{row}\n");
        let chunker = RecursiveChunker::new(max);

        let chunks = chunker.chunk(&text, "t.md").expect("chunk table");

        for chunk in &chunks {
            assert!(
                chunk.text.chars().count() <= max,
                "chunk exceeds max_chunk_size: {chunk:?}"
            );
        }
        assert_eq!(chunks.concat_text(), text);
    }

    #[test]
    fn test_row_followed_by_blank_line_never_exceeds_max_chunk_size() {
        // The "\n\n" delimiter is attached to the row's piece, so row + "\n\n"
        // is what has to fit, not just the row line.
        let max = 60;
        let body: String = "ab. ".repeat(max).chars().take(max - 5).collect();
        let row = format!("| {body} |");
        assert_eq!(row.chars().count(), max - 1);
        let text = format!("{row}\n\nnext paragraph\n\n{row}\n\n");
        let chunker = RecursiveChunker::new(max);

        let chunks = chunker.chunk(&text, "t.md").expect("chunk table");

        for chunk in &chunks {
            assert!(
                chunk.text.chars().count() <= max,
                "chunk exceeds max_chunk_size: {chunk:?}"
            );
        }
        assert_eq!(chunks.concat_text(), text);
    }

    #[test]
    fn test_row_longer_than_max_chunk_size_still_falls_back_to_sentence_split() {
        let row = table_row_with_sentences("ZROWID9", "ZVAL9");
        let text = format!("{}{row}\n{}", filler_rows(3), filler_rows(3));
        let chunker = RecursiveChunker::new(100);

        let chunks = chunker.chunk(&text, "t.md").expect("chunk table");

        assert!(!chunks.iter().any(|chunk| chunk.text.contains(&row)));
        assert_eq!(chunks.concat_text(), text);
    }

    #[test]
    fn test_prose_sentences_still_split_at_sentence_boundaries() {
        let text = "Alpha one. Beta two. Gamma three. Delta four.";
        let chunker = RecursiveChunker::new(22);

        let chunks = chunker.chunk(text, "f.md").expect("chunk prose");

        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat_text(), text);
    }

    trait ChunkTextExt {
        fn concat_text(&self) -> String;
    }

    impl ChunkTextExt for [Chunk] {
        fn concat_text(&self) -> String {
            self.iter().map(|chunk| chunk.text.as_str()).collect()
        }
    }
}
