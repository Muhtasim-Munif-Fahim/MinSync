use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "minsync", version, about = "Git-free incremental vector index")]
pub struct Cli {
    #[arg(long, short, global = true)]
    pub verbose: bool,

    #[arg(long, short, global = true)]
    pub quiet: bool,

    #[arg(long, global = true, default_value = "text")]
    pub format: OutputFormat,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Clone, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

/// Machine-readable progress stream format for `sync --progress`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum ProgressFormat {
    /// One JSON object per line.
    Jsonl,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum QueryMode {
    Vector,
    Bm25,
    Hybrid,
}

impl QueryMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vector => "vector",
            Self::Bm25 => "bm25",
            Self::Hybrid => "hybrid",
        }
    }
}

#[derive(Subcommand)]
pub enum Commands {
    /// Initialize .minsync/ in current directory
    Init {
        #[arg(long)]
        force: bool,
        #[arg(long, default_value = crate::config::DEFAULT_EMBEDDER_ID)]
        embedder: String,
        #[arg(long, default_value = "recursive")]
        chunker: String,
        #[arg(long, default_value = "simple")]
        language: String,
    },
    /// Sync files to vector index
    Sync {
        #[arg(long)]
        full: bool,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        wait: bool,
        #[arg(long)]
        batch_size: Option<usize>,
        /// Stream progress events to stdout while syncing (human logs stay
        /// on stderr). With `--format json` the final result becomes the
        /// last stdout line, compact.
        #[arg(long, value_enum)]
        progress: Option<ProgressFormat>,
        /// Also write the progress stream (JSONL) to this file, leaving
        /// stdout for the final result.
        #[arg(long, value_name = "PATH")]
        progress_file: Option<std::path::PathBuf>,
        /// Longest gap between progress events before a heartbeat is sent.
        #[arg(
            long,
            default_value_t = crate::sync::DEFAULT_HEARTBEAT_SECONDS,
            value_parser = clap::value_parser!(u64).range(1..)
        )]
        progress_heartbeat_seconds: u64,
    },
    /// Search indexed content
    Query {
        text: String,
        #[arg(long, short, default_value = "10")]
        k: usize,
        #[arg(long, value_enum, default_value = "vector")]
        mode: QueryMode,
    },
    /// Show sync status
    Status,
    /// Compact, re-index and prune the vector store without syncing.
    /// Recovers stores left fragmented by interrupted syncs.
    Optimize,
    /// Check environment health
    Check,
    /// Verify index consistency
    Verify {
        #[arg(long)]
        fix: bool,
        #[arg(long)]
        all: bool,
        #[arg(long, default_value = "10")]
        sample: usize,
    },
    /// Watch the directory and incrementally re-index changed UTF-8 text files
    Watch {
        #[arg(long)]
        debounce_ms: Option<u64>,
        /// Keep watching when the initial sync fails; later events retry sync.
        #[arg(long)]
        watch_on_sync_error: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::{Cli, Commands};
    use clap::Parser;

    #[test]
    fn watch_accepts_sync_error_recovery_flag() {
        let cli = Cli::try_parse_from(["minsync", "watch", "--watch-on-sync-error"])
            .expect("watch recovery flag is accepted");

        assert!(matches!(
            cli.command,
            Commands::Watch {
                watch_on_sync_error: true,
                ..
            }
        ));
    }
}
