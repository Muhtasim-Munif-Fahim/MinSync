//! Machine-readable progress stream for a running sync (`--progress jsonl`).
//!
//! One JSON object per line, flushed per event, so a host that spawned
//! `minsync` can read progress while the run is still going. Events:
//!
//! - `start`: before the first batch; totals and run flavour.
//! - `batch`: one per committed batch.
//! - `heartbeat`: only when no other event was emitted for the heartbeat
//!   interval, so the gap between events never exceeds it. A dedicated
//!   thread emits it, so a blocking store commit or checkpoint cannot
//!   silence it; compare `processed` between heartbeats to tell a slow run
//!   from a stuck one.
//! - `done`: last event, counters equal the final result.
//!
//! This is unrelated to [`super::progress`], the crash-resume log.

use crate::types::{FileFailure, FileStage, SyncResult};
use parking_lot::Mutex;
use serde::Serialize;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// What `start` reports about the run.
pub(super) struct RunStart<'a> {
    pub started: Instant,
    pub total: usize,
    pub resumed: usize,
    pub full: bool,
    pub dry_run: bool,
    /// `None` when the run writes nothing (dry run, already up to date).
    pub sync_token: Option<&'a str>,
}

/// Default `--progress-heartbeat-seconds`.
pub const DEFAULT_HEARTBEAT_SECONDS: u64 = 5;

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum ProgressEvent<'a> {
    Start {
        total_files: usize,
        resumed: usize,
        full: bool,
        dry_run: bool,
        /// `null` when the run writes nothing (dry run, already up to date).
        sync_token: Option<&'a str>,
    },
    Batch {
        processed: usize,
        total: usize,
        path_first: &'a str,
        path_last: &'a str,
        elapsed_ms: u128,
    },
    Heartbeat {
        processed: usize,
        total: usize,
        elapsed_ms: u128,
    },
    FileFailed {
        path: &'a str,
        stage: FileStage,
        error: &'a str,
    },
    Done {
        processed: usize,
        total: usize,
        files_processed: usize,
        files_added: usize,
        files_modified: usize,
        files_deleted: usize,
        files_resumed: usize,
        chunks_added: usize,
        chunks_updated: usize,
        chunks_deleted: usize,
        chunks_truncated: usize,
        dry_run: bool,
        already_up_to_date: bool,
        files_failed: usize,
        partial: bool,
        elapsed_seconds: f64,
    },
}

struct Sinks {
    writers: Vec<Box<dyn Write + Send>>,
    last_emit: Instant,
    started: Instant,
}

impl Sinks {
    /// Write one event to every sink. A sink that fails (host closed the
    /// pipe, disk full) is dropped with one warning; progress is advisory and
    /// must never abort the sync.
    fn emit(&mut self, event: &ProgressEvent<'_>) {
        let mut line = match serde_json::to_vec(event) {
            Ok(line) => line,
            Err(error) => {
                tracing::warn!("cannot serialize progress event: {error}");
                return;
            }
        };
        line.push(b'\n');
        self.writers.retain_mut(|writer| {
            match writer.write_all(&line).and_then(|()| writer.flush()) {
                Ok(()) => true,
                Err(error) => {
                    tracing::warn!("progress sink closed: {error}");
                    false
                }
            }
        });
        self.last_emit = Instant::now();
    }
}

struct Shared {
    sinks: Mutex<Sinks>,
    processed: AtomicUsize,
    total: AtomicUsize,
}

impl Shared {
    fn lock(&self) -> parking_lot::MutexGuard<'_, Sinks> {
        self.sinks.lock()
    }
}

/// Stops and joins the heartbeat thread when dropped.
struct Heartbeat {
    stop: Option<mpsc::Sender<()>>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub struct ProgressReporter {
    shared: Arc<Shared>,
    interval: Duration,
    heartbeat: Mutex<Option<Heartbeat>>,
}

impl ProgressReporter {
    /// `writers` receive every event; `heartbeat` bounds the gap between
    /// events once the run has started.
    pub fn new(writers: Vec<Box<dyn Write + Send>>, heartbeat: Duration) -> Self {
        let now = Instant::now();
        Self {
            shared: Arc::new(Shared {
                sinks: Mutex::new(Sinks {
                    writers,
                    last_emit: now,
                    started: now,
                }),
                processed: AtomicUsize::new(0),
                total: AtomicUsize::new(0),
            }),
            interval: heartbeat,
            heartbeat: Mutex::new(None),
        }
    }

    /// Emit `start`; heartbeat only while the run writes (has a sync token).
    pub(super) fn start(&self, run: RunStart<'_>) {
        let RunStart {
            started,
            total,
            resumed,
            full,
            dry_run,
            sync_token,
        } = run;
        let heartbeat = sync_token.is_some();
        self.shared.processed.store(resumed, Ordering::Relaxed);
        self.shared.total.store(total, Ordering::Relaxed);
        {
            let mut sinks = self.shared.lock();
            sinks.started = started;
            sinks.emit(&ProgressEvent::Start {
                total_files: total,
                resumed,
                full,
                dry_run,
                sync_token,
            });
        }
        if heartbeat {
            *self.heartbeat.lock() = Some(self.spawn_heartbeat());
        }
    }

    /// One skipped file (skip-and-record), emitted as soon as it is known.
    pub(super) fn file_failed(&self, failure: &FileFailure) {
        self.shared.lock().emit(&ProgressEvent::FileFailed {
            path: &failure.path,
            stage: failure.stage,
            error: &failure.error,
        });
    }

    /// `processed` counts resumed files too, like the `total` it is
    /// compared with.
    pub(super) fn batch(&self, processed: usize, path_first: &str, path_last: &str) {
        self.shared.processed.store(processed, Ordering::Relaxed);
        let mut sinks = self.shared.lock();
        let elapsed_ms = sinks.started.elapsed().as_millis();
        sinks.emit(&ProgressEvent::Batch {
            processed,
            total: self.shared.total.load(Ordering::Relaxed),
            path_first,
            path_last,
            elapsed_ms,
        });
    }

    /// Stop heartbeating without emitting `done` (the run failed).
    pub(super) fn abort(&self) {
        drop(self.heartbeat.lock().take());
    }

    /// Stop heartbeating and emit the last event.
    pub(super) fn done(&self, result: &SyncResult, total: usize) {
        drop(self.heartbeat.lock().take());
        self.shared.lock().emit(&ProgressEvent::Done {
            processed: result.files_processed + result.files_resumed,
            total,
            files_processed: result.files_processed,
            files_added: result.files_added,
            files_modified: result.files_modified,
            files_deleted: result.files_deleted,
            files_resumed: result.files_resumed,
            chunks_added: result.chunks_added,
            chunks_updated: result.chunks_updated,
            chunks_deleted: result.chunks_deleted,
            chunks_truncated: result.chunks_truncated,
            dry_run: result.dry_run,
            already_up_to_date: result.already_up_to_date,
            files_failed: result.files_failed.len(),
            partial: result.partial,
            elapsed_seconds: result.elapsed_seconds,
        });
    }

    fn spawn_heartbeat(&self) -> Heartbeat {
        let shared = Arc::clone(&self.shared);
        let interval = self.interval;
        let (stop, stopped) = mpsc::channel::<()>();
        let handle = std::thread::Builder::new()
            .name("minsync-heartbeat".to_string())
            .spawn(move || loop {
                let wait = {
                    let sinks = shared.lock();
                    interval.saturating_sub(sinks.last_emit.elapsed())
                };
                match stopped.recv_timeout(wait) {
                    Err(RecvTimeoutError::Timeout) => {
                        let mut sinks = shared.lock();
                        // A batch event may have landed while waiting.
                        if sinks.last_emit.elapsed() >= interval {
                            let elapsed_ms = sinks.started.elapsed().as_millis();
                            sinks.emit(&ProgressEvent::Heartbeat {
                                processed: shared.processed.load(Ordering::Relaxed),
                                total: shared.total.load(Ordering::Relaxed),
                                elapsed_ms,
                            });
                        }
                    }
                    // Sender dropped (or a stray message): the run is over.
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
                }
            })
            .ok();
        if handle.is_none() {
            tracing::warn!("cannot start progress heartbeat thread");
        }
        Heartbeat {
            stop: Some(stop),
            handle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn run(
        started: Instant,
        total: usize,
        resumed: usize,
        full: bool,
        dry_run: bool,
        sync_token: Option<&str>,
    ) -> RunStart<'_> {
        RunStart {
            started,
            total,
            resumed,
            full,
            dry_run,
            sync_token,
        }
    }

    #[derive(Clone, Default)]
    struct Capture(Arc<Mutex<Vec<u8>>>);

    impl Write for Capture {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Capture {
        fn events(&self) -> Vec<Value> {
            String::from_utf8(self.0.lock().clone())
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        }
    }

    struct Broken;

    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn heartbeat_fires_only_after_a_quiet_interval_and_stops_at_done() {
        let capture = Capture::default();
        let reporter =
            ProgressReporter::new(vec![Box::new(capture.clone())], Duration::from_millis(60));
        let started = Instant::now();
        reporter.start(run(started, 10, 3, false, false, Some("tok")));
        // Wait for the observable condition, not a fixed time: a loaded CI
        // runner can starve the heartbeat thread for far longer than 200ms.
        let deadline = Instant::now() + Duration::from_secs(30);
        while capture
            .events()
            .iter()
            .filter(|e| e["event"] == "heartbeat")
            .count()
            < 2
        {
            assert!(
                Instant::now() < deadline,
                "no heartbeats: {:?}",
                capture.events()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        reporter.batch(5, "a.md", "b.md");
        reporter.done(&crate::sync::result::empty_sync_result(false, false), 10);
        let after_done = capture.events().len();
        std::thread::sleep(Duration::from_millis(150));
        let events = capture.events();

        assert_eq!(events.len(), after_done, "no events after done");
        assert_eq!(events[0]["event"], "start");
        assert_eq!(events[0]["resumed"], 3);
        assert_eq!(events[0]["sync_token"], "tok");
        let heartbeats: Vec<_> = events
            .iter()
            .filter(|e| e["event"] == "heartbeat")
            .collect();
        assert!(heartbeats.len() >= 2, "got {events:?}");
        // Quiet period: counters carry the resumed baseline.
        assert_eq!(heartbeats[0]["processed"], 3);
        assert_eq!(heartbeats[0]["total"], 10);
        let batch = events
            .iter()
            .find(|e| e["event"] == "batch")
            .expect("batch event");
        assert_eq!(batch["processed"], 5);
        assert_eq!(events.last().unwrap()["event"], "done");
    }

    #[test]
    fn busy_run_emits_no_heartbeat() {
        let capture = Capture::default();
        let reporter =
            ProgressReporter::new(vec![Box::new(capture.clone())], Duration::from_secs(60));
        reporter.start(run(Instant::now(), 4, 0, true, false, Some("t")));
        reporter.batch(2, "a", "b");
        reporter.batch(4, "c", "d");
        reporter.done(&crate::sync::result::empty_sync_result(false, false), 4);

        let names: Vec<_> = capture
            .events()
            .iter()
            .map(|e| e["event"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, ["start", "batch", "batch", "done"]);
    }

    #[test]
    fn failing_sink_is_dropped_without_affecting_other_sinks() {
        let capture = Capture::default();
        let reporter = ProgressReporter::new(
            vec![Box::new(Broken), Box::new(capture.clone())],
            Duration::from_secs(60),
        );
        reporter.start(run(Instant::now(), 1, 0, false, true, None));
        reporter.done(&crate::sync::result::empty_sync_result(true, false), 1);

        let events = capture.events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["sync_token"], Value::Null);
        assert_eq!(events[1]["dry_run"], true);
    }
}
