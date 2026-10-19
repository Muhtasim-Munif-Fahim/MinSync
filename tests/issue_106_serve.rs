//! `minsync serve` (issue #106): CLI contract, and a live end-to-end run
//! against the real native model (`--ignored`; needs the cached weights).

use assert_cmd::Command;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Stdio};
use std::time::{Duration, Instant};

fn minsync(root: &Path) -> Command {
    let mut command = Command::cargo_bin("minsync").expect("find minsync binary");
    command.current_dir(root);
    command
}

#[test]
fn non_loopback_hosts_are_refused_before_loading_a_model() {
    let dir = tempfile::tempdir().unwrap();
    for host in ["0.0.0.0", "::", "192.168.0.10"] {
        let output = minsync(dir.path())
            .args(["serve", "--host", host, "--state-dir"])
            .arg(dir.path().join("state"))
            .timeout(Duration::from_secs(30))
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{host}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("not a loopback address"),
            "{host}: {stderr}"
        );
    }
    assert!(!dir.path().join("state").exists());
}

#[test]
fn only_native_embedders_can_be_served() {
    let dir = tempfile::tempdir().unwrap();
    let output = minsync(dir.path())
        .args(["serve", "--embedder", "tei:foo/bar"])
        .timeout(Duration::from_secs(30))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("only serves native"), "{stderr}");
}

/// A `minsync serve --format json` child and its first stdout line.
struct Served {
    child: Child,
    event: Value,
}

fn spawn_serve(root: &Path, state: &Path, extra: &[&str]) -> Served {
    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin("minsync"))
        .current_dir(root)
        .args(["--format", "json", "serve", "--state-dir"])
        .arg(state)
        .args(extra)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let event = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line:?}"));
    Served { child, event }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm(a) * norm(b))
}

fn vectors(value: &Value) -> Vec<Vec<f32>> {
    serde_json::from_value(value.clone()).unwrap()
}

/// Real model: parity with in-process `native:`, both protocols, prefixes,
/// limits, malformed input, adoption, SIGTERM.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "live: loads native:Qwen/Qwen3-Embedding-0.6B; run with --ignored"]
async fn live_server_matches_in_process_native_embeddings() {
    use minsync::embedder::Embedder;
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let served = spawn_serve(dir.path(), &state, &["--idle-timeout", "0"]);
    let mut child = served.child;
    assert_eq!(served.event["event"], "serving", "{}", served.event);
    let url = served.event["url"].as_str().unwrap().to_string();
    assert_eq!(served.event["dimension"], 1024);
    let client = reqwest::Client::new();

    let texts = vec![
        "MinSync indexes text files incrementally.".to_string(),
        "벡터 검색은 의미가 비슷한 문서를 찾는다.".to_string(),
        "The quick brown fox jumps over the lazy dog.".to_string(),
    ];
    let tei: Value = client
        .post(format!("{url}/embed"))
        .json(&json!({ "inputs": texts }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let tei = vectors(&tei);

    let config = minsync::config::Config::default_for("live");
    let local = minsync::embedder::native::NativeEmbedder::from_config(&config.embedder).unwrap();
    let expected = local.embed(&texts).await.unwrap();
    for (row, (a, b)) in expected.iter().zip(&tei).enumerate() {
        let diff = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(diff < 1e-6, "row {row}: server differs by {diff}");
    }

    let openai: Value = client
        .post(format!("{url}/v1/embeddings"))
        .json(&json!({ "model": "Qwen/Qwen3-Embedding-0.6B", "input": texts }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(openai["object"], "list");
    for (row, item) in openai["data"].as_array().unwrap().iter().enumerate() {
        assert_eq!(item["index"], row);
        let v: Vec<f32> = serde_json::from_value(item["embedding"].clone()).unwrap();
        assert!(cosine(&v, &tei[row]) > 0.999_999, "row {row}");
    }
    assert!(openai["usage"]["prompt_tokens"].as_u64().unwrap() > 10);

    // No prefixes configured: `input_type` must not change the text.
    let ext: Value = client
        .post(format!("{url}/embed"))
        .json(&json!({ "inputs": ["hello"], "input_type": "query" }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let plain: Value = client
        .post(format!("{url}/embed"))
        .json(&json!({ "inputs": ["hello"] }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(ext, plain);

    // Over the context: 413, server keeps serving.
    let long = "token ".repeat(3000);
    let response = client
        .post(format!("{url}/embed"))
        .json(&json!({ "inputs": [long] }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 413);
    let response = client
        .post(format!("{url}/embed"))
        .header("content-type", "application/json")
        .body("{\"inputs\": [")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 400);
    let too_many: Vec<String> = (0..513).map(|i| format!("t{i}")).collect();
    let response = client
        .post(format!("{url}/embed"))
        .json(&json!({ "inputs": too_many }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 413);

    // A second serve adopts instead of loading.
    let started = Instant::now();
    let adopted = spawn_serve(dir.path(), &state, &[]);
    let mut second = adopted.child;
    assert_eq!(adopted.event["event"], "adopted", "{}", adopted.event);
    assert_eq!(adopted.event["url"], url.as_str());
    assert!(second.wait().unwrap().success());
    assert!(started.elapsed() < Duration::from_secs(5));

    // SIGTERM: clean exit, files removed.
    #[cfg(unix)]
    {
        unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
        let status = child.wait().unwrap();
        assert_eq!(status.code(), Some(0));
        assert!(!state.join("port").exists());
        assert!(!state.join("pid").exists());
    }
    #[cfg(not(unix))]
    child.kill().unwrap();
}

fn write_corpus(root: &Path) {
    let docs = [
        (
            "rust.md",
            "Rust guarantees memory safety without a garbage collector.",
        ),
        (
            "bread.md",
            "Sourdough bread rises slowly with a wild yeast starter.",
        ),
        (
            "space.md",
            "The James Webb telescope observes infrared light from early galaxies.",
        ),
        (
            "korean.md",
            "김치는 배추를 소금에 절여 양념과 함께 발효시킨 음식이다.",
        ),
    ];
    for (name, text) in docs {
        std::fs::write(root.join(name), text).unwrap();
    }
}

fn query_json(root: &Path, server: &str, text: &str) -> (Vec<(String, f64)>, String) {
    let output = std::process::Command::new(assert_cmd::cargo::cargo_bin("minsync"))
        .current_dir(root)
        .env("MINSYNC_SERVER", server)
        .env("HOME", std::env::var("HOME").unwrap())
        .args(["--format", "json", "query", text, "--k", "4"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let results: Value = serde_json::from_slice(&output.stdout).unwrap();
    let ranked = results
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["path"].as_str().unwrap().to_string(),
                r["score"].as_f64().unwrap(),
            )
        })
        .collect();
    (ranked, String::from_utf8_lossy(&output.stderr).into_owned())
}

/// Real model: `sync`/`query` in a `native:` workspace use a running
/// server for the same instance (found through the default state dir) and
/// rank exactly like the in-process engine.
#[test]
#[ignore = "live: loads native:Qwen/Qwen3-Embedding-0.6B; run with --ignored"]
fn live_sync_and_query_discover_a_running_server() {
    let served = tempfile::tempdir().unwrap();
    let local = tempfile::tempdir().unwrap();
    for root in [served.path(), local.path()] {
        write_corpus(root);
        minsync(root).arg("init").assert().success();
    }
    // Default state dir, so discovery needs no configuration.
    let mut server = std::process::Command::new(assert_cmd::cargo::cargo_bin("minsync"))
        .current_dir(served.path())
        .args(["--format", "json", "serve", "--idle-timeout", "0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(server.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let event: Value = serde_json::from_str(&line).unwrap();
    assert!(
        matches!(event["event"].as_str(), Some("serving" | "adopted")),
        "{event}"
    );

    let sync = minsync(served.path())
        .arg("sync")
        .env("MINSYNC_SERVER", "on")
        .output()
        .unwrap();
    assert!(sync.status.success(), "{sync:?}");
    assert!(
        String::from_utf8_lossy(&sync.stderr).contains("using running minsync serve"),
        "{}",
        String::from_utf8_lossy(&sync.stderr)
    );
    minsync(local.path())
        .arg("sync")
        .env("MINSYNC_SERVER", "off")
        .assert()
        .success();

    for text in [
        "memory safe systems language",
        "발효 음식",
        "galaxy observation",
    ] {
        let (remote, stderr) = query_json(served.path(), "on", text);
        assert!(stderr.contains("using running minsync serve"), "{stderr}");
        let (in_process, stderr) = query_json(local.path(), "off", text);
        assert!(!stderr.contains("using running minsync serve"), "{stderr}");
        assert_eq!(
            remote.iter().map(|r| &r.0).collect::<Vec<_>>(),
            in_process.iter().map(|r| &r.0).collect::<Vec<_>>(),
            "{text}"
        );
        for ((_, a), (_, b)) in remote.iter().zip(&in_process) {
            assert!((a - b).abs() < 1e-4, "{text}: {a} vs {b}");
        }
    }

    if event["event"] == "serving" {
        #[cfg(unix)]
        unsafe {
            libc::kill(server.id() as i32, libc::SIGTERM)
        };
        #[cfg(not(unix))]
        server.kill().unwrap();
    }
    let _ = server.wait();
}
