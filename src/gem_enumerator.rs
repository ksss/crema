use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use crate::bundle_root::{GEMFILE, GEMFILE_LOCK};


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumeratedGem {
    pub name: String,
    pub gem_dir: PathBuf,
}

pub fn enumerate_bundled_gems() -> Result<Vec<EnumeratedGem>, String> {
    enumerate_bundled_gems_in(Path::new("."), None)
}

pub(crate) fn enumerate_bundled_gems_in(
    cwd: &Path,
    path_override: Option<&OsStr>,
) -> Result<Vec<EnumeratedGem>, String> {
    if !cwd.join(GEMFILE).exists() || !cwd.join(GEMFILE_LOCK).exists() {
        return Ok(Vec::new());
    }

    let script = bundled_gem_enumerator_script();
    let mut cmd = build_bundled_gem_enumerator_command(script);
    cmd.current_dir(cwd);
    if let Some(path) = path_override {
        cmd.env("PATH", path);
    }

    let output = cmd
        .spawn()
        .map_err(|e| format!("error: failed to spawn bundle: {}", e))?
        .wait_with_output()
        .map_err(|e| format!("error: failed to wait for ruby: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("error: ruby exited with failure: {}", stderr));
    }

    let stdout = std::str::from_utf8(&output.stdout)
        .map_err(|e| format!("error: ruby stdout is not UTF-8: {}", e))?;
    parse_enumerated_gems(stdout)
}

fn build_bundled_gem_enumerator_command(script: &str) -> std::process::Command {
    let mut cmd = std::process::Command::new("bundle");
    cmd.args(["exec", "ruby", "-e", script]);
    cmd.stdin(std::process::Stdio::piped());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    cmd
}

fn bundled_gem_enumerator_script() -> &'static str {
    r##"
require "bundler"
Bundler.load.specs.each do |spec|
  puts "#{spec.name}\t#{spec.full_gem_path}"
end
"##
}

fn parse_enumerated_gems(stdout: &str) -> Result<Vec<EnumeratedGem>, String> {
    let mut gems = Vec::new();
    for line in stdout.lines() {
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split('\t');
        let name = fields.next().unwrap_or_default();
        let gem_dir = fields.next().unwrap_or_default();
        if fields.next().is_some() || name.is_empty() || gem_dir.is_empty() {
            return Err(format!(
                "error: malformed bundled gem enumerator output line: {}",
                line
            ));
        }
        gems.push(EnumeratedGem {
            name: name.to_string(),
            gem_dir: PathBuf::from(gem_dir),
        });
    }
    Ok(gems)
}
