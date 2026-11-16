//! Port of RubyGems' `Gem::Platform` (rubygems 4.0.x, `rubygems/platform.rb`):
//! the string parser, `===` (does a gem built for this platform run on
//! that one), and `platform_specificity_match` (how closely, lower is
//! closer). Bundler uses these to pick which installed variant of a
//! locked gem to load; `gem_lockfile` uses them to place the same dir.
//!
//! RubyGems' `"ruby"` platform is a plain string, not a `Gem::Platform`;
//! callers model it as `None` next to a `Platform`.


pub const RUBY: &str = "ruby";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Platform {
    pub cpu: Option<String>,
    pub os: String,
    pub version: Option<String>,
}

const RUBY_SPECIFICITY: i64 = 1_000_000;

impl Platform {
    /// `Gem::Platform.new(String)`. Os names that RubyGems does not know
    /// become `"unknown"`, which matches no real local platform.
    pub fn parse(arch: &str) -> Platform {
        let arch = arch.trim_end_matches('-');
        let (cpu_part, os_part) = match arch.split_once('-') {
            Some((cpu, os)) => (cpu, Some(os)),
            None => (arch, None),
        };
        let mut cpu = if has_i_x86(cpu_part) {
            Some("x86".to_string())
        } else {
            Some(cpu_part.to_string())
        };
        // Legacy jruby: a single word is the os, with no cpu.
        let os = match os_part {
            Some(os) => os,
            None => {
                cpu = None;
                cpu_part
            }
        };
        let (os, version) = parse_os(os, &mut cpu);
        Platform {
            cpu,
            os: os.to_string(),
            version,
        }
    }

    /// `Gem::Platform#===`: `self` is the gem's platform, `other` the one
    /// it would run on.
    pub fn matches(&self, other: &Platform) -> bool {
        let universal = |cpu: &Option<String>| cpu.as_deref() == Some("universal");
        if (universal(&self.cpu) || universal(&other.cpu))
            && self.os.starts_with("mingw")
            && other.os.starts_with("mingw")
        {
            return true;
        }
        let cpu_ok = self.cpu.is_none()
            || universal(&self.cpu)
            || other.cpu.is_none()
            || universal(&other.cpu)
            || self.cpu == other.cpu
            || (self.cpu.as_deref() == Some("arm")
                && other.cpu.as_deref().is_some_and(|c| c.starts_with("armv")));
        if !cpu_ok || self.os != other.os {
            return false;
        }
        if self.os != "linux" {
            return self.version.is_none()
                || other.version.is_none()
                || self.version == other.version;
        }
        if self.normalized_linux_version() == other.normalized_linux_version() {
            return true;
        }
        let v = self.version.as_deref().unwrap_or("");
        if let Some(ov) = other.version.as_deref()
            && [
                format!("musl{v}"),
                format!("musleabi{v}"),
                format!("musleabihf{v}"),
            ]
            .iter()
            .any(|m| m == ov)
        {
            return true;
        }
        self.version == other.version
    }

    fn normalized_linux_version(&self) -> Option<&str> {
        let v = self.version.as_deref()?;
        let v = v.strip_prefix("gnu").unwrap_or(v);
        let v = v
            .strip_suffix("eabihf")
            .or_else(|| v.strip_suffix("eabi"))
            .unwrap_or(v);
        (!v.is_empty()).then_some(v)
    }
}

/// `Gem::Platform.platform_specificity_match(spec_platform, user_platform)`
/// with `None` as the `"ruby"` platform. Lower is a closer match; an
/// exact match is the minimum.
pub fn specificity(spec: Option<&Platform>, user: Option<&Platform>) -> i64 {
    if spec == user {
        return -1;
    }
    let (Some(spec), Some(user)) = (spec, user) else {
        return RUBY_SPECIFICITY;
    };
    let os = if spec.os == user.os { 0 } else { 1 };
    let cpu = if spec.cpu == user.cpu
        || (spec.cpu.as_deref() == Some("arm")
            && user.cpu.as_deref().unwrap_or("").starts_with("arm"))
    {
        0
    } else if spec.cpu.is_none() || spec.cpu.as_deref() == Some("universal") {
        1
    } else {
        2
    };
    let version = if spec.version == user.version {
        0
    } else if spec.version.is_none() {
        1
    } else {
        2
    };
    os + cpu * 10 + version * 100
}

/// `/i\d86/` anywhere in the cpu.
fn has_i_x86(cpu: &str) -> bool {
    let b = cpu.as_bytes();
    b.windows(4)
        .any(|w| w[0] == b'i' && w[1].is_ascii_digit() && w[2] == b'8' && w[3] == b'6')
}

#[derive(Clone, Copy)]
enum Capture {
    /// `\d+`
    Digits,
    /// `\w+`
    Word,
    /// `\d+(?:\.\d+)*`
    Dotted,
    /// `\d+\.\d+`
    MajorMinor,
}

/// The tail of `key-?(capture)?` once `key` has matched: the optional
/// dash, then the longest prefix of `rest` the capture accepts.
fn optional_capture(rest: &str, capture: Capture) -> Option<String> {
    let rest = rest.strip_prefix('-').unwrap_or(rest);
    let digits = |s: &str| s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let len = match capture {
        Capture::Digits => digits(rest),
        Capture::Word => rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len()),
        Capture::Dotted => {
            let mut len = digits(rest);
            if len > 0 {
                while let Some(after) = rest[len..].strip_prefix('.') {
                    let more = digits(after);
                    if more == 0 {
                        break;
                    }
                    len += 1 + more;
                }
            }
            len
        }
        Capture::MajorMinor => {
            let major = digits(rest);
            match rest[major..].strip_prefix('.') {
                Some(after) if major > 0 && digits(after) > 0 => major + 1 + digits(after),
                _ => 0,
            }
        }
    };
    (len > 0).then(|| rest[..len].to_string())
}

/// `rest` after the leftmost `key` (unanchored `/key.../`).
fn after<'a>(os: &'a str, key: &str) -> Option<&'a str> {
    os.find(key).map(|i| &os[i + key.len()..])
}

/// The `case os` ladder of `Gem::Platform#initialize`, in its order.
fn parse_os<'a>(os: &'a str, cpu: &mut Option<String>) -> (&'a str, Option<String>) {
    if let Some(rest) = after(os, "aix") {
        return ("aix", optional_capture(rest, Capture::Digits));
    }
    if os.contains("cygwin") {
        return ("cygwin", None);
    }
    if let Some(rest) = after(os, "darwin") {
        return ("darwin", optional_capture(rest, Capture::Digits));
    }
    if os == "macruby" {
        return ("macruby", None);
    }
    if let Some(rest) = os.strip_prefix("macruby") {
        return ("macruby", optional_capture(rest, Capture::Dotted));
    }
    if let Some(rest) = after(os, "freebsd") {
        return ("freebsd", optional_capture(rest, Capture::Digits));
    }
    if os == "java" || os == "jruby" {
        return ("java", None);
    }
    if let Some(rest) = os.strip_prefix("java") {
        return ("java", optional_capture(rest, Capture::Dotted));
    }
    if let Some(rest) = os.strip_prefix("dalvik") {
        let tail = rest.strip_prefix('-').unwrap_or(rest);
        if tail.bytes().all(|b| b.is_ascii_digit()) {
            return ("dalvik", (!tail.is_empty()).then(|| tail.to_string()));
        }
    }
    if os == "dotnet" {
        return ("dotnet", None);
    }
    if let Some(rest) = os.strip_prefix("dotnet") {
        return ("dotnet", optional_capture(rest, Capture::Dotted));
    }
    if let Some(rest) = after(os, "linux") {
        return ("linux", optional_capture(rest, Capture::Word));
    }
    if os.contains("mingw32") {
        return ("mingw32", None);
    }
    if let Some(rest) = after(os, "mingw") {
        return ("mingw", optional_capture(rest, Capture::Word));
    }
    if let Some((name, version)) = mswin(os) {
        if cpu.is_none() && name.ends_with("32") {
            *cpu = Some("x86".to_string());
        }
        return (name, version);
    }
    if os.contains("netbsdelf") {
        return ("netbsdelf", None);
    }
    if let Some(rest) = after(os, "openbsd") {
        return ("openbsd", optional_capture(rest, Capture::MajorMinor));
    }
    if let Some(rest) = after(os, "solaris") {
        return ("solaris", optional_capture(rest, Capture::MajorMinor));
    }
    if os.contains("wasi") {
        return ("wasi", None);
    }
    ("unknown", None)
}

/// `/(mswin\d+)(?:[_-](\d+))?/`: the leftmost `mswin` followed by digits.
fn mswin(os: &str) -> Option<(&str, Option<String>)> {
    let mut from = 0;
    while let Some(i) = os[from..].find("mswin") {
        let start = from + i;
        let rest = &os[start + 5..];
        let digits = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        if digits > 0 {
            let name = &os[start..start + 5 + digits];
            let tail = &rest[digits..];
            let version = tail.strip_prefix(['_', '-']).and_then(|t| {
                let n = t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
                (n > 0).then(|| t[..n].to_string())
            });
            return Some((name, version));
        }
        from = start + 5;
    }
    None
}
