use std::collections::HashMap;
use std::path::PathBuf;

/// Gem-dir map resolved for one crema invocation. `rbs_gem_dir` is
/// hoisted out of the library map because the rest of the program
/// always needs it (for `core/` and the `stdlib/` fallback) and an
/// rbs entry must always be present — keeping it in `libraries` and
/// reading it back as `gem_dirs["rbs"]` re-introduces the implicit
/// "is this key present?" check at every caller.
///
/// Each `libraries` entry is `Some(path)` when Ruby reported a gem
/// dir for that name and `None` when `Gem::Specification.find_by_name`
/// raised `MissingSpecError` — the latter is the stdlib-fallback
/// sentinel that `library_loader::resolve` interprets.
///
/// `stale` maps a gem name (including `"rbs"`) to the installed
/// version string when its lock-pinned version wasn't installed but a
/// fallback (installed) version resolved instead — see
/// `resolve_gem_dirs`'s self-heal path. Callers use this to warn about
/// the stale pin.
///
/// `rbs_runtime_deps` lists `(name, gem_dir)` for each gemspec runtime
/// dependency of rbs that ships a `sig/` and is not shadowed by a
/// same-named `stdlib/` entry. Only populated when the resolver was
/// asked for it (script mode); empty on the normal path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedGemDirs {
    pub rbs_gem_dir: PathBuf,
    pub libraries: HashMap<String, Option<PathBuf>>,
    pub stale: HashMap<String, String>,
    pub rbs_runtime_deps: Vec<(String, PathBuf)>,
}
