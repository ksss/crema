//! Port of ActiveSupport::Inflector default rules. Mirrors the rule set in
//! activesupport/lib/active_support/inflections.rb plus the rule application
//! semantics in inflector/inflections.rb / inflector/methods.rb.
//!
//! Rails stores rules with `prepend` and iterates head-to-tail. We store with
//! `append` and iterate tail-to-head — same outcome: the latest-added rule
//! wins. This keeps the builder API natural for app overrides (see the
//! companion todo `mid_infusion_inflections_user_config`).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::config::InflectionsLocaleTable;

pub struct Inflector {
    plurals: Vec<Rule>,
    singulars: Vec<Rule>,
    uncountables: Vec<String>,
    acronyms: HashMap<String, String>,
}

struct Rule {
    pattern: Pattern,
    replacement: Vec<RepPart>,
}

struct Pattern {
    start_anchored: bool,
    end_anchored: bool,
    atoms: Vec<Atom>,
}

enum Atom {
    Lit(String),
    Char(CharSet),
    Alt(Vec<Vec<Atom>>),
    Capture(usize, Vec<Atom>),
    Optional(Vec<Atom>),
    // Case-sensitive single byte. Used by `add_irregular`'s
    // mismatched-head branch to pin the first character's case while
    // letting the remaining `Atom::Lit` match case-insensitively —
    // crema's hand-rolled Pattern has no per-atom case mode toggle,
    // so the head-of-word distinction lives at the Atom layer.
    // `parse_pattern` never emits this variant.
    ExactByte(u8),
}

#[derive(Clone, Copy)]
enum CharSet {
    NotVowel,
    NotF,
    LR,
    TI,
    IE,
}

enum RepPart {
    Lit(String),
    Backref(usize),
}

pub fn default_en() -> &'static Inflector {
    static INSTANCE: OnceLock<Inflector> = OnceLock::new();
    INSTANCE.get_or_init(Inflector::build_default_en)
}

/// Build the inflector callers should pass into the AR infusion
/// pipeline. When `inflections` is `None` (no
/// `[infusion.rails.inflections.en]` table in `crema.toml`) the
/// returned reference is the `default_en()` singleton; otherwise a
/// fresh inflector is allocated and pushed into the returned
/// `OwnedOrDefault`. This keeps the zero-config hot path allocation-
/// free while keeping the binary's call site free of the irregular /
/// acronym vocabulary.
pub fn build_for_config(inflections: Option<&InflectionsLocaleTable>) -> OwnedOrDefault {
    match inflections {
        None => OwnedOrDefault::Default,
        Some(table) => {
            let irregulars: Vec<(&str, &str)> = table
                .irregular
                .iter()
                .map(|pair| (pair[0].as_str(), pair[1].as_str()))
                .collect();
            let acronyms: Vec<&str> = table.acronym.iter().map(String::as_str).collect();
            OwnedOrDefault::Owned(build_default_en_with(&irregulars, &acronyms))
        }
    }
}

/// `Inflector` carrier returned by `build_for_config`. The singleton
/// path stays static and the owned path holds the allocation; callers
/// dereference to `&Inflector` through the `Deref` impl below.
pub enum OwnedOrDefault {
    Default,
    Owned(Inflector),
}

impl std::ops::Deref for OwnedOrDefault {
    type Target = Inflector;

    fn deref(&self) -> &Inflector {
        match self {
            OwnedOrDefault::Default => default_en(),
            OwnedOrDefault::Owned(inflector) => inflector,
        }
    }
}

/// Build a fresh inflector seeded with the Rails English defaults and
/// then extended with the user's `inflect.irregular` / `inflect.acronym`
/// directives. Extensions are pushed at the tail so the `apply` loop's
/// reverse iteration checks them first — matching Rails, where
/// `Inflections#irregular` prepends to its head-checked list.
///
/// The returned inflector is owned; callers thread `&Inflector` through
/// the AR synthesis path. The `default_en()` singleton is still used
/// when no extensions are supplied, so the zero-config hot path stays
/// allocation-free.
pub(crate) fn build_default_en_with(irregulars: &[(&str, &str)], acronyms: &[&str]) -> Inflector {
    let mut inflector = Inflector::build_default_en();
    for (singular, plural) in irregulars {
        inflector.add_irregular(singular, plural);
    }
    for word in acronyms {
        inflector.add_acronym(word);
    }
    inflector
}

impl Default for Inflector {
    fn default() -> Self {
        Self::new()
    }
}

impl Inflector {
    pub(crate) fn new() -> Self {
        Self {
            plurals: Vec::new(),
            singulars: Vec::new(),
            uncountables: Vec::new(),
            acronyms: HashMap::new(),
        }
    }

    pub(crate) fn singularize(&self, word: &str) -> String {
        self.apply(word, &self.singulars)
    }

    pub(crate) fn pluralize(&self, word: &str) -> String {
        self.apply(word, &self.plurals)
    }

    pub(crate) fn camelize(&self, word: &str) -> String {
        let mut out = String::new();
        for part in word.split('_').filter(|p| !p.is_empty()) {
            if let Some(replacement) = self.acronyms.get(&part.to_ascii_lowercase()) {
                out.push_str(replacement);
            } else {
                let mut chars = part.chars();
                if let Some(first) = chars.next() {
                    out.extend(first.to_uppercase());
                    out.push_str(chars.as_str());
                }
            }
        }
        out
    }

    pub(crate) fn add_plural(&mut self, pattern: &str, replacement: &str) {
        self.plurals.push(Rule {
            pattern: parse_pattern(pattern),
            replacement: parse_replacement(replacement),
        });
    }

    pub(crate) fn add_singular(&mut self, pattern: &str, replacement: &str) {
        self.singulars.push(Rule {
            pattern: parse_pattern(pattern),
            replacement: parse_replacement(replacement),
        });
    }

    pub(crate) fn add_irregular(&mut self, singular: &str, plural: &str) {
        // Mirrors ActiveSupport::Inflector::Inflections#irregular. The two
        // branches diverge in how casing flows through:
        //
        // * matched-head: a single capture-based rule per direction lets
        //   the input's first char flow into the replacement unchanged.
        // * mismatched-head: we emit 8 case-pinned rules (4 plural,
        //   4 singular) — Rails uses `/X(?i)rest$/` to anchor the head
        //   case and leave the tail case-insensitive. crema's hand-rolled
        //   Pattern has no per-atom case mode, so we pin the head byte
        //   with `CharSet::Exact` and rely on the existing case-
        //   insensitive `Atom::Lit` for the tail.
        let s0 = singular.chars().next().expect("irregular: empty singular");
        let srest = &singular[s0.len_utf8()..];
        let p0 = plural.chars().next().expect("irregular: empty plural");
        let prest = &plural[p0.len_utf8()..];

        if s0.eq_ignore_ascii_case(&p0) {
            let s0_str = s0.to_string();
            let s_pat = format!("({}){}$", s0_str, srest);
            let p_pat = format!("({}){}$", s0_str, prest);
            let plural_rep = format!("\\1{}", prest);
            let singular_rep = format!("\\1{}", srest);
            self.add_plural(&s_pat, &plural_rep);
            self.add_plural(&p_pat, &plural_rep);
            self.add_singular(&s_pat, &singular_rep);
            self.add_singular(&p_pat, &singular_rep);
        } else {
            // ASCII assumption matches the rest of the inflector
            // (`ascii_eq_ci`, byte-level Lit). Non-ASCII first letters
            // are out of scope here — the matched-head branch above is
            // equally byte-oriented today.
            let s0_up = s0.to_ascii_uppercase() as u8;
            let s0_dn = s0.to_ascii_lowercase() as u8;
            let p0_up = p0.to_ascii_uppercase() as u8;
            let p0_dn = p0.to_ascii_lowercase() as u8;
            let p_upper = format!("{}{}", p0.to_ascii_uppercase(), prest);
            let p_lower = format!("{}{}", p0.to_ascii_lowercase(), prest);
            let s_upper = format!("{}{}", s0.to_ascii_uppercase(), srest);
            let s_lower = format!("{}{}", s0.to_ascii_lowercase(), srest);

            push_exact_head_rule(&mut self.plurals, s0_up, srest, &p_upper);
            push_exact_head_rule(&mut self.plurals, s0_dn, srest, &p_lower);
            push_exact_head_rule(&mut self.plurals, p0_up, prest, &p_upper);
            push_exact_head_rule(&mut self.plurals, p0_dn, prest, &p_lower);

            push_exact_head_rule(&mut self.singulars, s0_up, srest, &s_upper);
            push_exact_head_rule(&mut self.singulars, s0_dn, srest, &s_lower);
            push_exact_head_rule(&mut self.singulars, p0_up, prest, &s_upper);
            push_exact_head_rule(&mut self.singulars, p0_dn, prest, &s_lower);
        }
    }

    pub(crate) fn add_uncountable(&mut self, word: &str) {
        self.uncountables.push(word.to_ascii_lowercase());
    }

    pub(crate) fn add_acronym(&mut self, word: &str) {
        self.acronyms
            .insert(word.to_ascii_lowercase(), word.to_string());
    }

    fn apply(&self, word: &str, rules: &[Rule]) -> String {
        if word.is_empty() {
            return String::new();
        }
        if self.is_uncountable(word) {
            return word.to_string();
        }
        let bytes = word.as_bytes();
        for rule in rules.iter().rev() {
            if let Some(MatchHit {
                start,
                end,
                captures,
            }) = match_rule(&rule.pattern, bytes)
            {
                let mut out = String::new();
                out.push_str(&word[..start]);
                for part in &rule.replacement {
                    match part {
                        RepPart::Lit(s) => out.push_str(s),
                        RepPart::Backref(idx) => {
                            if let Some(Some((s, e))) = captures.get(idx - 1).copied() {
                                out.push_str(&word[s..e]);
                            }
                        }
                    }
                }
                out.push_str(&word[end..]);
                return out;
            }
        }
        word.to_string()
    }

    fn is_uncountable(&self, word: &str) -> bool {
        // Matches Rails' `\b<word>\Z/i`: the entry sits at the end of the
        // input, preceded by start-of-string or a non-word char.
        let input = word.as_bytes();
        for entry in &self.uncountables {
            let entry_bytes = entry.as_bytes();
            if input.len() < entry_bytes.len() {
                continue;
            }
            let head = input.len() - entry_bytes.len();
            if !suffix_eq_ci(&input[head..], entry_bytes) {
                continue;
            }
            if head == 0 || !is_word_byte(input[head - 1]) {
                return true;
            }
        }
        false
    }

    fn build_default_en() -> Self {
        let mut inflector = Self::new();

        // plural rules — source order. Tail iteration means the last rule
        // registered is checked first.
        inflector.add_plural("$", "s");
        inflector.add_plural("s$", "s");
        inflector.add_plural("^(ax|test)is$", "\\1es");
        inflector.add_plural("(octop|vir)us$", "\\1i");
        inflector.add_plural("(octop|vir)i$", "\\1i");
        inflector.add_plural("(alias|status)$", "\\1es");
        inflector.add_plural("(bu)s$", "\\1ses");
        inflector.add_plural("(buffal|tomat)o$", "\\1oes");
        inflector.add_plural("([ti])um$", "\\1a");
        inflector.add_plural("([ti])a$", "\\1a");
        inflector.add_plural("sis$", "ses");
        inflector.add_plural("(?:([^f])fe|([lr])f)$", "\\1\\2ves");
        inflector.add_plural("(hive)$", "\\1s");
        inflector.add_plural("([^aeiouy]|qu)y$", "\\1ies");
        inflector.add_plural("(x|ch|ss|sh)$", "\\1es");
        inflector.add_plural("(matr|vert|ind)(?:ix|ex)$", "\\1ices");
        inflector.add_plural("^(m|l)ouse$", "\\1ice");
        inflector.add_plural("^(m|l)ice$", "\\1ice");
        inflector.add_plural("^(ox)$", "\\1en");
        inflector.add_plural("^(oxen)$", "\\1");
        inflector.add_plural("(quiz)$", "\\1zes");

        // singular rules
        inflector.add_singular("s$", "");
        inflector.add_singular("(ss)$", "\\1");
        inflector.add_singular("(n)ews$", "\\1ews");
        inflector.add_singular("([ti])a$", "\\1um");
        inflector.add_singular(
            "((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)(sis|ses)$",
            "\\1sis",
        );
        inflector.add_singular("(^analy)(sis|ses)$", "\\1sis");
        inflector.add_singular("([^f])ves$", "\\1fe");
        inflector.add_singular("(hive)s$", "\\1");
        inflector.add_singular("(tive)s$", "\\1");
        inflector.add_singular("([lr])ves$", "\\1f");
        inflector.add_singular("([^aeiouy]|qu)ies$", "\\1y");
        inflector.add_singular("(s)eries$", "\\1eries");
        inflector.add_singular("(m)ovies$", "\\1ovie");
        inflector.add_singular("(x|ch|ss|sh)es$", "\\1");
        inflector.add_singular("^(m|l)ice$", "\\1ouse");
        inflector.add_singular("(bus)(es)?$", "\\1");
        inflector.add_singular("(o)es$", "\\1");
        inflector.add_singular("(shoe)s$", "\\1");
        inflector.add_singular("(cris|test)(is|es)$", "\\1is");
        inflector.add_singular("^(a)x[ie]s$", "\\1xis");
        inflector.add_singular("(octop|vir)(us|i)$", "\\1us");
        inflector.add_singular("(alias|status)(es)?$", "\\1");
        inflector.add_singular("^(ox)en", "\\1");
        inflector.add_singular("(vert|ind)ices$", "\\1ex");
        inflector.add_singular("(matr)ices$", "\\1ix");
        inflector.add_singular("(quiz)zes$", "\\1");
        inflector.add_singular("(database)s$", "\\1");

        // irregulars — Rails prepends; we append in source order so the
        // first-declared irregular still ends up checked LAST among the
        // irregular block. Functionally identical to Rails.
        inflector.add_irregular("person", "people");
        inflector.add_irregular("man", "men");
        inflector.add_irregular("child", "children");
        inflector.add_irregular("sex", "sexes");
        inflector.add_irregular("move", "moves");
        inflector.add_irregular("zombie", "zombies");

        for word in [
            "equipment",
            "information",
            "rice",
            "money",
            "species",
            "series",
            "fish",
            "sheep",
            "jeans",
            "police",
        ] {
            inflector.add_uncountable(word);
        }

        inflector
    }
}

/// Build a 2-atom anchored Rule whose head is a case-pinned byte and
/// whose tail is the existing case-insensitive `Atom::Lit`, then push
/// it onto `rules`. The replacement is a plain literal — backrefs are
/// unnecessary because each of the 8 mismatched-head rules already
/// encodes its target case in the literal text.
fn push_exact_head_rule(rules: &mut Vec<Rule>, head: u8, rest: &str, replacement: &str) {
    let mut atoms = vec![Atom::ExactByte(head)];
    if !rest.is_empty() {
        atoms.push(Atom::Lit(rest.to_string()));
    }
    let pattern = Pattern {
        start_anchored: false,
        end_anchored: true,
        atoms,
    };
    rules.push(Rule {
        pattern,
        replacement: vec![RepPart::Lit(replacement.to_string())],
    });
}

struct MatchHit {
    start: usize,
    end: usize,
    captures: Vec<Option<(usize, usize)>>,
}

fn match_rule(pattern: &Pattern, input: &[u8]) -> Option<MatchHit> {
    let total_captures = count_captures(&pattern.atoms);
    let max_start = if pattern.start_anchored {
        0
    } else {
        input.len()
    };
    for start in 0..=max_start {
        let mut caps = vec![None; total_captures];
        if let Some(end) = match_seq(&pattern.atoms, input, start, &mut caps) {
            if pattern.end_anchored && end != input.len() {
                continue;
            }
            return Some(MatchHit {
                start,
                end,
                captures: caps,
            });
        }
    }
    None
}

fn match_seq(
    atoms: &[Atom],
    input: &[u8],
    pos: usize,
    caps: &mut Vec<Option<(usize, usize)>>,
) -> Option<usize> {
    let mut cur = pos;
    for atom in atoms {
        cur = match_atom(atom, input, cur, caps)?;
    }
    Some(cur)
}

fn match_atom(
    atom: &Atom,
    input: &[u8],
    pos: usize,
    caps: &mut Vec<Option<(usize, usize)>>,
) -> Option<usize> {
    match atom {
        Atom::Lit(s) => {
            let bytes = s.as_bytes();
            if pos + bytes.len() > input.len() {
                return None;
            }
            for (i, expected) in bytes.iter().enumerate() {
                if !ascii_eq_ci(*expected, input[pos + i]) {
                    return None;
                }
            }
            Some(pos + bytes.len())
        }
        Atom::ExactByte(byte) => {
            if pos < input.len() && input[pos] == *byte {
                Some(pos + 1)
            } else {
                None
            }
        }
        Atom::Char(set) => {
            if pos >= input.len() {
                return None;
            }
            if set.contains(input[pos]) {
                Some(pos + 1)
            } else {
                None
            }
        }
        Atom::Alt(branches) => {
            for branch in branches {
                let saved = caps.clone();
                if let Some(end) = match_seq(branch, input, pos, caps) {
                    return Some(end);
                }
                *caps = saved;
            }
            None
        }
        Atom::Capture(idx, sub) => {
            let start = pos;
            let end = match_seq(sub, input, pos, caps)?;
            caps[*idx - 1] = Some((start, end));
            Some(end)
        }
        Atom::Optional(sub) => {
            let saved = caps.clone();
            if let Some(end) = match_seq(sub, input, pos, caps) {
                Some(end)
            } else {
                *caps = saved;
                Some(pos)
            }
        }
    }
}

impl CharSet {
    fn contains(&self, b: u8) -> bool {
        let lower = b.to_ascii_lowercase();
        match self {
            CharSet::NotVowel => !matches!(lower, b'a' | b'e' | b'i' | b'o' | b'u' | b'y'),
            CharSet::NotF => lower != b'f',
            CharSet::LR => matches!(lower, b'l' | b'r'),
            CharSet::TI => matches!(lower, b't' | b'i'),
            CharSet::IE => matches!(lower, b'i' | b'e'),
        }
    }
}

fn ascii_eq_ci(a: u8, b: u8) -> bool {
    a.eq_ignore_ascii_case(&b)
}

fn suffix_eq_ci(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| ascii_eq_ci(*x, *y))
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn count_captures(atoms: &[Atom]) -> usize {
    let mut max = 0;
    for atom in atoms {
        let n = match atom {
            Atom::Capture(idx, sub) => (*idx).max(count_captures(sub)),
            Atom::Alt(branches) => branches
                .iter()
                .map(|b| count_captures(b))
                .max()
                .unwrap_or(0),
            Atom::Optional(sub) => count_captures(sub),
            _ => 0,
        };
        if n > max {
            max = n;
        }
    }
    max
}

/// Hand-rolled parser for the small regex dialect Rails default rules use:
/// `^` `$` anchors, `[set]` char classes, `(group)` captures,
/// `(?:group)` non-captures, `(?:group)?` / `(group)?` optionals,
/// alternation `|` inside groups, literal text otherwise.
///
/// Pre-scan strips every `^` outside `[...]` and sets `start_anchored` —
/// this lets us accept the one Rails rule that puts `^` inside a capture
/// (`(^analy)(sis|ses)$`) without a parser-side special case.
fn parse_pattern(src: &str) -> Pattern {
    let mut cleaned: Vec<u8> = Vec::with_capacity(src.len());
    let mut start_anchored = false;
    let mut in_class = false;
    for &b in src.as_bytes() {
        if in_class {
            cleaned.push(b);
            if b == b']' {
                in_class = false;
            }
        } else if b == b'[' {
            in_class = true;
            cleaned.push(b);
        } else if b == b'^' {
            start_anchored = true;
        } else {
            cleaned.push(b);
        }
    }
    let mut end_anchored = false;
    let mut end = cleaned.len();
    if end > 0 && cleaned[end - 1] == b'$' {
        end_anchored = true;
        end -= 1;
    }
    let mut idx = 0;
    let mut capture_counter: usize = 0;
    let atoms = parse_atoms(&cleaned, &mut idx, end, &mut capture_counter);
    Pattern {
        start_anchored,
        end_anchored,
        atoms,
    }
}

fn parse_atoms(src: &[u8], idx: &mut usize, end: usize, capture_counter: &mut usize) -> Vec<Atom> {
    let mut atoms: Vec<Atom> = Vec::new();
    while *idx < end {
        let b = src[*idx];
        match b {
            b'|' | b')' => break,
            b'(' => {
                *idx += 1;
                let (is_capture, captured_idx) =
                    if *idx + 1 < end && src[*idx] == b'?' && src[*idx + 1] == b':' {
                        *idx += 2;
                        (false, 0)
                    } else {
                        *capture_counter += 1;
                        (true, *capture_counter)
                    };
                let mut branches: Vec<Vec<Atom>> = Vec::new();
                let inner_end = find_group_end(src, *idx, end);
                let mut local_idx = *idx;
                loop {
                    let branch = parse_atoms(src, &mut local_idx, inner_end, capture_counter);
                    branches.push(branch);
                    if local_idx >= inner_end {
                        break;
                    }
                    if src[local_idx] == b'|' {
                        local_idx += 1;
                        continue;
                    }
                    break;
                }
                *idx = inner_end;
                assert_eq!(src[*idx], b')', "unbalanced group");
                *idx += 1;
                let mut optional = false;
                if *idx < end && src[*idx] == b'?' {
                    optional = true;
                    *idx += 1;
                }
                if is_capture {
                    let inner = if branches.len() == 1 {
                        branches.into_iter().next().unwrap()
                    } else {
                        vec![Atom::Alt(branches)]
                    };
                    let cap = Atom::Capture(captured_idx, inner);
                    if optional {
                        atoms.push(Atom::Optional(vec![cap]));
                    } else {
                        atoms.push(cap);
                    }
                } else if branches.len() == 1 {
                    let branch = branches.into_iter().next().unwrap();
                    if optional {
                        atoms.push(Atom::Optional(branch));
                    } else {
                        atoms.extend(branch);
                    }
                } else {
                    let alt = Atom::Alt(branches);
                    if optional {
                        atoms.push(Atom::Optional(vec![alt]));
                    } else {
                        atoms.push(alt);
                    }
                }
            }
            b'[' => {
                let close = src[*idx..end]
                    .iter()
                    .position(|c| *c == b']')
                    .map(|p| *idx + p)
                    .expect("unbalanced char class");
                let class = char_class_from_source(&src[*idx + 1..close]);
                atoms.push(Atom::Char(class));
                *idx = close + 1;
            }
            _ => {
                let literal_start = *idx;
                while *idx < end {
                    let c = src[*idx];
                    if matches!(c, b'(' | b')' | b'[' | b'|') {
                        break;
                    }
                    *idx += 1;
                }
                if *idx > literal_start {
                    let s = std::str::from_utf8(&src[literal_start..*idx])
                        .expect("non-utf8 literal in pattern")
                        .to_string();
                    atoms.push(Atom::Lit(s));
                }
            }
        }
    }
    atoms
}

fn find_group_end(src: &[u8], start: usize, end: usize) -> usize {
    let mut depth = 1;
    let mut i = start;
    while i < end {
        match src[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return i;
                }
            }
            b'[' => {
                while i < end && src[i] != b']' {
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    panic!("unbalanced group");
}

fn char_class_from_source(body: &[u8]) -> CharSet {
    let s = std::str::from_utf8(body).expect("non-utf8 char class");
    match s {
        "^aeiouy" => CharSet::NotVowel,
        "^f" => CharSet::NotF,
        "lr" => CharSet::LR,
        "ti" => CharSet::TI,
        "ie" => CharSet::IE,
        other => panic!("unsupported char class [{other}]"),
    }
}

fn parse_replacement(src: &str) -> Vec<RepPart> {
    let bytes = src.as_bytes();
    let mut parts: Vec<RepPart> = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
            if !buf.is_empty() {
                parts.push(RepPart::Lit(std::mem::take(&mut buf)));
            }
            let idx = (bytes[i + 1] - b'0') as usize;
            parts.push(RepPart::Backref(idx));
            i += 2;
        } else {
            buf.push(bytes[i] as char);
            i += 1;
        }
    }
    if !buf.is_empty() {
        parts.push(RepPart::Lit(buf));
    }
    parts
}

