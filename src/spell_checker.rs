//! Port of Ruby's `did_you_mean` gem spell checker — the typo-suggestion
//! engine behind `DidYouMean` corrections.
//!
//! Credit: this is a faithful Rust port of the `did_you_mean` gem
//! (https://github.com/ruby/did_you_mean), released under the MIT
//! License, copyright (c) 2014 Yuki Nishijima. The `levenshtein.rb`
//! source it ports is in turn derived from the Text gem, copyright (c)
//! 2006-2013 Paul Battley, Michael Neumann, Tim Fletcher. The two-stage
//! JaroWinkler-then-Levenshtein design and its threshold constants are
//! reproduced as-is to stay behavior-compatible with Ruby's `DidYouMean`.
//!
//! Three pieces, mirroring the gem's files:
//!
//! - [`correct`] mirrors `DidYouMean::SpellChecker#correct`
//!   (`did_you_mean/lib/did_you_mean/spell_checker.rb`): a two-stage
//!   JaroWinkler-then-Levenshtein filter.
//! - [`jaro_winkler_distance`] mirrors `DidYouMean::JaroWinkler.distance`
//!   (`jaro_winkler.rb`) and [`jaro_distance`] its `Jaro.distance`.
//! - [`levenshtein_distance`] mirrors `DidYouMean::Levenshtein.distance`
//!   (`levenshtein.rb`, itself from the Text gem).
//!
//! The gem records Jaro match positions in arbitrary-precision integer
//! bit flags (`flags2 |= (1 << j)`); this port uses `Vec<bool>` instead,
//! behaviorally identical but free of the overflow a fixed-width integer
//! would hit on long inputs.


const JARO_WEIGHT: f64 = 0.1;
const JARO_THRESHOLD: f64 = 0.7;

/// Mirrors `DidYouMean::SpellChecker#correct`. Returns dictionary
/// elements judged near `input`, best match first, as their original
/// (un-normalized) spelling. Empty when nothing passes the thresholds.
pub fn correct(input: &str, dictionary: &[String]) -> Vec<String> {
    let normalized_input = normalize(input);
    let input_len = normalized_input.chars().count();
    let threshold = if input_len > 3 { 0.834 } else { 0.77 };

    let mut words: Vec<&String> = dictionary
        .iter()
        .filter(|word| jaro_winkler_distance(&normalize(word), &normalized_input) >= threshold)
        .filter(|word| input != word.as_str())
        .collect();
    words.sort_by(|a, b| {
        let da = jaro_winkler_distance(a, &normalized_input);
        let db = jaro_winkler_distance(b, &normalized_input);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    words.reverse();

    // Correct mistypes.
    let mistype_threshold = (input_len as f64 * 0.25).ceil() as usize;
    let corrections: Vec<String> = words
        .iter()
        .filter(|word| {
            levenshtein_distance(&normalize(word), &normalized_input) <= mistype_threshold
        })
        .map(|word| (*word).clone())
        .collect();
    if !corrections.is_empty() {
        return corrections;
    }

    // Correct misspells: take the first survivor whose edit distance is
    // below the shorter of the two lengths.
    words
        .iter()
        .filter(|word| {
            let normalized_word = normalize(word);
            let word_len = normalized_word.chars().count();
            let length = input_len.min(word_len);
            levenshtein_distance(&normalized_word, &normalized_input) < length
        })
        .take(1)
        .map(|word| (*word).clone())
        .collect()
}

/// Mirrors `DidYouMean::SpellChecker#normalize`: downcase, then drop `@`.
fn normalize(str: &str) -> String {
    str.to_lowercase().replace('@', "")
}

/// Mirrors `DidYouMean::JaroWinkler.distance`: Jaro distance with a
/// prefix boost (up to 4 leading matching characters) when the Jaro
/// distance clears `THRESHOLD`.
pub fn jaro_winkler_distance(str1: &str, str2: &str) -> f64 {
    let jaro = jaro_distance(str1, str2);
    if jaro > JARO_THRESHOLD {
        let codepoints2: Vec<char> = str2.chars().collect();
        let mut prefix_bonus = 0usize;
        for char1 in str1.chars() {
            if prefix_bonus < 4 && codepoints2.get(prefix_bonus) == Some(&char1) {
                prefix_bonus += 1;
            } else {
                break;
            }
        }
        jaro + (prefix_bonus as f64 * JARO_WEIGHT * (1.0 - jaro))
    } else {
        jaro
    }
}

/// Mirrors `DidYouMean::Jaro.distance`. The shorter string is forced
/// into `str1` (the gem swaps so `length1 <= length2`).
pub fn jaro_distance(str1: &str, str2: &str) -> f64 {
    let (str1, str2) = if str1.chars().count() > str2.chars().count() {
        (str2, str1)
    } else {
        (str1, str2)
    };
    let c1: Vec<char> = str1.chars().collect();
    let c2: Vec<char> = str2.chars().collect();
    let length1 = c1.len();
    let length2 = c2.len();

    let mut m = 0.0f64;
    let mut t = 0.0f64;
    let range = if length2 > 3 { length2 / 2 - 1 } else { 0 };
    let mut flags1 = vec![false; length1];
    let mut flags2 = vec![false; length2];

    let mut i = 0;
    while i < length1 {
        let last = i + range;
        // Gem: `(i >= range) ? i - range : 0` — a saturating subtraction.
        let mut j = i.saturating_sub(range);
        while j <= last && j < length2 {
            if !flags2[j] && c1[i] == c2[j] {
                flags2[j] = true;
                flags1[i] = true;
                m += 1.0;
                break;
            }
            j += 1;
        }
        i += 1;
    }

    let mut k = 0;
    let mut i = 0;
    while i < length1 {
        if flags1[i] {
            let mut j = k;
            let mut index = k;
            while j < length2 {
                index = j;
                if flags2[j] {
                    k = j + 1;
                    break;
                }
                j += 1;
            }
            if c1[i] != c2[index] {
                t += 1.0;
            }
        }
        i += 1;
    }
    let t = (t / 2.0).floor();

    if m == 0.0 {
        0.0
    } else {
        (m / length1 as f64 + m / length2 as f64 + (m - t) / m) / 3.0
    }
}

/// Mirrors `DidYouMean::Levenshtein.distance`: single-row dynamic
/// programming edit distance (insertion / deletion / substitution all
/// cost 1).
pub fn levenshtein_distance(str1: &str, str2: &str) -> usize {
    let c1: Vec<char> = str1.chars().collect();
    let c2: Vec<char> = str2.chars().collect();
    let n = c1.len();
    let m = c2.len();
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }

    let mut d: Vec<usize> = (0..=m).collect();
    let mut x = 0;
    for (i0, &char1) in c1.iter().enumerate() {
        let mut i = i0 + 1;
        for j in 0..m {
            let cost = if char1 == c2[j] { 0 } else { 1 };
            x = min3(d[j + 1] + 1, i + 1, d[j] + cost);
            d[j] = i;
            i = x;
        }
        d[m] = x;
    }
    x
}

fn min3(a: usize, b: usize, c: usize) -> usize {
    if a < b && a < c {
        a
    } else if b < c {
        b
    } else {
        c
    }
}
