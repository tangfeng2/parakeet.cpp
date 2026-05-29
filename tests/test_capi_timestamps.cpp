#include "parakeet_capi.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Flat C-API JSON timestamps test (Task 4 of the timestamps+confidence plan).
//
// Loads the parakeet-tdt_ctc-110m anchor GGUF, calls
// parakeet_capi_transcribe_path_json on tests/fixtures/speech.wav with the TDT
// head (decoder == 2), and asserts against the Task 3 validated result:
//   * the "text" field equals the known NeMo TDT transcript,
//   * the words array has exactly 23 entries,
//   * the first word is "Well," with start ~= 0.48 s and a sensible word conf
//     (the 'min' aggregate over its tokens, ~0.79 per the validated baseline),
//   * the JSON also carries a non-empty "tokens" array.
// Parsing is intentionally minimal (a tiny hand-rolled object scanner over the
// JSON the C-API emits) — enough to validate the contract without a JSON lib.
//
// LABEL model
// WORKING_DIRECTORY (tests run from the project root; the wav path is relative)
//
// Env:
//   PARAKEET_TEST_GGUF   model weights (skip 77 if unset)

static const char* kExpected =
    "Well, I don't wish to see it any more, observed Phoebe, turning away her "
    "eyes. It is certainly very like the old portrait.";

namespace {

// One parsed word object {"w","start","end","conf"}.
struct JWord {
    std::string w;
    double start = 0.0, end = 0.0, conf = 0.0;
};

// Tiny scanner over the C-API JSON. Not a general JSON parser — it understands
// exactly the document parakeet_capi_transcribe_path_json emits.
struct Scan {
    const std::string& s;
    size_t i = 0;
    explicit Scan(const std::string& str) : s(str) {}
    void ws() { while (i < s.size() && (s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r')) ++i; }
    bool eat(char c) { ws(); if (i < s.size() && s[i]==c) { ++i; return true; } return false; }
    bool str(std::string& out) {
        ws();
        if (i >= s.size() || s[i] != '"') return false;
        ++i; out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[i+1];
                switch (c) {
                    case 'n': out += '\n'; break; case 't': out += '\t'; break;
                    case 'r': out += '\r'; break; case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break; case '"': out += '"'; break;
                    case '\\': out += '\\'; break; case '/': out += '/'; break;
                    default: out += c; break;
                }
                i += 2;
            } else { out += s[i++]; }
        }
        if (i >= s.size()) return false;
        ++i; return true;
    }
    bool num(double& out) {
        ws();
        size_t st = i;
        while (i < s.size() && std::strchr("+-0123456789.eE", s[i])) ++i;
        if (i == st) return false;
        out = std::strtod(s.substr(st, i - st).c_str(), nullptr);
        return true;
    }
    // Find the next key matching `key` at the current position, returning after
    // its ':'; advances to the value. (Linear scan from current i.)
    bool seek_key(const char* key) {
        std::string pat = std::string("\"") + key + "\"";
        size_t p = s.find(pat, i);
        if (p == std::string::npos) return false;
        i = p + pat.size();
        return eat(':');
    }
};

// Parse the "words":[ ... ] array out of the JSON into `out`.
bool parse_words(const std::string& s, std::vector<JWord>& out) {
    Scan sc(s);
    if (!sc.seek_key("words")) return false;
    if (!sc.eat('[')) return false;
    sc.ws();
    if (sc.i < s.size() && s[sc.i] == ']') return true;  // empty
    while (true) {
        if (!sc.eat('{')) return false;
        JWord jw;
        bool gw=false, gs=false, ge=false, gc=false;
        while (true) {
            std::string key;
            if (!sc.str(key)) return false;
            if (!sc.eat(':')) return false;
            if (key == "w")          { if (!sc.str(jw.w))    return false; gw = true; }
            else if (key == "start") { if (!sc.num(jw.start)) return false; gs = true; }
            else if (key == "end")   { if (!sc.num(jw.end))   return false; ge = true; }
            else if (key == "conf")  { if (!sc.num(jw.conf))  return false; gc = true; }
            else { return false; }
            if (sc.eat(',')) continue;
            break;
        }
        if (!sc.eat('}')) return false;
        if (!(gw && gs && ge && gc)) return false;
        out.push_back(std::move(jw));
        if (sc.eat(',')) continue;
        break;
    }
    return sc.eat(']');
}

bool parse_text(const std::string& s, std::string& out) {
    Scan sc(s);
    if (!sc.seek_key("text")) return false;
    return sc.str(out);
}

} // namespace

int main() {
    // ABI version sanity.
    if (parakeet_capi_abi_version() < 1) {
        std::fprintf(stderr, "test_capi_timestamps: abi version < 1\n");
        return 1;
    }

    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) {
        std::fprintf(stderr, "test_capi_timestamps: PARAKEET_TEST_GGUF not set; skip\n");
        return 77;
    }

    parakeet_ctx* ctx = parakeet_capi_load(gguf);
    if (!ctx) {
        std::fprintf(stderr, "test_capi_timestamps: load failed for %s\n", gguf);
        return 1;
    }

    // decoder == 2 -> TDT/transducer head.
    char* json = parakeet_capi_transcribe_path_json(ctx, "tests/fixtures/speech.wav", 2);
    if (!json) {
        std::fprintf(stderr, "test_capi_timestamps: transcribe_path_json NULL: %s\n",
                     parakeet_capi_last_error(ctx));
        parakeet_capi_free(ctx);
        return 1;
    }

    const std::string doc(json);
    parakeet_capi_free_string(json);
    parakeet_capi_free(ctx);

    std::fprintf(stderr, "test_capi_timestamps: json head = %.120s ...\n", doc.c_str());

    bool ok = true;

    // text field == known transcript.
    std::string text;
    if (!parse_text(doc, text)) {
        std::fprintf(stderr, "test_capi_timestamps: no \"text\" field\n");
        ok = false;
    } else if (text != kExpected) {
        std::fprintf(stderr, "test_capi_timestamps: text MISMATCH\n  got = %s\n  exp = %s\n",
                     text.c_str(), kExpected);
        ok = false;
    }

    // tokens array present and non-empty (substring check is enough).
    if (doc.find("\"tokens\":[{\"id\":") == std::string::npos) {
        std::fprintf(stderr, "test_capi_timestamps: missing/empty \"tokens\" array\n");
        ok = false;
    }

    // words array: 23 entries, first is "Well," with start ~= 0.48 and conf ~= 0.79.
    std::vector<JWord> words;
    if (!parse_words(doc, words)) {
        std::fprintf(stderr, "test_capi_timestamps: failed to parse \"words\"\n");
        ok = false;
    } else {
        if (words.size() != 23) {
            std::fprintf(stderr, "test_capi_timestamps: word COUNT got=%zu exp=23\n",
                         words.size());
            ok = false;
        }
        if (!words.empty()) {
            const JWord& w0 = words[0];
            std::fprintf(stderr,
                "test_capi_timestamps: word[0] = '%s' start=%.3f end=%.3f conf=%.4f\n",
                w0.w.c_str(), w0.start, w0.end, w0.conf);
            if (w0.w != "Well,") {
                std::fprintf(stderr, "test_capi_timestamps: word[0] text != 'Well,'\n");
                ok = false;
            }
            // start ~= 0.48 s (within one 0.08 s frame).
            if (std::fabs(w0.start - 0.48) > 0.085) {
                std::fprintf(stderr, "test_capi_timestamps: word[0].start %.3f != ~0.48\n",
                             w0.start);
                ok = false;
            }
            // conf is the 'min' aggregate over the word's tokens (~0.79 per the
            // Task 3 NeMo-validated baseline); allow a generous band.
            if (w0.conf < 0.70 || w0.conf > 0.85) {
                std::fprintf(stderr, "test_capi_timestamps: word[0].conf %.4f not ~0.79\n",
                             w0.conf);
                ok = false;
            }
            if (!(w0.end > w0.start)) {
                std::fprintf(stderr, "test_capi_timestamps: word[0].end <= start\n");
                ok = false;
            }
        }
    }

    if (!ok) { std::fprintf(stderr, "test_capi_timestamps: FAIL\n"); return 1; }
    std::fprintf(stderr, "test_capi_timestamps: PASS\n");
    return 0;
}
