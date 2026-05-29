#include "model.hpp"
#include "transcription.hpp"
#include "model_loader.hpp"
#include "decode_types.hpp"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

// Word grouping + Transcription parity vs NeMo (Task 3 of the
// timestamps+confidence plan).
//
// On tests/fixtures/speech.wav + the parakeet-tdt_ctc-110m anchor model, run
// pk::Model::transcribe_path_with_timestamps for BOTH the TDT head
// (Decoder::kTDT) and the CTC head (Decoder::kCTC) and assert, against the
// /tmp/baseline_ts.gguf NeMo baseline:
//   * per-token {id, frame, conf} == ts_<head>_token_{ids,frames,conf}
//     (re-using Task 2's checks; conf within 1e-3)
//   * word texts == baseline.<head>_words_json EXACTLY (count + each "w")
//   * word start/end within frame_sec (one frame) of the JSON
//   * word conf within 1e-3 of the JSON
//
// LABEL model
// WORKING_DIRECTORY (tests run from the project root; the wav path is relative)
//
// Env:
//   PARAKEET_TEST_GGUF          : model weights (skip 77 if unset)
//   PARAKEET_TEST_BASELINE_TS   : timestamps baseline gguf (skip 77 if unset)

namespace {

bool read_i32(ggml_context* ctx, const char* name, std::vector<int32_t>& out) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) { std::fprintf(stderr, "[ts3] tensor '%s' not found\n", name); return false; }
    const int n = (int)ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_I32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(int32_t));
    } else if (t->type == GGML_TYPE_F32) {
        const float* d = (const float*)t->data;
        for (int i = 0; i < n; ++i) out[i] = (int32_t)d[i];
    } else {
        std::fprintf(stderr, "[ts3] tensor '%s' unexpected type %d\n", name, (int)t->type);
        return false;
    }
    return true;
}

bool read_f32(ggml_context* ctx, const char* name, std::vector<float>& out) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) { std::fprintf(stderr, "[ts3] tensor '%s' not found\n", name); return false; }
    const int n = (int)ggml_nelements(t);
    out.resize(n);
    if (t->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "[ts3] tensor '%s' unexpected type %d\n", name, (int)t->type);
        return false;
    }
    std::memcpy(out.data(), t->data, (size_t)n * sizeof(float));
    return true;
}

// A NeMo word offset parsed out of baseline.<head>_words_json.
struct RefWord {
    std::string w;
    double start = 0.0, end = 0.0, conf = 0.0;
};

// Minimal parser for `[{"w":"..","start":N,"end":N,"conf":N}, ...]` (JSON
// emitted by gen_nemo_baseline.py). Handles string escapes (\" \\ \/ etc.).
// Returns false on a structural surprise.
bool parse_words_json(const std::string& s, std::vector<RefWord>& out) {
    size_t i = 0;
    auto skip_ws = [&]() { while (i < s.size() && (s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r')) ++i; };
    auto parse_string = [&](std::string& dst) -> bool {
        skip_ws();
        if (i >= s.size() || s[i] != '"') return false;
        ++i;
        dst.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[i+1];
                switch (c) {
                    case 'n': dst += '\n'; break;
                    case 't': dst += '\t'; break;
                    case 'r': dst += '\r'; break;
                    case '"': dst += '"';  break;
                    case '\\': dst += '\\'; break;
                    case '/': dst += '/';  break;
                    default:  dst += c;    break;
                }
                i += 2;
            } else {
                dst += s[i++];
            }
        }
        if (i >= s.size()) return false;
        ++i; // closing quote
        return true;
    };
    auto parse_number = [&](double& dst) -> bool {
        skip_ws();
        size_t start = i;
        while (i < s.size() && (std::strchr("+-0123456789.eE", s[i]) != nullptr)) ++i;
        if (i == start) return false;
        dst = std::strtod(s.substr(start, i - start).c_str(), nullptr);
        return true;
    };
    auto expect = [&](char c) -> bool { skip_ws(); if (i < s.size() && s[i] == c) { ++i; return true; } return false; };

    if (!expect('[')) return false;
    skip_ws();
    if (i < s.size() && s[i] == ']') return true; // empty
    while (true) {
        if (!expect('{')) return false;
        RefWord rw;
        bool got_w = false, got_s = false, got_e = false, got_c = false;
        while (true) {
            std::string key;
            if (!parse_string(key)) return false;
            if (!expect(':')) return false;
            if (key == "w") { if (!parse_string(rw.w)) return false; got_w = true; }
            else if (key == "start") { if (!parse_number(rw.start)) return false; got_s = true; }
            else if (key == "end")   { if (!parse_number(rw.end))   return false; got_e = true; }
            else if (key == "conf")  { if (!parse_number(rw.conf))  return false; got_c = true; }
            else { // skip an unknown value (string or number)
                skip_ws();
                if (i < s.size() && s[i] == '"') { std::string tmp; parse_string(tmp); }
                else { double tmp; parse_number(tmp); }
            }
            skip_ws();
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            break;
        }
        if (!expect('}')) return false;
        if (!(got_w && got_s && got_e && got_c)) return false;
        out.push_back(std::move(rw));
        skip_ws();
        if (i < s.size() && s[i] == ',') { ++i; continue; }
        break;
    }
    return expect(']');
}

bool read_words_json(gguf_context* g, const char* key, std::vector<RefWord>& out) {
    int64_t kid = gguf_find_key(g, key);
    if (kid < 0) { std::fprintf(stderr, "[ts3] kv '%s' not found\n", key); return false; }
    const char* str = gguf_get_val_str(g, kid);
    if (!str) { std::fprintf(stderr, "[ts3] kv '%s' not a string\n", key); return false; }
    if (!parse_words_json(std::string(str), out)) {
        std::fprintf(stderr, "[ts3] failed to parse '%s'\n", key);
        return false;
    }
    return true;
}

// Re-use Task 2's per-token check: ids exact, frames exact, conf within tol.
bool compare_tokens(const char* tag,
                    const std::vector<pk::TokenInfo>& got,
                    const std::vector<int32_t>& ref_ids,
                    const std::vector<int32_t>& ref_frames,
                    const std::vector<float>& ref_conf,
                    float conf_tol) {
    if (got.size() != ref_ids.size()) {
        std::fprintf(stderr, "[ts3:%s] token COUNT mismatch got=%zu ref=%zu\n",
                     tag, got.size(), ref_ids.size());
        return false;
    }
    bool ok = true;
    float max_cdiff = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float cd = std::fabs(got[i].conf - ref_conf[i]);
        if (cd > max_cdiff) max_cdiff = cd;
        if (got[i].id != ref_ids[i]) {
            std::fprintf(stderr, "[ts3:%s] token id @%zu got=%d ref=%d\n", tag, i, got[i].id, ref_ids[i]);
            ok = false;
        }
        if (got[i].frame != ref_frames[i]) {
            std::fprintf(stderr, "[ts3:%s] token frame @%zu got=%d ref=%d (id=%d)\n",
                         tag, i, got[i].frame, ref_frames[i], got[i].id);
            ok = false;
        }
        if (cd > conf_tol) {
            std::fprintf(stderr, "[ts3:%s] token conf @%zu got=%.6f ref=%.6f diff=%.6f\n",
                         tag, i, got[i].conf, ref_conf[i], cd);
            ok = false;
        }
    }
    std::fprintf(stderr, "[ts3:%s] %zu tokens, max conf diff = %.3e (tol %.1e) -> %s\n",
                 tag, got.size(), max_cdiff, conf_tol, ok ? "ok" : "FAIL");
    return ok;
}

// Word-level parity: text exact, start/end within frame_sec, conf within 1e-3.
bool compare_words(const char* tag,
                   const std::vector<pk::Word>& got,
                   const std::vector<RefWord>& ref,
                   float frame_sec) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[ts3:%s] word COUNT mismatch got=%zu ref=%zu\n",
                     tag, got.size(), ref.size());
        for (size_t i = 0; i < got.size() || i < ref.size(); ++i) {
            const char* g = i < got.size() ? got[i].text.c_str() : "<none>";
            const char* r = i < ref.size() ? ref[i].w.c_str()    : "<none>";
            std::fprintf(stderr, "    @%zu got=%-14s ref=%s\n", i, g, r);
        }
        return false;
    }
    bool ok = true;
    float max_sdiff = 0.0f, max_ediff = 0.0f, max_cdiff = 0.0f;
    const float time_tol = frame_sec + 1e-4f; // within one frame
    const float conf_tol = 1e-3f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float sd = std::fabs(got[i].start - (float)ref[i].start);
        const float ed = std::fabs(got[i].end   - (float)ref[i].end);
        const float cd = std::fabs(got[i].conf  - (float)ref[i].conf);
        if (sd > max_sdiff) max_sdiff = sd;
        if (ed > max_ediff) max_ediff = ed;
        if (cd > max_cdiff) max_cdiff = cd;
        if (got[i].text != ref[i].w) {
            std::fprintf(stderr, "[ts3:%s] word TEXT @%zu got=%s ref=%s\n",
                         tag, i, got[i].text.c_str(), ref[i].w.c_str());
            ok = false;
        }
        if (sd > time_tol) {
            std::fprintf(stderr, "[ts3:%s] word START @%zu got=%.4f ref=%.4f diff=%.4f (>%.4f) [%s]\n",
                         tag, i, got[i].start, (float)ref[i].start, sd, time_tol, ref[i].w.c_str());
            ok = false;
        }
        if (ed > time_tol) {
            std::fprintf(stderr, "[ts3:%s] word END @%zu got=%.4f ref=%.4f diff=%.4f (>%.4f) [%s]\n",
                         tag, i, got[i].end, (float)ref[i].end, ed, time_tol, ref[i].w.c_str());
            ok = false;
        }
        if (cd > conf_tol) {
            std::fprintf(stderr, "[ts3:%s] word CONF @%zu got=%.6f ref=%.6f diff=%.6f (>%.1e) [%s]\n",
                         tag, i, got[i].conf, (float)ref[i].conf, cd, conf_tol, ref[i].w.c_str());
            ok = false;
        }
    }
    std::fprintf(stderr,
        "[ts3:%s] %zu words: max start diff=%.4f end diff=%.4f conf diff=%.3e "
        "(time tol %.4f, conf tol %.1e) -> %s\n",
        tag, got.size(), max_sdiff, max_ediff, max_cdiff, time_tol, conf_tol,
        ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main() {
    const char* gguf_path = std::getenv("PARAKEET_TEST_GGUF");
    const char* base_path = std::getenv("PARAKEET_TEST_BASELINE_TS");
    if (!gguf_path || !base_path) {
        std::fprintf(stderr, "test_timestamps: PARAKEET_TEST_GGUF / "
                             "PARAKEET_TEST_BASELINE_TS not set; skip\n");
        return 77;
    }

    try {
        std::unique_ptr<pk::Model> model = pk::Model::load(gguf_path);
        if (!model) { std::fprintf(stderr, "[ts3] model load failed\n"); return 1; }

        // ---- Open the baseline gguf (tensors materialised into bctx) ----
        ggml_context* bctx = nullptr;
        gguf_init_params p{ /*no_alloc=*/false, &bctx };
        gguf_context* bg = gguf_init_from_file(base_path, p);
        if (!bg) { std::fprintf(stderr, "[ts3] open baseline failed: %s\n", base_path); return 1; }

        int64_t fs_kid = gguf_find_key(bg, "baseline.frame_sec");
        if (fs_kid < 0) { std::fprintf(stderr, "[ts3] baseline.frame_sec missing\n"); gguf_free(bg); ggml_free(bctx); return 1; }
        const float frame_sec = gguf_get_val_f32(bg, fs_kid);
        std::fprintf(stderr, "[ts3] baseline.frame_sec = %.6f\n", frame_sec);

        std::vector<int32_t> tdt_ids, tdt_frames, ctc_ids, ctc_frames;
        std::vector<float>   tdt_conf, ctc_conf;
        std::vector<RefWord> tdt_words, ctc_words;
        bool read_ok =
            read_i32(bctx, "ts_tdt_token_ids",    tdt_ids) &&
            read_i32(bctx, "ts_tdt_token_frames", tdt_frames) &&
            read_f32(bctx, "ts_tdt_token_conf",   tdt_conf) &&
            read_i32(bctx, "ts_ctc_token_ids",    ctc_ids) &&
            read_i32(bctx, "ts_ctc_token_frames", ctc_frames) &&
            read_f32(bctx, "ts_ctc_token_conf",   ctc_conf) &&
            read_words_json(bg, "baseline.tdt_words_json", tdt_words) &&
            read_words_json(bg, "baseline.ctc_words_json", ctc_words);
        gguf_free(bg);
        if (!read_ok) { ggml_free(bctx); return 1; }

        bool all_ok = true;

        // ---- TDT head ----
        {
            pk::Transcription tr =
                model->transcribe_path_with_timestamps("tests/fixtures/speech.wav", pk::Decoder::kTDT);
            std::fprintf(stderr, "[ts3:tdt] text: %s\n", tr.text.c_str());
            all_ok &= compare_tokens("tdt", tr.tokens, tdt_ids, tdt_frames, tdt_conf, 1e-3f);
            all_ok &= compare_words("tdt", tr.words, tdt_words, frame_sec);
        }

        // ---- CTC head ----
        {
            pk::Transcription tr =
                model->transcribe_path_with_timestamps("tests/fixtures/speech.wav", pk::Decoder::kCTC);
            std::fprintf(stderr, "[ts3:ctc] text: %s\n", tr.text.c_str());
            all_ok &= compare_tokens("ctc", tr.tokens, ctc_ids, ctc_frames, ctc_conf, 1e-3f);
            all_ok &= compare_words("ctc", tr.words, ctc_words, frame_sec);
        }

        ggml_free(bctx);
        if (!all_ok) { std::fprintf(stderr, "[ts3] FAIL\n"); return 1; }
        std::fprintf(stderr, "[ts3] PASS — token + word timestamps/confidence match "
                             "NeMo for both TDT and CTC heads\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ts3] threw: %s\n", e.what());
        return 1;
    }
}
