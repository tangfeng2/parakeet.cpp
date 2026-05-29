#include "model_loader.hpp"
#include "audio_io.hpp"
#include "mel.hpp"
#include "encoder.hpp"
#include "ctc_decoder.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "tdt.hpp"
#include "search.hpp"
#include "decode_types.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

// Per-token timestamp + confidence parity vs NeMo (Task 2 of the
// timestamps+confidence plan).
//
// On tests/fixtures/speech.wav + the parakeet-tdt_ctc-110m anchor model, run the
// full C++ encoder and BOTH greedy heads producing std::vector<TokenInfo>, then
// compare per-token {id, frame, conf} against the NeMo baseline arrays:
//   * TDT head  (tdt_greedy)  -> ts_tdt_token_{ids,frames,conf}
//   * CTC head  (ctc_greedy)  -> ts_ctc_token_{ids,frames,conf}
// Asserts ids exact, frames exact, conf within 1e-3 (NeMo's rescaled 'max_prob'
// confidence, method alpha=1.0; CTC token conf is the min over the token's
// consecutive argmax run; CTC frame is the run-START / NeMo start_offset).
//
// LABEL model
// WORKING_DIRECTORY (tests run from the project root; the wav path is relative)
//
// Env:
//   PARAKEET_TEST_GGUF          : model weights (skip 77 if unset)
//   PARAKEET_TEST_BASELINE_TS   : timestamps baseline gguf with ts_<head>_token_*
//                                 (skip 77 if unset)

namespace {

// Read an int32 tensor from a baseline gguf into `out`. Returns false on error.
bool read_i32(ggml_context* ctx, const char* name, std::vector<int32_t>& out) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) { std::fprintf(stderr, "[ts] tensor '%s' not found\n", name); return false; }
    const int n = (int)ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_I32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(int32_t));
    } else if (t->type == GGML_TYPE_F32) {
        const float* d = (const float*)t->data;
        for (int i = 0; i < n; ++i) out[i] = (int32_t)d[i];
    } else {
        std::fprintf(stderr, "[ts] tensor '%s' unexpected type %d\n", name, (int)t->type);
        return false;
    }
    return true;
}

// Read an f32 tensor from a baseline gguf into `out`. Returns false on error.
bool read_f32(ggml_context* ctx, const char* name, std::vector<float>& out) {
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    if (!t) { std::fprintf(stderr, "[ts] tensor '%s' not found\n", name); return false; }
    const int n = (int)ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(float));
    } else {
        std::fprintf(stderr, "[ts] tensor '%s' unexpected type %d\n", name, (int)t->type);
        return false;
    }
    return true;
}

// Compare the C++ per-token decode against a NeMo baseline head. Returns true on
// full parity (ids exact, frames exact, conf within `conf_tol`).
bool compare_head(const char* tag,
                  const std::vector<pk::TokenInfo>& got,
                  const std::vector<int32_t>& ref_ids,
                  const std::vector<int32_t>& ref_frames,
                  const std::vector<float>& ref_conf,
                  float conf_tol) {
    std::fprintf(stderr, "[ts:%s] got %zu tokens, ref %zu\n",
                 tag, got.size(), ref_ids.size());
    if (got.size() != ref_ids.size()) {
        std::fprintf(stderr, "[ts:%s] LENGTH MISMATCH got=%zu ref=%zu\n",
                     tag, got.size(), ref_ids.size());
        return false;
    }
    bool ok = true;
    float max_conf_diff = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float cdiff = std::fabs(got[i].conf - ref_conf[i]);
        if (cdiff > max_conf_diff) max_conf_diff = cdiff;
        if (got[i].id != ref_ids[i]) {
            std::fprintf(stderr, "[ts:%s] id mismatch @%zu got=%d ref=%d\n",
                         tag, i, got[i].id, ref_ids[i]);
            ok = false;
        }
        if (got[i].frame != ref_frames[i]) {
            std::fprintf(stderr, "[ts:%s] frame mismatch @%zu got=%d ref=%d (id=%d)\n",
                         tag, i, got[i].frame, ref_frames[i], got[i].id);
            ok = false;
        }
        if (cdiff > conf_tol) {
            std::fprintf(stderr,
                "[ts:%s] conf mismatch @%zu got=%.6f ref=%.6f diff=%.6f (id=%d)\n",
                tag, i, got[i].conf, ref_conf[i], cdiff, got[i].id);
            ok = false;
        }
    }
    std::fprintf(stderr, "[ts:%s] max conf diff = %.3e (tol %.1e) -> %s\n",
                 tag, max_conf_diff, conf_tol, ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main() {
    const char* gguf_path = std::getenv("PARAKEET_TEST_GGUF");
    const char* base_path = std::getenv("PARAKEET_TEST_BASELINE_TS");
    if (!gguf_path || !base_path) {
        std::fprintf(stderr, "test_timestamps_tokens: PARAKEET_TEST_GGUF / "
                             "PARAKEET_TEST_BASELINE_TS not set; skip\n");
        return 77;
    }

    try {
        pk::ModelLoader ml;
        if (!ml.load(gguf_path)) { std::fprintf(stderr, "[ts] model load failed\n"); return 1; }
        const pk::ParakeetConfig& cfg = ml.config();
        const int blank_id    = (int)cfg.blank_id;
        const int max_symbols = (int)cfg.max_symbols;

        // ---- Open the baseline gguf (tensors materialised into ctx) ----
        ggml_context* bctx = nullptr;
        gguf_init_params p{ /*no_alloc=*/false, &bctx };
        gguf_context* bg = gguf_init_from_file(base_path, p);
        if (!bg) { std::fprintf(stderr, "[ts] open baseline failed: %s\n", base_path); return 1; }

        std::vector<int32_t> tdt_ids, tdt_frames, ctc_ids, ctc_frames;
        std::vector<float>   tdt_conf, ctc_conf;
        bool read_ok =
            read_i32(bctx, "ts_tdt_token_ids",    tdt_ids) &&
            read_i32(bctx, "ts_tdt_token_frames", tdt_frames) &&
            read_f32(bctx, "ts_tdt_token_conf",   tdt_conf) &&
            read_i32(bctx, "ts_ctc_token_ids",    ctc_ids) &&
            read_i32(bctx, "ts_ctc_token_frames", ctc_frames) &&
            read_f32(bctx, "ts_ctc_token_conf",   ctc_conf);
        gguf_free(bg);
        if (!read_ok) { ggml_free(bctx); return 1; }

        // ---- Audio -> mel -> encoder (shared by both heads) ----
        pk::Audio audio;
        if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio)) {
            std::fprintf(stderr, "[ts] failed to load speech.wav\n");
            ggml_free(bctx); return 1;
        }
        pk::MelFrontend mel(ml);
        std::vector<float> feats; int n_mels = 0, T = 0;
        mel.compute(audio.samples, feats, n_mels, T);

        pk::Encoder encoder(ml);
        std::vector<float> enc_out; int d_model = 0, Tout = 0;
        encoder.forward(feats, n_mels, T, enc_out, d_model, Tout);
        std::fprintf(stderr, "[ts] encoder_out: d_model=%d Tout=%d\n", d_model, Tout);

        bool all_ok = true;

        // ---- TDT head (110m anchor is TDT) ----
        if (cfg.tdt_durations.empty()) {
            std::fprintf(stderr, "[ts] config has no tdt_durations\n");
            ggml_free(bctx); return 1;
        }
        {
            // Encoder is channels-first [d_model, Tout]; tdt_greedy wants
            // row-major [Tout, d_model].
            std::vector<float> enc_row((size_t)Tout * d_model);
            for (int t = 0; t < Tout; ++t)
                for (int c = 0; c < d_model; ++c)
                    enc_row[(size_t)t * d_model + c] = enc_out[(size_t)c * Tout + t];

            pk::PredictionNet prednet(ml);
            pk::Joint joint(ml);
            std::vector<pk::TokenInfo> toks;
            pk::tdt_greedy(prednet, joint, enc_row, Tout, d_model,
                           cfg.tdt_durations, blank_id, max_symbols, &toks);
            all_ok &= compare_head("tdt", toks, tdt_ids, tdt_frames, tdt_conf, 1e-3f);

            // TDT span = the predicted duration applied to each token. Only
            // asserted when the baseline carries ts_tdt_token_span (newer dumps).
            std::vector<int32_t> tdt_span;
            if (read_i32(bctx, "ts_tdt_token_span", tdt_span)) {
                if (tdt_span.size() != toks.size()) {
                    std::fprintf(stderr, "[ts:tdt] span length mismatch got=%zu ref=%zu\n",
                                 toks.size(), tdt_span.size());
                    all_ok = false;
                } else {
                    for (size_t i = 0; i < toks.size(); ++i) {
                        if (toks[i].span != tdt_span[i]) {
                            std::fprintf(stderr,
                                "[ts:tdt] span mismatch @%zu got=%d ref=%d (id=%d)\n",
                                i, toks[i].span, tdt_span[i], toks[i].id);
                            all_ok = false;
                        }
                    }
                    std::fprintf(stderr, "[ts:tdt] span parity checked (%zu tokens)\n",
                                 toks.size());
                }
            }
        }

        // ---- CTC head ----
        {
            pk::CTCDecoder ctc(ml);
            std::vector<float> logits; int vocab_plus_1 = 0;
            ctc.forward(enc_out, d_model, Tout, logits, vocab_plus_1);
            std::vector<pk::TokenInfo> toks;
            pk::ctc_greedy(logits, Tout, vocab_plus_1, blank_id, &toks);
            all_ok &= compare_head("ctc", toks, ctc_ids, ctc_frames, ctc_conf, 1e-3f);
        }

        ggml_free(bctx);
        if (!all_ok) { std::fprintf(stderr, "[ts] FAIL\n"); return 1; }
        std::fprintf(stderr, "[ts] PASS — per-token id/frame/conf match NeMo "
                             "for both TDT and CTC heads\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ts] threw: %s\n", e.what());
        return 1;
    }
}
