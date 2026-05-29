#include "prediction.hpp"
#include "joint.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>

// Transducer-core integration parity test.
//
// Composes pk::PredictionNet::forward and pk::Joint::forward end-to-end and
// compares the composed C++ output to the NeMo baseline joint_out.  This
// catches interface-seam bugs (e.g. orientation mismatches between the
// prediction net's output and the joint's pred input) that the individual
// unit tests cannot.
//
// Pipeline:
//   pred_input_ids  [U=4]           (int32, from baseline)
//   encoder_out     [d_model, T]    (f32 channels-first, from baseline)
//   joint_enc_frames               (int32 scalar, from baseline)
//
//   Step A: pk::PredictionNet::forward(ids, add_sos=true) → pred [U+1=5, 640]
//           (uses C++ pred output, NOT baseline pred_out)
//   Step B: transpose encoder_out[:, 0..T_slice) to row-major [T_slice, d_model]
//   Step C: pk::Joint::forward(enc_slice, T_slice, d_model,
//                               pred, U+1, 640, logits, V_plus)
//   Step D: compare logits to baseline joint_out [T_slice, U+1, V_plus]
//           atol/rtol 5e-3

int main() {
    const char* gguf_path = std::getenv("PARAKEET_TEST_GGUF");
    const char* base_path = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf_path || !base_path) {
        std::fprintf(stderr, "env not set; skip\n");
        return 77;
    }

    pk::ModelLoader ml;
    if (!ml.load(gguf_path)) return 1;

    // ---- Load pred_input_ids (int32) from baseline ----
    std::vector<int32_t> ids;
    {
        ggml_context* ctx = nullptr;
        gguf_init_params p{ /*no_alloc=*/false, &ctx };
        gguf_context* g = gguf_init_from_file(base_path, p);
        if (!g) { std::fprintf(stderr, "[transducer] open baseline failed\n"); return 1; }
        ggml_tensor* t = ggml_get_tensor(ctx, "pred_input_ids");
        if (!t) { std::fprintf(stderr, "[transducer] pred_input_ids missing\n"); gguf_free(g); ggml_free(ctx); return 1; }
        const int n = (int)ggml_nelements(t);
        ids.resize(n);
        if (t->type == GGML_TYPE_I32) {
            std::memcpy(ids.data(), t->data, (size_t)n * sizeof(int32_t));
        } else if (t->type == GGML_TYPE_F32) {
            const float* d = (const float*)t->data;
            for (int i = 0; i < n; ++i) ids[i] = (int32_t)d[i];
        } else {
            std::fprintf(stderr, "[transducer] pred_input_ids unexpected type\n");
            gguf_free(g); ggml_free(ctx); return 1;
        }
        gguf_free(g); ggml_free(ctx);
    }
    std::fprintf(stderr, "[transducer] pred_input_ids (U=%zu):", ids.size());
    for (int32_t v : ids) std::fprintf(stderr, " %d", v);
    std::fprintf(stderr, "\n");

    // ---- Load joint_enc_frames (int32 scalar) from baseline ----
    int T_slice = 0;
    {
        ggml_context* ctx = nullptr;
        gguf_init_params p{ /*no_alloc=*/false, &ctx };
        gguf_context* g = gguf_init_from_file(base_path, p);
        if (!g) { std::fprintf(stderr, "[transducer] open baseline failed\n"); return 1; }
        ggml_tensor* t = ggml_get_tensor(ctx, "joint_enc_frames");
        if (!t) { std::fprintf(stderr, "[transducer] joint_enc_frames missing\n"); gguf_free(g); ggml_free(ctx); return 1; }
        if (t->type == GGML_TYPE_I32) {
            T_slice = ((const int32_t*)t->data)[0];
        } else if (t->type == GGML_TYPE_F32) {
            T_slice = (int)((const float*)t->data)[0];
        } else {
            std::fprintf(stderr, "[transducer] joint_enc_frames unexpected type %d\n", (int)t->type);
            gguf_free(g); ggml_free(ctx); return 1;
        }
        gguf_free(g); ggml_free(ctx);
    }
    std::fprintf(stderr, "[transducer] T_slice=%d\n", T_slice);

    // ---- Step A: Run prediction net with add_sos=true (C++ output) ----
    pk::PredictionNet prednet(ml);
    std::vector<float> pred_cpp; int U_out = 0, pred_hidden = 0;
    prednet.forward(ids, /*add_sos=*/true, pred_cpp, U_out, pred_hidden);
    std::fprintf(stderr, "[transducer] pred_cpp: U_out=%d hidden=%d (n=%zu)\n",
                 U_out, pred_hidden, pred_cpp.size());

    // ---- Step B: Load encoder_out [d_model, T_full] channels-first, slice ----
    // load_baseline reports shape outer..inner = [ne[1], ne[0]] = [d_model, T_full]
    // Memory: enc_raw[c * T_full + t]
    std::vector<float> enc_raw; std::vector<int64_t> eshape;
    if (!pktest::load_baseline(base_path, "encoder_out", enc_raw, eshape)) return 1;
    if (eshape.size() != 2) { std::fprintf(stderr, "[transducer] encoder_out rank=%zu\n", eshape.size()); return 1; }
    const int d_model = (int)eshape[0];  // ne[1] = d_model = 512
    const int T_full  = (int)eshape[1];  // ne[0] = T = 26
    std::fprintf(stderr, "[transducer] encoder_out: d_model=%d T_full=%d\n", d_model, T_full);
    if (T_slice > T_full) { std::fprintf(stderr, "[transducer] T_slice>T_full\n"); return 1; }

    // Transpose channels-first to row-major [T_slice, d_model]
    // enc_raw[c * T_full + t] → enc_slice[t * d_model + c]
    std::vector<float> enc_slice((size_t)T_slice * d_model);
    for (int t = 0; t < T_slice; ++t)
        for (int c = 0; c < d_model; ++c)
            enc_slice[(size_t)t * d_model + c] = enc_raw[(size_t)c * T_full + t];

    // ---- Step C: Run joint forward with C++ pred output ----
    pk::Joint joint(ml);
    std::vector<float> logits; int V_plus = 0;
    joint.forward(enc_slice, T_slice, d_model, pred_cpp, U_out, pred_hidden, logits, V_plus);
    std::fprintf(stderr, "[transducer] joint output: T=%d U=%d V_plus=%d (n=%zu)\n",
                 T_slice, U_out, V_plus, logits.size());

    // ---- Step D: Load joint_out baseline and compare ----
    // load_baseline shape outer..inner = [ne[2], ne[1], ne[0]] = [T_slice, U, V_plus]
    std::vector<float> ref; std::vector<int64_t> jshape;
    if (!pktest::load_baseline(base_path, "joint_out", ref, jshape)) return 1;
    if (jshape.size() != 3) { std::fprintf(stderr, "[transducer] joint_out rank=%zu\n", jshape.size()); return 1; }
    const int ref_T = (int)jshape[0];
    const int ref_U = (int)jshape[1];
    const int ref_V = (int)jshape[2];
    std::fprintf(stderr, "[transducer] ref joint_out: T=%d U=%d V=%d (n=%zu)\n",
                 ref_T, ref_U, ref_V, ref.size());

    if (ref_T != T_slice || ref_U != U_out || ref_V != V_plus) {
        std::fprintf(stderr, "[transducer] shape mismatch: got [%d,%d,%d] ref [%d,%d,%d]\n",
                     T_slice, U_out, V_plus, ref_T, ref_U, ref_V);
        return 1;
    }

    bool ok = pktest::compare(logits, ref, "transducer_core", /*atol=*/5e-3f, /*rtol=*/5e-3f);
    return ok ? 0 : 1;
}
