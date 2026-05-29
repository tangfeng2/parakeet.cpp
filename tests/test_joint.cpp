#include "joint.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

// Parity test for the RNN-Transducer joint network vs NeMo RNNTJoint.joint.
//
// NeMo math (RNNTJoint.joint_after_projection):
//   enc_proj[t]  = joint.enc.weight  · enc[t]  + enc.bias   (512->640)
//   pred_proj[u] = joint.pred.weight · pred[u] + pred.bias  (640->640)
//   f[t,u]       = ReLU(enc_proj[t] + pred_proj[u])         (ReLU on SUM)
//   logits[t,u]  = joint_net.2.weight · f[t,u] + joint_net.2.bias  (640->1030)
//
// Baseline:
//   encoder_out      f32 [d_model=512, T=26]  channels-first (enc_out[c*T + t])
//   pred_out         f32 [U=5, pred_hidden=640]  row-major
//   joint_out        f32 [T_slice=4, U=5, 1030]  raw logits (NO log_softmax)
//   joint_enc_frames i32 [1] = 4  (number of encoder frames used)
//
// The test:
//   1. Read joint_enc_frames to get T_slice.
//   2. Slice encoder_out columns [0..T_slice) → transpose to row-major [T_slice, 512].
//   3. Read pred_out [U=5, 640] row-major.
//   4. Run pk::Joint::forward(enc_slice, T_slice, 512, pred_out, 5, 640, logits, V_plus).
//   5. Compare logits to joint_out (atol/rtol 5e-3).
int main() {
    const char* gguf_path = std::getenv("PARAKEET_TEST_GGUF");
    const char* base_path = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf_path || !base_path) {
        std::fprintf(stderr, "env not set; skip\n");
        return 77;
    }

    pk::ModelLoader ml;
    if (!ml.load(gguf_path)) return 1;

    // ---- 1. Read joint_enc_frames (i32 scalar) from baseline ----
    int T_slice = 0;
    {
        ggml_context* ctx = nullptr;
        gguf_init_params p{ /*no_alloc=*/false, &ctx };
        gguf_context* g = gguf_init_from_file(base_path, p);
        if (!g) { std::fprintf(stderr, "[joint] open baseline failed\n"); return 1; }
        ggml_tensor* t = ggml_get_tensor(ctx, "joint_enc_frames");
        if (!t) { std::fprintf(stderr, "[joint] joint_enc_frames missing\n"); gguf_free(g); ggml_free(ctx); return 1; }
        // Stored as i32 (gguf type 26)
        if (t->type == GGML_TYPE_I32) {
            T_slice = ((const int32_t*)t->data)[0];
        } else if (t->type == GGML_TYPE_F32) {
            T_slice = (int)((const float*)t->data)[0];
        } else {
            std::fprintf(stderr, "[joint] joint_enc_frames unexpected type %d\n", (int)t->type);
            gguf_free(g); ggml_free(ctx); return 1;
        }
        gguf_free(g); ggml_free(ctx);
    }
    std::fprintf(stderr, "[joint] T_slice=%d\n", T_slice);

    // ---- 2. Read encoder_out [d_model, T_full] channels-first, slice to T_slice ----
    // load_baseline reports shape outer..inner = [ne[1], ne[0]] = [d_model, T_full].
    // Memory layout: enc_data[c*T_full + t]
    std::vector<float> enc_raw; std::vector<int64_t> eshape;
    if (!pktest::load_baseline(base_path, "encoder_out", enc_raw, eshape)) return 1;
    if (eshape.size() != 2) { std::fprintf(stderr, "[joint] encoder_out rank=%zu\n", eshape.size()); return 1; }
    const int d_model = (int)eshape[0];   // ne[1] = d_model = 512
    const int T_full  = (int)eshape[1];   // ne[0] = T = 26
    std::fprintf(stderr, "[joint] encoder_out: d_model=%d T_full=%d\n", d_model, T_full);
    if (T_slice > T_full) { std::fprintf(stderr, "[joint] T_slice>T_full\n"); return 1; }

    // Transpose channels-first [d_model, T_full] to row-major [T_slice, d_model].
    // enc_raw[c*T_full + t] → enc_slice[t*d_model + c]
    std::vector<float> enc_slice((size_t)T_slice * d_model);
    for (int t = 0; t < T_slice; ++t)
        for (int c = 0; c < d_model; ++c)
            enc_slice[(size_t)t * d_model + c] = enc_raw[(size_t)c * T_full + t];

    // ---- 3. Read pred_out [U, pred_hidden] row-major ----
    // load_baseline shape outer..inner = [ne[1], ne[0]] = [U, pred_hidden]
    std::vector<float> pred_out; std::vector<int64_t> pshape;
    if (!pktest::load_baseline(base_path, "pred_out", pred_out, pshape)) return 1;
    if (pshape.size() != 2) { std::fprintf(stderr, "[joint] pred_out rank=%zu\n", pshape.size()); return 1; }
    const int U           = (int)pshape[0];  // ne[1] = U = 5
    const int pred_hidden = (int)pshape[1];  // ne[0] = pred_hidden = 640
    std::fprintf(stderr, "[joint] pred_out: U=%d pred_hidden=%d\n", U, pred_hidden);

    // ---- 4. Run joint forward ----
    pk::Joint joint(ml);
    std::vector<float> logits; int V_plus = 0;
    joint.forward(enc_slice, T_slice, d_model, pred_out, U, pred_hidden, logits, V_plus);
    std::fprintf(stderr, "[joint] output: T=%d U=%d V_plus=%d (n=%zu)\n",
                 T_slice, U, V_plus, logits.size());

    // ---- 5. Read joint_out baseline [T_slice, U, V_plus] row-major ----
    // load_baseline shape outer..inner = [ne[2], ne[1], ne[0]] = [T_slice, U, V_plus]
    std::vector<float> ref; std::vector<int64_t> jshape;
    if (!pktest::load_baseline(base_path, "joint_out", ref, jshape)) return 1;
    if (jshape.size() != 3) { std::fprintf(stderr, "[joint] joint_out rank=%zu\n", jshape.size()); return 1; }
    const int ref_T = (int)jshape[0];
    const int ref_U = (int)jshape[1];
    const int ref_V = (int)jshape[2];
    std::fprintf(stderr, "[joint] ref joint_out: T=%d U=%d V=%d (n=%zu)\n", ref_T, ref_U, ref_V, ref.size());

    if (ref_T != T_slice || ref_U != U || ref_V != V_plus) {
        std::fprintf(stderr, "[joint] shape mismatch got [%d,%d,%d] ref [%d,%d,%d]\n",
                     T_slice, U, V_plus, ref_T, ref_U, ref_V);
        return 1;
    }

    bool ok = pktest::compare(logits, ref, "joint", /*atol=*/5e-3f, /*rtol=*/5e-3f);
    return ok ? 0 : 1;
}
