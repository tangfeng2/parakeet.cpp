#include "conformer.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>

// Parity test for the streaming conv module (layer_norm + causal depthwise conv)
// in ISOLATION, against NeMo's nvidia/parakeet_realtime_eou_120m-v1.
//
// The baseline `l0_conv_in` is the INPUT fed into layers[0].conv (= the conv
// module's norm_conv(residual), captured via a forward-pre-hook). We run JUST
// the ConformerConvolution sub-module (pointwise_conv1 -> GLU -> depthwise_conv
// -> LayerNorm-over-channels -> SiLU -> pointwise_conv2, with CAUSAL left-pad
// k-1) on it and assert the output equals the baseline `l0_conv_out`.
//
// Running the conv module on its own (rather than the whole conformer layer)
// decouples this check from the chunked-limited attention (a later task): a
// mismatch here localizes to the layer_norm axis/eps or the causal padding.
//
// Skips (77) unless both PARAKEET_TEST_GGUF_EOU (=/tmp/eou.gguf) and
// PARAKEET_TEST_BASELINE_EOU (=/tmp/baseline_eou.gguf) are set.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_EOU");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_EOU");
    if (!gguf || !base) { std::fprintf(stderr, "eou env not set; skip\n"); return 77; }

    pk::ModelLoader ml;
    if (!ml.load(gguf)) return 1;

    // Sanity: this baseline is only meaningful for the streaming variant.
    if (ml.config().conv_norm_type != "layer_norm" || !ml.config().conv_causal) {
        std::fprintf(stderr,
            "test_conv_eou expects conv_norm_type=layer_norm + conv_causal "
            "(got '%s', causal=%d)\n",
            ml.config().conv_norm_type.c_str(), (int)ml.config().conv_causal);
        return 1;
    }

    // Conv input: baseline "l0_conv_in" is [T, d_model] row-major.
    std::vector<float> cin; std::vector<int64_t> cinshape;
    if (!pktest::load_baseline(base, "l0_conv_in", cin, cinshape)) return 1;
    if (cinshape.size() != 2) {
        std::fprintf(stderr, "l0_conv_in rank=%zu\n", cinshape.size());
        return 1;
    }
    const int T       = (int)cinshape[0];
    const int d_model = (int)cinshape[1];
    if (d_model != (int)ml.config().d_model) {
        std::fprintf(stderr, "l0_conv_in d_model=%d != cfg d_model=%u\n",
                     d_model, ml.config().d_model);
        return 1;
    }

    // All encoder frames are valid for the clip (enc_len == T'), so no pad mask.
    const int valid_len = T;

    pk::ConformerLayer layer(ml, /*layer_idx*/0);
    std::vector<float> conv_out;
    layer.conv_module_forward(cin, T, valid_len, conv_out);

    // Reference: l0_conv_out is [T, d_model] row-major.
    std::vector<float> ref; std::vector<int64_t> rshape;
    if (!pktest::load_baseline(base, "l0_conv_out", ref, rshape)) return 1;
    if (rshape.size() != 2 || (int)rshape[0] != T || (int)rshape[1] != d_model) {
        std::fprintf(stderr, "l0_conv_out shape=[%lld,%lld] expected [%d,%d]\n",
                     (long long)rshape[0], (long long)rshape[1], T, d_model);
        return 1;
    }

    bool ok = pktest::compare(conv_out, ref, "conv_eou", /*atol*/3e-2f, /*rtol*/3e-2f);
    return ok ? 0 : 1;
}
