#include "parakeet.h"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

// North-star end-to-end TDT transcription test for the real
// nvidia/parakeet-tdt-0.6b-v2 / -v3 checkpoints.
//
// These checkpoints differ from the 110m anchor in ways the metadata-driven
// code must honour: d_model=1024 / 24 layers / 128 mels, FastConformer linears
// and conv convolutions configured with bias=False, and a STACKED 2-layer
// prediction LSTM (pred_rnn_layers=2). This test asserts the C++ TDT path
// reproduces NeMo's transcript of tests/fixtures/speech.wav word-for-word.
//
// The model GGUF is a ~2.4GB download not present in CI, so the test skips
// cleanly (exit 77) unless PARAKEET_TEST_GGUF_06B points to a converted GGUF.
//
// Env:
//   PARAKEET_TEST_GGUF_06B   path to the converted 0.6b GGUF (skip 77 if unset)
//   PARAKEET_TEST_06B_TEXT   optional override for the expected transcript;
//                            if unset, the v2 reference below is used (both v2
//                            and v3 produce the same words on this English clip,
//                            differing only in the spelling "Phebe"/"Phoebe").
//
// LABEL model
// WORKING_DIRECTORY (run from project root; wav path is relative)

// NeMo TDT reference for nvidia/parakeet-tdt-0.6b-v2 on speech.wav (verbatim).
static const char* kRefV2 =
    "Well, I don't wish to see it any more, observed Phebe, turning away her "
    "eyes. It is certainly very like the old portrait.";

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_06B");
    if (!gguf) {
        std::fprintf(stderr,
            "test_transcribe_0_6b: PARAKEET_TEST_GGUF_06B not set; skip "
            "(0.6b model is a ~2.4GB download, not in CI)\n");
        return 77;
    }

    const char* env_expected = std::getenv("PARAKEET_TEST_06B_TEXT");
    const std::string expected = env_expected ? std::string(env_expected)
                                              : std::string(kRefV2);

    std::string got;
    try {
        got = pk::transcribe(gguf, "tests/fixtures/speech.wav");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_transcribe_0_6b: pk::transcribe threw: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "test_transcribe_0_6b: got      = %s\n", got.c_str());
    std::fprintf(stderr, "test_transcribe_0_6b: expected = %s\n", expected.c_str());

    if (got != expected) {
        std::fprintf(stderr,
            "test_transcribe_0_6b: MISMATCH vs NeMo TDT reference\n"
            "  got:      %s\n"
            "  expected: %s\n",
            got.c_str(), expected.c_str());
        return 1;
    }

    std::fprintf(stderr, "test_transcribe_0_6b: PASS (word-for-word match with NeMo TDT)\n");
    return 0;
}
