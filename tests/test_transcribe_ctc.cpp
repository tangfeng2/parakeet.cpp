#include "parakeet.h"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

// Committed regression test for the standalone CTC decoder path.
//
// Locks the C++ CTC head against regression on the real
// nvidia/parakeet-ctc-0.6b checkpoint.  This model differs from the 110m
// anchor: d_model=1024, 24 layers, xscaling=True, standalone CTC head at
// decoder.decoder_layers.0.* (not ctc_decoder.*).  The reference string is the
// NeMo CTC transcript of tests/fixtures/speech.wav, validated at WER 0 in
// Phase 3.5 Task 1.
//
// The model GGUF is a ~2.4GB download not present in CI, so the test skips
// cleanly (exit 77) unless PARAKEET_TEST_GGUF_CTC points to a converted GGUF.
//
// Env:
//   PARAKEET_TEST_GGUF_CTC   path to the converted parakeet-ctc-0.6b GGUF
//                            (skip 77 if unset)
//   PARAKEET_TEST_CTC_TEXT   optional override for the expected transcript;
//                            if unset, the validated NeMo CTC reference is used.
//
// LABEL model
// WORKING_DIRECTORY (run from project root; wav path is relative)

// NeMo CTC reference for nvidia/parakeet-ctc-0.6b on speech.wav (verbatim).
// Standalone CTC models emit lowercase, unpunctuated text (their own
// tokenizer/training convention — distinct from the cased/punctuated TDT/hybrid
// models).
static const char* kRefCTC =
    "well i don't wish to see it any more observed phoebe turning away her "
    "eyes it is certainly very like the old portrait";

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_CTC");
    if (!gguf) {
        std::fprintf(stderr,
            "test_transcribe_ctc: PARAKEET_TEST_GGUF_CTC not set; skip "
            "(parakeet-ctc-0.6b is a ~2.4GB download, not in CI)\n");
        return 77;
    }

    const char* env_expected = std::getenv("PARAKEET_TEST_CTC_TEXT");
    const std::string expected = env_expected ? std::string(env_expected)
                                              : std::string(kRefCTC);

    std::string got;
    try {
        got = pk::transcribe(gguf, "tests/fixtures/speech.wav", pk::Decoder::kCTC);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_transcribe_ctc: pk::transcribe threw: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "test_transcribe_ctc: got      = %s\n", got.c_str());
    std::fprintf(stderr, "test_transcribe_ctc: expected = %s\n", expected.c_str());

    if (got != expected) {
        std::fprintf(stderr,
            "test_transcribe_ctc: MISMATCH vs NeMo CTC reference\n"
            "  got:      %s\n"
            "  expected: %s\n",
            got.c_str(), expected.c_str());
        return 1;
    }

    std::fprintf(stderr, "test_transcribe_ctc: PASS (word-for-word match with NeMo CTC)\n");
    return 0;
}
