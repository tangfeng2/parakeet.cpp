#include "parakeet.h"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

// Committed regression test for the RNNT greedy decoder path.
//
// Locks the C++ RNNT/transducer head against regression on the real
// nvidia/parakeet-rnnt-0.6b checkpoint.  This model has arch=rnnt,
// xscaling=True, no TDT duration table — so pk::Decoder::kTDT routes to
// pk::rnnt_greedy (cfg.tdt_durations is empty).  The reference string is the
// NeMo RNNT transcript of tests/fixtures/speech.wav, validated at WER 0 in
// Phase 3.5 Task 2.
//
// The model GGUF is a ~2.4GB download not present in CI, so the test skips
// cleanly (exit 77) unless PARAKEET_TEST_GGUF_RNNT points to a converted GGUF.
//
// Note on decoder flag: pk::Decoder::kTDT selects the transducer head.  For a
// model with no duration table (parakeet-rnnt-*), transcribe() routes this to
// pk::rnnt_greedy; for a model with a duration table (parakeet-tdt-*), it
// routes to pk::tdt_greedy.  Using kTDT here exercises exactly the rnnt_greedy
// path in a model-realistic way — the same routing the CLI --decoder tdt flag
// takes for RNNT models.
//
// Env:
//   PARAKEET_TEST_GGUF_RNNT   path to the converted parakeet-rnnt-0.6b GGUF
//                             (skip 77 if unset)
//   PARAKEET_TEST_RNNT_TEXT   optional override for the expected transcript;
//                             if unset, the validated NeMo RNNT reference is used.
//
// LABEL model
// WORKING_DIRECTORY (run from project root; wav path is relative)

// NeMo RNNT reference for nvidia/parakeet-rnnt-0.6b on speech.wav (verbatim).
// Pure-RNNT models emit lowercase, unpunctuated text (same training convention
// as the standalone CTC models — distinct from the cased/punctuated TDT/hybrid
// checkpoints).
static const char* kRefRNNT =
    "well i don't wish to see it any more observed phoebe turning away her "
    "eyes it is certainly very like the old portrait";

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_RNNT");
    if (!gguf) {
        std::fprintf(stderr,
            "test_transcribe_rnnt: PARAKEET_TEST_GGUF_RNNT not set; skip "
            "(parakeet-rnnt-0.6b is a ~2.4GB download, not in CI)\n");
        return 77;
    }

    const char* env_expected = std::getenv("PARAKEET_TEST_RNNT_TEXT");
    const std::string expected = env_expected ? std::string(env_expected)
                                              : std::string(kRefRNNT);

    std::string got;
    try {
        // kTDT selects the transducer head; routes to rnnt_greedy because
        // parakeet-rnnt-0.6b has no TDT duration table.
        got = pk::transcribe(gguf, "tests/fixtures/speech.wav", pk::Decoder::kTDT);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_transcribe_rnnt: pk::transcribe threw: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "test_transcribe_rnnt: got      = %s\n", got.c_str());
    std::fprintf(stderr, "test_transcribe_rnnt: expected = %s\n", expected.c_str());

    if (got != expected) {
        std::fprintf(stderr,
            "test_transcribe_rnnt: MISMATCH vs NeMo RNNT reference\n"
            "  got:      %s\n"
            "  expected: %s\n",
            got.c_str(), expected.c_str());
        return 1;
    }

    std::fprintf(stderr, "test_transcribe_rnnt: PASS (word-for-word match with NeMo RNNT)\n");
    return 0;
}
