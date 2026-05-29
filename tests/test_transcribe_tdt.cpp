#include "parakeet.h"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

// End-to-end TDT transcription test.
//
// Runs pk::transcribe on tests/fixtures/speech.wav using the arch-default
// decoder (which routes to the TDT head for hybrid_tdt_ctc / tdt archs —
// matching NeMo's default cur_decoder='rnnt').
//
// The expected string is read from `baseline.tdt_text` in the speech baseline
// gguf produced by scripts/gen_nemo_baseline.py (the authoritative NeMo TDT
// greedy transcript), and the test asserts an exact match (WER 0).
//
// LABEL model
// WORKING_DIRECTORY (tests run from the project root; wav path is relative)
//
// Env:
//   PARAKEET_TEST_GGUF               model weights (skip 77 if unset)
//   PARAKEET_TEST_BASELINE_SPEECH    speech baseline gguf with baseline.tdt_text
//                                    (skip 77 if unset)

static std::string read_string_kv(const char* path, const char* key_name) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf_context* g = gguf_init_from_file(path, p);
    if (!g) {
        std::fprintf(stderr, "test_transcribe_tdt: failed to open baseline: %s\n", path);
        return "";
    }
    int64_t ki = gguf_find_key(g, key_name);
    std::string result;
    if (ki >= 0) {
        const char* s = gguf_get_val_str(g, ki);
        if (s) result = s;
    } else {
        std::fprintf(stderr, "test_transcribe_tdt: key '%s' not found in %s\n",
                     key_name, path);
    }
    gguf_free(g);
    ggml_free(ctx);
    return result;
}

int main() {
    const char* gguf  = std::getenv("PARAKEET_TEST_GGUF");
    const char* bpath = std::getenv("PARAKEET_TEST_BASELINE_SPEECH");
    if (!gguf || !bpath) {
        std::fprintf(stderr,
            "test_transcribe_tdt: PARAKEET_TEST_GGUF / "
            "PARAKEET_TEST_BASELINE_SPEECH not set; skip\n");
        return 77;
    }

    // Read the authoritative NeMo TDT reference from the speech baseline.
    const std::string expected = read_string_kv(bpath, "baseline.tdt_text");
    if (expected.empty()) {
        std::fprintf(stderr, "test_transcribe_tdt: baseline.tdt_text is empty or missing\n");
        return 1;
    }
    std::fprintf(stderr, "test_transcribe_tdt: expected = %s\n", expected.c_str());

    // Run the C++ TDT path. pk::transcribe with kDefault routes to TDT for the
    // hybrid_tdt_ctc arch of parakeet-tdt_ctc-110m.
    std::string got;
    try {
        got = pk::transcribe(gguf, "tests/fixtures/speech.wav");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_transcribe_tdt: pk::transcribe threw: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "test_transcribe_tdt: got      = %s\n", got.c_str());

    if (got != expected) {
        std::fprintf(stderr,
            "test_transcribe_tdt: MISMATCH vs NeMo TDT reference\n"
            "  got:      %s\n"
            "  expected: %s\n",
            got.c_str(), expected.c_str());
        return 1;
    }

    std::fprintf(stderr, "test_transcribe_tdt: PASS (word-for-word match with NeMo TDT)\n");
    return 0;
}
