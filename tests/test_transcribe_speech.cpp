#include "parakeet.h"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

// Real-speech end-to-end CTC transcription test.
//
// Unlike test_transcribe (which uses the synthetic-tone clip whose NeMo CTC
// transcript is the empty string), this test runs the full C++ pipeline on a
// real English LibriSpeech utterance (tests/fixtures/speech.wav, sample
// 2086-149220-0033, 16 kHz mono) and asserts the produced text matches the
// authoritative NeMo CTC-head greedy transcript word-for-word.
//
// The expected string below was produced by NeMo 2.7.3 with
// nvidia/parakeet-tdt_ctc-110m, decoder_type='ctc'. CTC greedy decoding is
// deterministic, so this is a fixed reference. The clip is committed to the
// repo, making the test self-contained (it only needs PARAKEET_TEST_GGUF).
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) {
        std::fprintf(stderr, "test_transcribe_speech: PARAKEET_TEST_GGUF not set; skip\n");
        return 77;
    }

    const std::string expected =
        "Well, I don't wish to see it any more, observed Phoebe, "
        "turning away her eyes. It is certainly very like the old portrait.";

    std::string got;
    try {
        got = pk::transcribe(gguf, "tests/fixtures/speech.wav");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "test_transcribe_speech: pk::transcribe threw: %s\n", e.what());
        return 1;
    }

    std::fprintf(stderr, "test_transcribe_speech: got      = %s\n", got.c_str());
    std::fprintf(stderr, "test_transcribe_speech: expected = %s\n", expected.c_str());

    if (got != expected) {
        std::fprintf(stderr, "test_transcribe_speech: MISMATCH vs NeMo CTC reference\n");
        return 1;
    }

    std::fprintf(stderr, "test_transcribe_speech: PASS (word-for-word match with NeMo CTC)\n");
    return 0;
}
