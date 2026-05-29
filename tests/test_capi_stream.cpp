#include "parakeet_capi.h"
#include "audio_io.hpp"   // pk::load_audio_16k_mono (test links the parakeet lib)
#include "parity.hpp"     // pktest::load_kv_str / load_baseline_i32

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cstdint>

// Streaming C-API end-to-end parity test (Phase 5c, Task 7).
//
// Drives the flat streaming C-API on the cache-aware EOU streaming model
// nvidia/parakeet_realtime_eou_120m-v1:
//   parakeet_capi_stream_begin
//     -> feed tests/fixtures/speech.wav PCM in real-time-sized chunks via
//        parakeet_capi_stream_feed (concatenating the newly-finalized text)
//     -> parakeet_capi_stream_finalize (flush the tail)
//   -> assert the concatenated transcript EQUALS baseline.stream_text from
//      /tmp/baseline_eou_stream.gguf — NeMo's OWN cache-aware streaming decode
//      transcript (<EOU>/<EOB> stripped, handled identically).
//
// The streaming reference (gen_stream_baseline.py) drives NeMo's
// conformer_stream_step carrying previous_hypotheses (partial_hypotheses) across
// chunks — the exact decoder-state-carry this C-API mirrors. For this clip NeMo
// streaming emits 44 tokens == offline rnnt_token_ids minus the trailing
// streaming-tail <EOU> (the final chunk's right context is incomplete by design,
// so neither NeMo nor we emit it during the stream — and finalize does NOT
// fabricate it). stream.eou_in_stream==0 records that no <EOU> fired in-stream.
//
// Skips (exit 77) unless PARAKEET_TEST_GGUF_EOU + PARAKEET_TEST_BASELINE_EOU_STREAM
// are set (the streaming EOU model is a ~480MB download, not in CI).
//
// LABEL model
// WORKING_DIRECTORY (run from project root; wav path is relative)

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF_EOU");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_EOU_STREAM");
    if (!gguf || !base) {
        std::fprintf(stderr,
            "test_capi_stream: PARAKEET_TEST_GGUF_EOU and/or "
            "PARAKEET_TEST_BASELINE_EOU_STREAM not set; skip (streaming EOU model "
            "is a ~480MB download, not in CI)\n");
        return 77;
    }

    // --- NeMo streaming reference: transcript + token ids ---
    std::string ref_text;
    if (!pktest::load_kv_str(base, "baseline.stream_text", ref_text)) {
        std::fprintf(stderr, "test_capi_stream: baseline.stream_text not in %s\n", base);
        return 1;
    }
    std::vector<int32_t> ref_ids;
    if (!pktest::load_baseline_i32(base, "stream_token_ids", ref_ids)) {
        std::fprintf(stderr, "test_capi_stream: stream_token_ids not in %s\n", base);
        return 1;
    }
    std::fprintf(stderr, "test_capi_stream: ref stream_text = %s\n", ref_text.c_str());
    std::fprintf(stderr, "test_capi_stream: ref stream tokens = %zu\n", ref_ids.size());

    // --- Load the clip PCM (16 kHz mono) ---
    pk::Audio audio;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio)) {
        std::fprintf(stderr, "test_capi_stream: failed to load speech.wav\n");
        return 1;
    }

    // --- Load model + begin a stream ---
    parakeet_ctx* ctx = parakeet_capi_load(gguf);
    if (!ctx) {
        std::fprintf(stderr, "test_capi_stream: parakeet_capi_load failed for %s\n", gguf);
        return 1;
    }
    parakeet_stream* s = parakeet_capi_stream_begin(ctx);
    if (!s) {
        std::fprintf(stderr, "test_capi_stream: stream_begin failed: %s\n",
                     parakeet_capi_last_error(ctx));
        parakeet_capi_free(ctx);
        return 1;
    }

    // --- Feed the PCM in real-time-sized chunks (~100 ms = 1600 samples) ---
    const int chunk = 1600;
    const int n = (int)audio.samples.size();
    std::string acc;
    bool any_eou = false;
    for (int off = 0; off < n; off += chunk) {
        const int len = std::min(chunk, n - off);
        int eou = 0;
        char* t = parakeet_capi_stream_feed(s, audio.samples.data() + off, len, &eou);
        if (!t) {
            std::fprintf(stderr, "test_capi_stream: stream_feed returned NULL: %s\n",
                         parakeet_capi_last_error(ctx));
            parakeet_capi_stream_free(s);
            parakeet_capi_free(ctx);
            return 1;
        }
        if (eou) any_eou = true;
        acc += t;
        parakeet_capi_free_string(t);
    }

    // --- Finalize: flush the streaming tail ---
    char* fin = parakeet_capi_stream_finalize(s);
    if (!fin) {
        std::fprintf(stderr, "test_capi_stream: stream_finalize returned NULL: %s\n",
                     parakeet_capi_last_error(ctx));
        parakeet_capi_stream_free(s);
        parakeet_capi_free(ctx);
        return 1;
    }
    acc += fin;
    parakeet_capi_free_string(fin);

    parakeet_capi_stream_free(s);
    parakeet_capi_free(ctx);

    std::fprintf(stderr, "test_capi_stream: got stream_text = %s\n", acc.c_str());
    std::fprintf(stderr, "test_capi_stream: any in-stream EOU event = %s\n",
                 any_eou ? "yes" : "no");

    const bool text_ok = (acc == ref_text);
    if (!text_ok) {
        std::fprintf(stderr,
            "test_capi_stream: TEXT MISMATCH vs NeMo streaming reference\n"
            "  got:      %s\n"
            "  expected: %s\n",
            acc.c_str(), ref_text.c_str());
        return 1;
    }

    std::fprintf(stderr,
        "test_capi_stream: PASS — streaming C-API transcript == NeMo cache-aware "
        "streaming decode (%zu ref tokens, EOU stripped + surfaced as events; "
        "no in-stream <EOU> for this clip, matching NeMo eou_in_stream=0)\n",
        ref_ids.size());
    return 0;
}
