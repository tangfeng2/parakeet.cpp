#include "tokenizer.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main() {
    const char* gguf_path = std::getenv("PARAKEET_TEST_GGUF");
    const char* base_path = std::getenv("PARAKEET_TEST_BASELINE");
    if (!gguf_path || !base_path) {
        std::fprintf(stderr, "test_tokenizer: env vars not set; skip\n");
        return 77;
    }

    // Load tokenizer pieces from the model GGUF via ModelLoader.
    pk::ModelLoader ml;
    if (!ml.load(gguf_path)) {
        std::fprintf(stderr, "test_tokenizer: failed to load model %s\n", gguf_path);
        return 1;
    }
    const std::vector<std::string>& pieces = ml.tokenizer_pieces();
    if (pieces.empty()) {
        std::fprintf(stderr, "test_tokenizer: tokenizer_pieces is empty\n");
        return 1;
    }
    std::fprintf(stderr, "test_tokenizer: loaded %zu tokenizer pieces\n", pieces.size());

    // Open baseline GGUF to read detok_ids tensor and baseline.detok_text KV.
    ggml_context* bctx = nullptr;
    gguf_init_params bp{ false, &bctx };
    gguf_context* bg = gguf_init_from_file(base_path, bp);
    if (!bg) {
        std::fprintf(stderr, "test_tokenizer: failed to open baseline %s\n", base_path);
        return 1;
    }

    // Read detok_ids (int32 tensor) from baseline.
    ggml_tensor* id_tensor = ggml_get_tensor(bctx, "detok_ids");
    if (!id_tensor) {
        std::fprintf(stderr, "test_tokenizer: 'detok_ids' tensor not found in baseline\n");
        gguf_free(bg);
        ggml_free(bctx);
        return 1;
    }
    size_t n_ids = (size_t)ggml_nelements(id_tensor);
    std::vector<int32_t> ids(n_ids);
    std::memcpy(ids.data(), id_tensor->data, n_ids * sizeof(int32_t));

    // Read baseline.detok_text (STRING KV) from baseline.
    int64_t text_key_id = gguf_find_key(bg, "baseline.detok_text");
    if (text_key_id < 0) {
        std::fprintf(stderr, "test_tokenizer: 'baseline.detok_text' KV not found in baseline\n");
        gguf_free(bg);
        ggml_free(bctx);
        return 1;
    }
    std::string expected_text = gguf_get_val_str(bg, text_key_id);
    gguf_free(bg);
    ggml_free(bctx);

    std::fprintf(stderr, "test_tokenizer: detok_ids=[");
    for (size_t i = 0; i < ids.size(); ++i) {
        std::fprintf(stderr, "%d%s", ids[i], i+1 < ids.size() ? ", " : "");
    }
    std::fprintf(stderr, "]\n");
    std::fprintf(stderr, "test_tokenizer: expected_text=%s\n", expected_text.c_str());

    // Run detokenize.
    std::string got_text = pk::detokenize(pieces, ids);
    std::fprintf(stderr, "test_tokenizer: got_text=%s\n", got_text.c_str());

    if (got_text != expected_text) {
        std::fprintf(stderr, "test_tokenizer: MISMATCH\n  got:      %s\n  expected: %s\n",
                     got_text.c_str(), expected_text.c_str());
        return 1;
    }

    std::fprintf(stderr, "test_tokenizer: EXACT MATCH: %s\n", got_text.c_str());
    return 0;
}
