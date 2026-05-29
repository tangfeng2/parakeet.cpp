#include "subsampling.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstdint>
int main(){
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE");
    if(!gguf||!base){ std::fprintf(stderr,"env not set; skip\n"); return 77; }
    pk::ModelLoader ml; if(!ml.load(gguf)) return 1;

    // Input: baseline "mel" is [n_mels, T] row-major (feat-major inner=T).
    std::vector<float> mel; std::vector<int64_t> mshape;
    if(!pktest::load_baseline(base, "mel", mel, mshape)) return 1;
    if(mshape.size()!=2){ std::fprintf(stderr,"mel shape rank=%zu\n", mshape.size()); return 1; }
    const int n_mels = (int)mshape[0];
    const int T      = (int)mshape[1];

    pk::Subsampling sub(ml);
    std::vector<float> out; int Tout=0, d_model=0;
    sub.forward(mel, n_mels, T, out, Tout, d_model);

    // Reference: subsampling_out is [T', d_model] row-major.
    std::vector<float> ref; std::vector<int64_t> rshape;
    if(!pktest::load_baseline(base, "subsampling_out", ref, rshape)) return 1;
    if(rshape.size()!=2 || (int)rshape[0]!=Tout || (int)rshape[1]!=d_model){
        std::fprintf(stderr,"shape mismatch got=[%d,%d] ref=[%lld,%lld]\n",
            Tout, d_model, (long long)rshape[0], (long long)rshape[1]);
        return 1;
    }
    bool ok = pktest::compare(out, ref, "subsampling", /*atol*/1e-2f, /*rtol*/1e-2f);
    return ok?0:1;
}
