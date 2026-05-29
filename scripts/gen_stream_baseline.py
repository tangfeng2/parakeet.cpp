#!/usr/bin/env python3
"""Dump NeMo cache-aware *streaming* encoder outputs to a baseline gguf.

Phase 5b (Task 5) validates the C++ cache-aware streaming encoder
(``pk::StreamingEncoder``) chunk-by-chunk against the *exact* reference NeMo
produces by running its own cache-aware streaming loop. The C++ port must be
numerically equal to NeMo's streaming output (the primary parity gate); by the
cache-aware equivalence property the streaming output is also equal to the
OFFLINE encoder output for every frame EXCEPT the trailing extra outputs of the
final chunk (where ``keep_all_outputs=True`` keeps frames whose right context is
incomplete — those are streaming-tail artifacts, not an offline match).

This script mirrors NeMo's canonical streaming driver:
``ConformerEncoder.cache_aware_stream_step`` fed by
``CacheAwareStreamingAudioBuffer`` (the same buffering used by
``transcribe_simulate_cache_aware_streaming`` in
``parts/mixins/mixins.py`` lines 772-820), with:

  * mel computed once for the whole clip (dither=0),
  * the buffer yields chunk 0 of ``chunk_size[0]=9`` mel frames (no pre-encode
    cache, ``drop_extra_pre_encoded=0``) and every subsequent chunk of
    ``pre_encode_cache_size[1]=9 + chunk_size[1]=16 = 25`` mel frames (the 9
    cache frames overlap the previous window), ``buffer_idx`` advancing by
    ``shift_size`` (9 then 16),
  * ``drop_extra_pre_encoded = streaming_cfg.drop_extra_pre_encoded`` for
    ``step_num != 0`` else 0,
  * ``keep_all_outputs = streaming_buffer.is_buffer_empty()`` (True only on the
    last chunk),
  * the per-layer conv + attention caches carried across chunks.

Each step's valid encoder output (``encoded[:, :, :encoded_len]``) is
concatenated on the time axis -> ``stream_encoder_out`` ``[d_model, T']``
(channels-first, matching the offline ``encoder_out`` orientation).

Stored tensors / KVs (f32 except int32 ids):

* ``stream_encoder_out``  ``[d_model, T']``  concatenated valid streaming output
* ``stream_chunk0_out``   ``[d_model, v0]``   the FIRST chunk's valid output
                                              (for chunk-by-chunk C++ debug)
* ``stream_chunk0_mel``   ``[n_mels, c0]``     the first chunk's mel window
                                              (so the C++ test can feed the exact
                                              same input to localize divergence)
* ``stream.n_chunks`` uint32                  number of streaming steps
* ``stream.valid_out_len`` uint32             valid encoder frames per step
* ``stream.chunk_size_first`` / ``stream.chunk_size`` uint32  mel frames per step
* ``stream.pre_encode_cache_size`` uint32     pre-encode mel cache frames
* ``stream.drop_extra_pre_encoded`` uint32
* ``stream.last_channel_cache_size`` uint32
* ``stream.offline_match_T`` uint32           # leading frames that match offline
                                              (== T' minus the streaming tail)
* ``stream.offline_max_abs_diff`` f32         max|d| stream-vs-offline over the
                                              leading ``offline_match_T`` frames
                                              (sanity: must be tiny)

Streaming DECODE reference (Phase 5c, Task 7) — the authoritative streaming
transcript the C++ ``pk::StreamingSession`` / C-API must reproduce. Driven via
NeMo's own ``conformer_stream_step`` carrying ``previous_hypotheses`` across
chunks (the RNN-T ``partial_hypotheses`` path; mixins.py 592-729 / 772-820):

* ``stream_token_ids``  ``[L]`` int32   NeMo cache-aware streaming greedy token
                                        ids (carried decoder state). For this
                                        clip == offline ``rnnt_token_ids`` minus
                                        the trailing streaming-tail ``<EOU>``
                                        (Task 6 finding).
* ``baseline.stream_text``  str         the streaming transcript detokenized
                                        with ``<EOU>``/``<EOB>`` STRIPPED (what
                                        ``StreamingSession`` exposes as running
                                        text — specials surface as events).
* ``stream.token_count``  uint32        len(stream_token_ids)
* ``stream.eou_count``    uint32         number of ``<EOU>``/``<EOB>`` events
* ``stream_eou_frames``   ``[E]`` int32  encoder-frame index of each event
* ``stream_eou_tokens``   ``[E]`` int32  the special token id (``<EOU>``/``<EOB>``)
* ``stream.eou_in_stream`` uint32        1 if any ``<EOU>``/``<EOB>`` fired during
                                        the per-chunk stream, 0 if only on the
                                        final flush (Task 6: 0 for this clip).

Exit codes (ctest convention): 0 = ok, 2 = deps/model unavailable, 1 = fail.
"""
import argparse
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

try:
    import gguf
except ImportError as e:  # pragma: no cover - env guard
    print(f"stream-baseline: missing dependency 'gguf': {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

try:
    import torch
    import soundfile as sf
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )
except ImportError as e:  # pragma: no cover - env guard
    print(f"stream-baseline: missing dependency: {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet_realtime_eou_120m-v1",
                    help="HF id or local .nemo")
    ap.add_argument("--audio", required=True, help="16k mono wav clip")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    is_local = pathlib.Path(args.model).exists()
    try:
        if is_local:
            m = ASRModel.restore_from(args.model, map_location="cpu")
        else:
            m = ASRModel.from_pretrained(args.model, map_location="cpu")
    except Exception as e:  # pragma: no cover - network/cache guard
        print(f"PARAKEET_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)

    m.eval()
    m.preprocessor.featurizer.dither = 0.0
    enc = m.encoder
    if not hasattr(enc, "cache_aware_stream_step"):
        print("PARAKEET_MODEL_UNAVAILABLE: encoder has no cache_aware_stream_step "
              "(not a streaming model)", file=sys.stderr)
        sys.exit(2)
    enc.setup_streaming_params()
    sc = enc.streaming_cfg

    # Use the non-batched per-sample greedy RNN-T decoder for the streaming
    # reference. It supports partial_hypotheses (carried decoder state across
    # chunks) and produces list-typed per-token timestamps. The default
    # `greedy_batch` (loop_labels) path emits dict timestamps that crash
    # Hypothesis.merge_ on this NeMo version, so we switch strategies here. The
    # emitted token sequence is identical to greedy_batch (both are exact greedy).
    import copy as _copy
    from omegaconf import open_dict as _open_dict
    _dcfg = _copy.deepcopy(m.cfg.decoding)
    with _open_dict(_dcfg):
        _dcfg.strategy = "greedy"
        # compute_timestamps runs a char-offset post-pass that mishandles the
        # carried partial hypotheses; keep it off. The greedy decoder still fills
        # hyp.timestamp with the per-token encoder-frame indices (a plain list).
        _dcfg.compute_timestamps = False
        _dcfg.preserve_alignments = False
        if "greedy" in _dcfg:
            _dcfg.greedy.preserve_frame_confidence = False
    m.change_decoding_strategy(_dcfg)

    chunk_first = sc.chunk_size[0] if isinstance(sc.chunk_size, list) else sc.chunk_size
    chunk_main = sc.chunk_size[1] if isinstance(sc.chunk_size, list) else sc.chunk_size
    pre_cache = (sc.pre_encode_cache_size[1]
                 if isinstance(sc.pre_encode_cache_size, list)
                 else sc.pre_encode_cache_size)

    # Load the clip as float32 mono [1, num_samples].
    wav, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        print(f"PARAKEET_BASELINE_BAD_AUDIO: expected 16k mono, got sr={sr}",
              file=sys.stderr)
        sys.exit(1)
    wav_t = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)  # [1,S]
    len_t = torch.tensor([wav_t.shape[1]], dtype=torch.int64)

    with torch.no_grad():
        feats, feat_len = m.preprocessor(input_signal=wav_t, length=len_t)  # [1,n_mels,T]
        enc_off, enc_off_len = enc(audio_signal=feats, length=feat_len)     # [1,d_model,T']
    n_mels = int(feats.shape[1])
    enc_off_np = enc_off.detach().cpu().float().numpy()[0]  # [d_model, T']

    # ---- NeMo canonical cache-aware streaming loop (mixins.py 772-820) ----
    # We drive m.conformer_stream_step so BOTH the encoder output AND the RNN-T
    # decode hypothesis come from the same per-chunk loop, with the decoder state
    # carried across chunks via previous_hypotheses (partial_hypotheses) — exactly
    # what pk::StreamingSession mirrors.
    sb = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False,
                                        pad_and_drop_preencoded=False)
    sb.append_processed_signal(feats, stream_id=-1)
    sb_iter = iter(sb)
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        enc.get_initial_cache_state(batch_size=1)
    )

    outs = []
    chunk0_out = None
    chunk0_mel = None
    previous_hypotheses = None
    pred_out_stream = None
    best_hyp = None
    for step_num, (chunk_audio, chunk_lengths) in enumerate(sb_iter):
        drop = sc.drop_extra_pre_encoded if step_num != 0 else 0
        keep_all = sb.is_buffer_empty()
        if step_num == 0:
            # chunk_audio is [1, n_mels, c]; store [n_mels, c]
            chunk0_mel = chunk_audio.detach().cpu().float().numpy()[0]
        # 1) Encoder cache-aware step (kept separate so we can store the per-chunk
        #    valid encoder output exactly as before for the encoder-parity test).
        with torch.no_grad():
            (e, el, cache_last_channel, cache_last_time, cache_last_channel_len) = (
                enc.cache_aware_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=keep_all,
                    drop_extra_pre_encoded=drop,
                )
            )
        valid = int(el[0].item())
        e_valid = e[:, :, :valid].detach().cpu().float()  # [1, d_model, valid]
        outs.append(e_valid)
        if step_num == 0:
            chunk0_out = e_valid.numpy()[0]  # [d_model, v0]

        # 2) RNN-T greedy decode over THIS chunk's valid encoder frames, carrying
        #    the decoder hypothesis (LSTM state + last token) across chunks.
        with torch.inference_mode():
            dec = m.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=e,
                encoded_lengths=el,
                return_hypotheses=True,
                partial_hypotheses=previous_hypotheses,
            )
        best_hyp = dec
        previous_hypotheses = dec

    stream_out = torch.cat(outs, dim=2).numpy()[0]  # [d_model, T']
    n_chunks = len(outs)

    # ---- Streaming DECODE reference: tokens, text, EOU events ----
    stream_hyp = best_hyp[0]
    stream_ys = stream_hyp.y_sequence
    if isinstance(stream_ys, torch.Tensor):
        stream_ys = stream_ys.cpu().tolist()
    stream_token_ids = np.array(list(stream_ys), dtype=np.int32)

    # Per-token encoder-frame timestamps (carried across chunks by NeMo).
    stamp = stream_hyp.timestamp
    if isinstance(stamp, dict):
        stamp = stamp.get("timestep", [])
    if isinstance(stamp, torch.Tensor):
        stamp = stamp.cpu().tolist()
    stamp = list(stamp)

    # Resolve <EOU>/<EOB> ids from the tokenizer vocab (don't hardcode 1024/1025).
    vocab = m.tokenizer.vocab if hasattr(m.tokenizer, "vocab") else None
    eou_id = eob_id = -1
    try:
        toks = m.tokenizer.tokens_to_ids(["<EOU>"])
        eou_id = int(toks[0]) if toks else -1
    except Exception:
        eou_id = -1
    try:
        toks = m.tokenizer.tokens_to_ids(["<EOB>"])
        eob_id = int(toks[0]) if toks else -1
    except Exception:
        eob_id = -1
    if eou_id < 0 or eob_id < 0:
        # Fallback: scan pieces.
        try:
            pieces = m.tokenizer.tokenizer.id_to_piece
            for i in range(m.tokenizer.vocab_size):
                p = m.tokenizer.tokenizer.id_to_piece(i)
                if p == "<EOU>" and eou_id < 0:
                    eou_id = i
                if p == "<EOB>" and eob_id < 0:
                    eob_id = i
        except Exception:
            pass
    specials = {x for x in (eou_id, eob_id) if x >= 0}

    # Streaming text = detokenize the NON-special tokens (specials are events).
    non_special = [int(t) for t in stream_token_ids if int(t) not in specials]
    stream_text = m.tokenizer.ids_to_text(non_special)

    # EOU/EOB events: (encoder-frame index, token id). The streaming-decode path
    # may not emit a trailing <EOU> (Task 6: the final-chunk tail has incomplete
    # right context — NeMo streaming drops it). eou_in_stream reflects that.
    eou_frames, eou_tokens = [], []
    for idx, t in enumerate(stream_token_ids):
        if int(t) in specials:
            frame = int(stamp[idx]) if idx < len(stamp) else -1
            eou_frames.append(frame)
            eou_tokens.append(int(t))
    eou_in_stream = 1 if eou_frames else 0

    # Sanity: stream == offline over all frames EXCEPT the final-chunk tail.
    # The last chunk's keep_all_outputs=True keeps the trailing valid_out_len
    # frames whose right context is incomplete -> they differ from offline.
    valid_out_len = int(sc.valid_out_len)
    Tp = min(stream_out.shape[1], enc_off_np.shape[1])
    offline_match_T = max(0, Tp - valid_out_len)  # leading frames that must match
    if offline_match_T > 0:
        d_lead = np.abs(stream_out[:, :offline_match_T] - enc_off_np[:, :offline_match_T])
        offline_max = float(d_lead.max())
    else:
        offline_max = 0.0
    d_all = np.abs(stream_out[:, :Tp] - enc_off_np[:, :Tp])

    print(f"streaming: n_chunks={n_chunks} stream_T={stream_out.shape[1]} "
          f"offline_T={enc_off_np.shape[1]} valid_out_len={valid_out_len}")
    print(f"  chunk schedule: first={chunk_first} mel frames (no cache), "
          f"main={pre_cache}+{chunk_main}={pre_cache + chunk_main} mel frames")
    print(f"  stream vs offline (leading {offline_match_T} frames): "
          f"max|d|={offline_max:.3e}")
    print(f"  stream vs offline (ALL {Tp} frames incl tail): "
          f"max|d|={d_all.max():.3e} (tail diverges by design)")
    if offline_max > 5e-3:
        print(f"PARAKEET_STREAM_BASELINE_WARN: stream vs offline leading max|d|="
              f"{offline_max:.3e} > 5e-3; the reference may be set up wrong.",
              file=sys.stderr)

    print(f"streaming decode: {len(stream_token_ids)} tokens "
          f"(eou_id={eou_id} eob_id={eob_id})")
    print(f"  stream_token_ids: {stream_token_ids.tolist()}")
    print(f"  baseline.stream_text: {repr(stream_text)}")
    print(f"  eou events (in-stream={eou_in_stream}): "
          f"frames={eou_frames} tokens={eou_tokens}")

    w = gguf.GGUFWriter(args.output, "parakeet-stream-baseline")
    w.add_tensor("stream_encoder_out", np.ascontiguousarray(stream_out, dtype=np.float32))
    w.add_tensor("stream_chunk0_out", np.ascontiguousarray(chunk0_out, dtype=np.float32))
    w.add_tensor("stream_chunk0_mel", np.ascontiguousarray(chunk0_mel, dtype=np.float32))
    w.add_uint32("stream.n_chunks", int(n_chunks))
    w.add_uint32("stream.valid_out_len", valid_out_len)
    w.add_uint32("stream.chunk_size_first", int(chunk_first))
    w.add_uint32("stream.chunk_size", int(chunk_main))
    w.add_uint32("stream.pre_encode_cache_size", int(pre_cache))
    w.add_uint32("stream.drop_extra_pre_encoded", int(sc.drop_extra_pre_encoded))
    w.add_uint32("stream.last_channel_cache_size", int(sc.last_channel_cache_size))
    w.add_uint32("stream.offline_match_T", int(offline_match_T))
    w.add_float32("stream.offline_max_abs_diff", offline_max)
    # ---- Streaming DECODE reference (Task 7) ----
    w.add_uint32("stream.token_count", int(stream_token_ids.shape[0]))
    if stream_token_ids.shape[0] > 0:
        w.add_tensor("stream_token_ids", np.ascontiguousarray(stream_token_ids))
    w.add_string("baseline.stream_text", stream_text)
    w.add_uint32("stream.eou_count", int(len(eou_frames)))
    w.add_uint32("stream.eou_in_stream", int(eou_in_stream))
    w.add_int32("stream.eou_id", int(eou_id))
    w.add_int32("stream.eob_id", int(eob_id))
    if eou_frames:
        w.add_tensor("stream_eou_frames",
                     np.ascontiguousarray(np.array(eou_frames, dtype=np.int32)))
        w.add_tensor("stream_eou_tokens",
                     np.ascontiguousarray(np.array(eou_tokens, dtype=np.int32)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"wrote {args.output}: stream_encoder_out={stream_out.shape} "
          f"chunk0_out={chunk0_out.shape} chunk0_mel={chunk0_mel.shape} "
          f"stream_tokens={stream_token_ids.shape[0]}")


if __name__ == "__main__":
    main()
