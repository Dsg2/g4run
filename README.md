# g4run
Lightweight CPU optimised .GGUF inference

## Build: (MSYS2 MinGW64)
```
MSYS_NO_PATHCONV=1 cmd /c "set PATH=C:\msys64\mingw64\bin;%PATH% && C:\msys64\mingw64\bin\gcc.exe -Ofast -march=native -funroll-loops -ffp-contract=fast -o C:\llama\g4run\g4run.exe C:\llama\g4run\src\gguf.c C:\llama\g4run\src\quants.c C:\llama\g4run\src\tokenizer.c C:\llama\g4run\src\model.c C:\llama\g4run\src\sampler.c C:\llama\g4run\src\vulkan.c C:\llama\g4run\src\server.c C:\llama\g4run\src\main.c -lm -lsynchronization -lws2_32"
```

Portable build: replace `-march=native` with `-march=x86-64-v3` (AVX2; loses VNNI).

## Usage:
g4run — gemma4 GGUF inference (CPU)
usage: g4run -m model.gguf [options]
  -p STR          prompt (one-shot completion through chat template)
  -i              interactive chat
  --raw           no chat template (-p used verbatim)
  -n N            max tokens to generate (default 256; -1 = until EOG)
  -c N            context size (default 8192)
  -t N            threads (default: physical core count)
  --temp F        temperature (default from model, 0 = greedy)
  --top-k N       top-k (default from model)
  --top-p F       top-p (default from model)
  --repeat-penalty F  (default 1.0; qwen35 defaults to 1.1 anti-loop)
  --repeat-last-n N   (default 128)
  --no-repeat-ngram N block any verbatim N-gram repeat (anti-loop; 0=off)
  --ban "A,B"        suppress trigger tokens, e.g. --ban "Wait,Hmm" (anti-loop)
  --avx512        opt into AVX-512 q8_0/q4_0 dots (default AVX2; 512 was
                  slower on Tiger Lake — see README; --no-avx512 forces AVX2)
  --seed N        RNG seed (default time)
  --sys STR       system prompt
  --think         enable the thinking channel (model reasons in a
                  <thought> block before answering; chat mode strips
                  thoughts from the context after each turn)
  --mtp PATH      gemma4-assistant draft head .gguf for speculative
                  decoding (greedy; output identical to plain decode)
  --draft N       draft tokens per round (default = nextn_predict_layers)
  --spec N        qwen35 prompt-lookup speculative decoding, draft N tokens
                  (greedy; output identical to plain greedy; one-shot -p)
  --draft-quant T with --spec N: self-speculative decode using a low-bit
                  in-RAM draft of the model (q2_k|q3_k|q4_0); helps free-form
  --jit-quant T   requantize weights in RAM at load: q8_0|q4_0|q3_k|q2_k
                  (smaller = faster but rougher; no smaller file needed)
 qwen35 fine-tune (easy):
  --finetune F    MeZO (forward-only) LoRA finetune on text file F, then
  --ft-out G        save a fine-tuned copy to G. Good defaults built in:
                    g4run -m M --finetune F.txt --ft-out tuned.gguf
  --ft-preset P   quick | standard | thorough  (sets rank/steps/lr/clip/q)
 qwen35 fine-tune (advanced overrides):
                  --ft-rank/--ft-steps/--ft-lr/--ft-eps/--ft-scale/--ft-batch
                  --ft-layers a:b (default: auto middle band)  --ft-eval N
                  --ft-clip C (anti-divergence)   --ft-q N (avg N perturbations/step)
 qwen35 FFN prune (easy):
  --prune-preset P  light | balanced | aggressive  (keep 80/70/50% + LS comp),
                    then --export G to save:  g4run -m M --prune-preset balanced --export p.gguf
 qwen35 FFN prune (advanced overrides):
  --prune K       keep fraction K of FFN neurons (1.0 = no-op); reports PPL+KL
  --prune-down T  ffn_down requant: q8_0(def)|q4_0|q3_k|q2_k
  --compensate    dropped-neuron mean as a per-layer bias (M2)
  --comp-ls       blockwise least-squares comp, folded into ffn_down (M3;
                  exports cleanly); --ls-block N sets the block size (256)
  --export F      write the (pruned) model to GGUF file F
  --perplexity    eval: perplexity of -p corpus (quality metric)
  --prune-stats F qwen35: dump FFN neuron importance over -p corpus to F
  --cache-type-k T  KV cache K type: f16 (default) | q8_0 | q4_0
  --cache-type-v T  KV cache V type: f16 (default) | q8_0 | q4_0
  --server        run as an HTTP/JSON server (sockets)
  --host STR      server bind address (default 127.0.0.1)
  --port N        server port (default 8080)
  --no-bos        don't add BOS (override metadata)
  --bos           force add BOS
  --dump          dump GGUF metadata and exit
  --fix-llama     patch a pruned model's feed_forward_length metadata in place
                  to match its tensors, so it loads in llama.cpp (qwen35)
  --selftest      run kernel selftests and exit
  --tokenize STR  tokenize STR, print ids, exit
  --parse-special parse special tokens in prompt text
  --verbose       timings and diagnostics
