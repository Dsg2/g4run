# g4run
Lightweight CPU optimised .GGUF inference

## Build: (MSYS2 MinGW64)
```
MSYS_NO_PATHCONV=1 cmd /c "set PATH=C:\msys64\mingw64\bin;%PATH% && C:\msys64\mingw64\bin\gcc.exe -O3 -march=native -funroll-loops -ffp-contract=fast -o C:\llama\g4run\g4run.exe C:\llama\g4run\src\gguf.c C:\llama\g4run\src\quants.c C:\llama\g4run\src\tokenizer.c C:\llama\g4run\src\model.c C:\llama\g4run\src\sampler.c C:\llama\g4run\src\vulkan.c C:\llama\g4run\src\server.c C:\llama\g4run\src\main.c -lm -lsynchronization -lws2_32"
```

Portable build: replace `-march=native` with `-march=x86-64-v3` (AVX2; loses VNNI).
