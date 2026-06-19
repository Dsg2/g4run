// vulkan.c — V0: runtime Vulkan loader, device enumeration, capability probe.
// vulkan-1.dll is loaded at runtime (LoadLibrary) so the exe has no hard
// dependency and runs unchanged on machines without Vulkan.
#include "g4run.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

static HMODULE vk_dll;
static PFN_vkGetInstanceProcAddr gipa;

#define VK_IFN(name) PFN_##name name = (PFN_##name)gipa(inst, #name)
#define VK_GFN(name) PFN_##name name = (PFN_##name)gipa(NULL, #name)

int g4_vk_list(void) {
    vk_dll = LoadLibraryA("vulkan-1.dll");
    if (!vk_dll) { printf("vulkan: vulkan-1.dll not found\n"); return 1; }
    gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(vk_dll, "vkGetInstanceProcAddr");
    if (!gipa) { printf("vulkan: no vkGetInstanceProcAddr\n"); return 1; }

    VkInstance inst = VK_NULL_HANDLE;
    {
        VK_GFN(vkCreateInstance);
        VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        ai.pApplicationName = "g4run";
        ai.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &ai;
        VkResult r = vkCreateInstance(&ici, NULL, &inst);
        if (r != VK_SUCCESS) { printf("vulkan: vkCreateInstance failed (%d)\n", r); return 1; }
    }

    VK_IFN(vkEnumeratePhysicalDevices);
    VK_IFN(vkGetPhysicalDeviceProperties2);
    VK_IFN(vkGetPhysicalDeviceFeatures2);
    VK_IFN(vkEnumerateDeviceExtensionProperties);
    VK_IFN(vkGetPhysicalDeviceMemoryProperties);
    VK_IFN(vkDestroyInstance);

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice devs[8];
    if (n > 8) n = 8;
    vkEnumeratePhysicalDevices(inst, &n, devs);
    printf("vulkan: %u device(s)\n", n);

    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceSubgroupProperties sg = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES };
        VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg };
        vkGetPhysicalDeviceProperties2(devs[i], &p2);

        VkPhysicalDeviceShaderFloat16Int8Features f16i8 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
        VkPhysicalDevice16BitStorageFeatures st16 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &f16i8 };
        VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &st16 };
        vkGetPhysicalDeviceFeatures2(devs[i], &f2);

        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, NULL);
        VkExtensionProperties * ex = malloc(ne * sizeof(*ex));
        vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, ex);
        bool has_emh = false, has_idot = false;
        uint32_t emh_align = 0;
        for (uint32_t e = 0; e < ne; e++) {
            if (!strcmp(ex[e].extensionName, "VK_EXT_external_memory_host")) has_emh = true;
            if (!strcmp(ex[e].extensionName, "VK_KHR_shader_integer_dot_product")) has_idot = true;
        }
        free(ex);
        if (has_emh) {
            VkPhysicalDeviceExternalMemoryHostPropertiesEXT emh = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
            VkPhysicalDeviceProperties2 p2b = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &emh };
            vkGetPhysicalDeviceProperties2(devs[i], &p2b);
            emh_align = (uint32_t)emh.minImportedHostPointerAlignment;
        }

        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(devs[i], &mp);
        uint64_t devmem = 0;
        for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                if (mp.memoryHeaps[h].size > devmem) devmem = mp.memoryHeaps[h].size;

        printf("  [%u] %s\n", i, p2.properties.deviceName);
        printf("      api %u.%u.%u | type %s | heap %.1f GB\n",
               VK_API_VERSION_MAJOR(p2.properties.apiVersion),
               VK_API_VERSION_MINOR(p2.properties.apiVersion),
               VK_API_VERSION_PATCH(p2.properties.apiVersion),
               p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "iGPU" :
               p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "dGPU" : "other",
               devmem / 1e9);
        printf("      subgroup %u | sharedmem %u KB | fp16 %d | int8 %d | intdot %d | ext_mem_host %d (align %u)\n",
               sg.subgroupSize,
               p2.properties.limits.maxComputeSharedMemorySize / 1024,
               f16i8.shaderFloat16, f16i8.shaderInt8, has_idot, has_emh, emh_align);
    }

    vkDestroyInstance(inst, NULL);
    FreeLibrary(vk_dll);
    return 0;
}

// ===========================================================================
// V1 test harness: Q6_K matmul on the GPU with zero-copy weight import.
// Proves the full stack: loader -> device -> pipeline -> external_memory_host
// import of the GGUF mmap -> dispatch -> readback -> parity vs CPU kernel.
// ===========================================================================

#include "shaders.h"

#define VKCHK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "vulkan: %s failed (%d) at line %d\n", #x, _r, __LINE__); return 1; } } while (0)

int g4_vk_test(const char * model_path) {
    g4_init_tables();
    g4_model_file mf;
    if (g4_gguf_open(&mf, model_path, false)) return 1;

    // pick a Q6_K tensor to multiply (ffn_gate of layer 0: 1536 -> 6144)
    const g4_tensor * T = NULL;
    for (uint64_t i = 0; i < mf.n_tensors; i++)
        if (mf.tensors[i].type == G4_Q6_K && mf.tensors[i].ne[1] >= 4096 && strstr(mf.tensors[i].name, "ffn_gate")) { T = &mf.tensors[i]; break; }
    if (!T) { fprintf(stderr, "vulkan: no Q6_K ffn tensor in this file (use the Q6_K_P model)\n"); return 1; }
    const int64_t ncols = T->ne[0], nrows = T->ne[1];
    const int nb = 16;   // batch of 16 tokens
    printf("vulkan: testing %s [%lld x %lld] x %d tokens\n", T->name, (long long)ncols, (long long)nrows, nb);

    // --- activations: random, quantized to q8_K (same as the CPU path) ---
    const size_t astride = ((size_t)(ncols/QK_K) * sizeof(block_q8_K) + 63) & ~(size_t)63;
    float * x = malloc((size_t)nb * ncols * 4);
    block_q8_K * aq = malloc((size_t)nb * astride);
    uint64_t rng = 0x2bd7f1a5c0ffee11ULL;
    for (int64_t i = 0; i < nb*ncols; i++) {
        rng = rng*6364136223846793005ULL + 1442695040888963407ULL;
        x[i] = ((int32_t)(rng >> 33)) / (float)(1u << 30);
    }
    for (int b = 0; b < nb; b++)
        g4_quantize_q8_K(x + (size_t)b*ncols, (block_q8_K *)((uint8_t *)aq + b*astride), ncols);

    // --- CPU reference (x1 kernel per row) ---
    float * y_cpu = malloc((size_t)nb * nrows * 4);
    double t0 = g4_time_ms();
    for (int b = 0; b < nb; b++)
        for (int64_t r = 0; r < nrows; r++)
            y_cpu[(size_t)b*nrows + r] = g4_dot_q6_K_q8_K(
                (const uint8_t *)T->data + r * g4_row_size(G4_Q6_K, ncols),
                (const uint8_t *)aq + b*astride, ncols);
    double t_cpu = g4_time_ms() - t0;

    // --- Vulkan setup ---
    vk_dll = LoadLibraryA("vulkan-1.dll");
    if (!vk_dll) { fprintf(stderr, "vulkan: no vulkan-1.dll\n"); return 1; }
    gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(vk_dll, "vkGetInstanceProcAddr");

    VkInstance inst;
    {
        VK_GFN(vkCreateInstance);
        VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        ai.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &ai;
        VKCHK(vkCreateInstance(&ici, NULL, &inst));
    }
    VK_IFN(vkEnumeratePhysicalDevices);
    VK_IFN(vkGetPhysicalDeviceQueueFamilyProperties);
    VK_IFN(vkCreateDevice);
    VK_IFN(vkGetDeviceProcAddr);
    VK_IFN(vkGetPhysicalDeviceMemoryProperties);

    VkPhysicalDevice phys;
    { uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &phys); if (!n) { fprintf(stderr, "vulkan: no device\n"); return 1; } }

    uint32_t qfam = 0;
    {
        VkQueueFamilyProperties qf[8]; uint32_t n = 8;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, qf);
        for (uint32_t i = 0; i < n; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    }

    VkDevice dev;
    {
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
        const char * exts[] = { "VK_EXT_external_memory_host", "VK_KHR_8bit_storage", "VK_KHR_shader_float16_int8" };
        VkPhysicalDeviceShaderFloat16Int8Features f16i8 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
        f16i8.shaderInt8 = VK_TRUE;
        VkPhysicalDevice8BitStorageFeatures s8 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, &f16i8 };
        s8.storageBuffer8BitAccess = VK_TRUE;
        s8.uniformAndStorageBuffer8BitAccess = VK_TRUE;
        VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &s8 };
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 3; dci.ppEnabledExtensionNames = exts;
        VKCHK(vkCreateDevice(phys, &dci, NULL, &dev));
    }
    #define VK_DFN(name) PFN_##name name = (PFN_##name)vkGetDeviceProcAddr(dev, #name)
    VK_DFN(vkGetDeviceQueue); VK_DFN(vkCreateBuffer); VK_DFN(vkGetBufferMemoryRequirements);
    VK_DFN(vkAllocateMemory); VK_DFN(vkBindBufferMemory); VK_DFN(vkMapMemory);
    VK_DFN(vkCreateShaderModule); VK_DFN(vkCreateDescriptorSetLayout); VK_DFN(vkCreatePipelineLayout);
    VK_DFN(vkCreateComputePipelines); VK_DFN(vkCreateDescriptorPool); VK_DFN(vkAllocateDescriptorSets);
    VK_DFN(vkUpdateDescriptorSets); VK_DFN(vkCreateCommandPool); VK_DFN(vkAllocateCommandBuffers);
    VK_DFN(vkBeginCommandBuffer); VK_DFN(vkCmdBindPipeline); VK_DFN(vkCmdBindDescriptorSets);
    VK_DFN(vkCmdPushConstants); VK_DFN(vkCmdDispatch); VK_DFN(vkEndCommandBuffer);
    VK_DFN(vkQueueSubmit); VK_DFN(vkCreateFence); VK_DFN(vkWaitForFences); VK_DFN(vkResetFences);
    VK_DFN(vkDestroyDevice);
    PFN_vkGetMemoryHostPointerPropertiesEXT vkGetMemoryHostPointerPropertiesEXT =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");

    VkQueue queue; vkGetDeviceQueue(dev, qfam, 0, &queue);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);

    // --- W: zero-copy import of the page-aligned region around the tensor ---
    const size_t wbytes = g4_row_size(G4_Q6_K, ncols) * (size_t)nrows;
    const uint8_t * wptr = (const uint8_t *)T->data;
    const uintptr_t page = 4096;
    const uintptr_t wlo = (uintptr_t)wptr & ~(page - 1);
    const uintptr_t whi = ((uintptr_t)wptr + wbytes + page - 1) & ~(page - 1);
    const uint32_t wbase = (uint32_t)((uintptr_t)wptr - wlo);
    VkBuffer wbuf; VkDeviceMemory wmem;
    {
        VkExternalMemoryBufferCreateInfo emb = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
        emb.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &emb };
        bci.size = whi - wlo;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VKCHK(vkCreateBuffer(dev, &bci, NULL, &wbuf));

        VkMemoryHostPointerPropertiesEXT hpp = { VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
        VKCHK(vkGetMemoryHostPointerPropertiesEXT(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                                  (void *)wlo, &hpp));
        uint32_t mt = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((hpp.memoryTypeBits & (1u << i))) { mt = i; break; }
        if (mt == UINT32_MAX) { fprintf(stderr, "vulkan: no memory type for host import\n"); return 1; }

        VkImportMemoryHostPointerInfoEXT imp = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
        imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        imp.pHostPointer = (void *)wlo;
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp };
        mai.allocationSize = whi - wlo;
        mai.memoryTypeIndex = mt;
        VKCHK(vkAllocateMemory(dev, &mai, NULL, &wmem));
        VKCHK(vkBindBufferMemory(dev, wbuf, wmem, 0));
    }

    // --- A (activations) and Y (output): host-visible UMA buffers ---
    VkBuffer abuf, ybuf; VkDeviceMemory amem, ymem;
    void * amap, * ymap;
    {
        uint32_t hv = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { hv = i; break; }
        }
        VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.size = (size_t)nb * astride;
        VKCHK(vkCreateBuffer(dev, &bci, NULL, &abuf));
        bci.size = (size_t)nb * nrows * 4;
        VKCHK(vkCreateBuffer(dev, &bci, NULL, &ybuf));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, abuf, &mr);
        VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        mai.allocationSize = mr.size; mai.memoryTypeIndex = hv;
        VKCHK(vkAllocateMemory(dev, &mai, NULL, &amem));
        VKCHK(vkBindBufferMemory(dev, abuf, amem, 0));
        VKCHK(vkMapMemory(dev, amem, 0, VK_WHOLE_SIZE, 0, &amap));
        vkGetBufferMemoryRequirements(dev, ybuf, &mr);
        mai.allocationSize = mr.size;
        VKCHK(vkAllocateMemory(dev, &mai, NULL, &ymem));
        VKCHK(vkBindBufferMemory(dev, ybuf, ymem, 0));
        VKCHK(vkMapMemory(dev, ymem, 0, VK_WHOLE_SIZE, 0, &ymap));
    }
    memcpy(amap, aq, (size_t)nb * astride);

    // --- pipeline ---
    VkShaderModule sm;
    {
        VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smci.codeSize = sizeof(spv_q6k_mm);
        smci.pCode = (const uint32_t *)spv_q6k_mm;
        VKCHK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    }
    VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe;
    {
        VkDescriptorSetLayoutBinding binds[3];
        for (int i = 0; i < 3; i++) {
            binds[i] = (VkDescriptorSetLayoutBinding){ (uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
        }
        VkDescriptorSetLayoutCreateInfo dlci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        dlci.bindingCount = 3; dlci.pBindings = binds;
        VKCHK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));
        VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 20 };
        VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        VKCHK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
        VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage = (VkPipelineShaderStageCreateInfo){ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                       NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL };
        cpci.layout = pl;
        VKCHK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    }

    // --- descriptors ---
    VkDescriptorSet ds;
    {
        VkDescriptorPoolSize dps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
        VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
        VkDescriptorPool dp;
        VKCHK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
        VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
        VKCHK(vkAllocateDescriptorSets(dev, &dsai, &ds));
        VkDescriptorBufferInfo bi[3] = {
            { wbuf, 0, VK_WHOLE_SIZE }, { abuf, 0, VK_WHOLE_SIZE }, { ybuf, 0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet wr[3];
        for (int i = 0; i < 3; i++) {
            wr[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            wr[i].dstSet = ds; wr[i].dstBinding = (uint32_t)i;
            wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(dev, 3, wr, 0, NULL);
    }

    // --- record + submit ---
    VkCommandPool cp; VkCommandBuffer cb; VkFence fence;
    {
        VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        cpci.queueFamilyIndex = qfam;
        VKCHK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
        VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
        VKCHK(vkAllocateCommandBuffers(dev, &cbai, &cb));
        VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VKCHK(vkCreateFence(dev, &fci, NULL, &fence));
    }
    struct { uint32_t ncols, nrows, nb, astride, wbase; } pc =
        { (uint32_t)ncols, (uint32_t)nrows, (uint32_t)nb, (uint32_t)astride, wbase };

    double t_gpu = 0;
    const int reps = 3;
    for (int rep = 0; rep < reps; rep++) {
        VkCommandBufferBeginInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        VKCHK(vkBeginCommandBuffer(cb, &cbi));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, (uint32_t)(((uint64_t)nrows * nb + 63) / 64), 1, 1);
        VKCHK(vkEndCommandBuffer(cb));
        double tt = g4_time_ms();
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VKCHK(vkQueueSubmit(queue, 1, &si, fence));
        VKCHK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 30ull*1000*1000*1000));
        VKCHK(vkResetFences(dev, 1, &fence));
        double el = g4_time_ms() - tt;
        if (rep > 0 || reps == 1) t_gpu += el;   // skip first (pipeline warm)
    }
    t_gpu /= (reps > 1 ? reps - 1 : 1);

    // --- compare ---
    const float * y_gpu = (const float *)ymap;
    double maxrel = 0; int bad = 0;
    for (int64_t i = 0; i < nb*nrows; i++) {
        double diff = fabs((double)y_gpu[i] - y_cpu[i]);
        double rel = diff / (1e-3 + fabs(y_cpu[i]));
        if (rel > maxrel) maxrel = rel;
        if (rel > 1e-3) bad++;
    }
    const double gmacs = (double)ncols * nrows * nb;
    printf("vulkan: GPU %.2f ms (%.1f GMAC/s) | CPU x1 %.2f ms (%.1f GMAC/s)\n",
           t_gpu, gmacs/t_gpu/1e6, t_cpu, gmacs/t_cpu/1e6);
    printf("vulkan: max rel err %.3g, >1e-3: %d of %lld %s\n",
           maxrel, bad, (long long)(nb*nrows), bad == 0 ? "PASS" : "FAIL");

    vkDestroyDevice(dev, NULL);
    { VK_IFN(vkDestroyInstance); vkDestroyInstance(inst, NULL); }
    FreeLibrary(vk_dll);
    free(x); free(aq); free(y_cpu);
    g4_gguf_close(&mf);
    return bad ? 1 : 0;
}
