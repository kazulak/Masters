#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

__global__ void write_smoke_value(int* out) {
    out[0] = 6600;
}

static void print_json_string(const char* value) {
    std::putchar('"');
    if (value != nullptr) {
        for (const char* cursor = value; *cursor != '\0'; ++cursor) {
            if (*cursor == '"' || *cursor == '\\') {
                std::putchar('\\');
            }
            std::putchar(*cursor);
        }
    }
    std::putchar('"');
}

static int fail_json(const char* reason, hipError_t err = hipSuccess) {
    std::printf("{\"status\":\"failed\",\"gpu_program_executed\":false,\"reason\":");
    print_json_string(reason);
    if (err != hipSuccess) {
        std::printf(",\"hip_error\":");
        print_json_string(hipGetErrorString(err));
    }
    std::printf("}\n");
    return 1;
}

int main() {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess) {
        return fail_json("hipGetDeviceCount_failed", err);
    }
    if (device_count < 1) {
        return fail_json("no_hip_devices");
    }

    err = hipSetDevice(0);
    if (err != hipSuccess) {
        return fail_json("hipSetDevice_failed", err);
    }

    hipDeviceProp_t props;
    std::memset(&props, 0, sizeof(props));
    err = hipGetDeviceProperties(&props, 0);
    if (err != hipSuccess) {
        return fail_json("hipGetDeviceProperties_failed", err);
    }

    int* device_value = nullptr;
    err = hipMalloc(&device_value, sizeof(int));
    if (err != hipSuccess) {
        return fail_json("hipMalloc_failed", err);
    }

    write_smoke_value<<<1, 1>>>(device_value);
    err = hipGetLastError();
    if (err != hipSuccess) {
        (void)hipFree(device_value);
        return fail_json("kernel_launch_failed", err);
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        (void)hipFree(device_value);
        return fail_json("hipDeviceSynchronize_failed", err);
    }

    int host_value = 0;
    err = hipMemcpy(&host_value, device_value, sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(device_value);
    if (err != hipSuccess) {
        return fail_json("hipMemcpy_failed", err);
    }
    if (host_value != 6600) {
        return fail_json("kernel_result_mismatch");
    }

    std::printf("{\"status\":\"ok\",\"gpu_program_executed\":true,\"gpu_backend_verified\":true,\"gpu_synchronized\":true,\"device_count\":%d,", device_count);
    std::printf("\"gpu_device_name\":");
    print_json_string(props.name);
    std::printf(",\"gcn_arch_name\":");
    print_json_string(props.gcnArchName);
    std::printf(",\"multi_processor_count\":%d,\"total_global_mem\":%llu}\n",
                props.multiProcessorCount,
                static_cast<unsigned long long>(props.totalGlobalMem));
    return 0;
}
