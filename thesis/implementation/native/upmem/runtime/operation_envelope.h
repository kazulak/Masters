#ifndef UPMEM_UPMEM_RUNTIME_OPERATION_ENVELOPE_H
#define UPMEM_UPMEM_RUNTIME_OPERATION_ENVELOPE_H

#include <stddef.h>
#include <stdint.h>

#include "request.h"

#define EXECUTION_PLAN_V4_OPERATION_MAGIC "UPOENV2\0"
#define EXECUTION_PLAN_V4_OPERATION_VERSION 2u
#define EXECUTION_PLAN_V4_OPERATION_HEADER_BYTES 96u
#define EXECUTION_PLAN_V4_OPERATION_DESCRIPTOR_BYTES 200u

typedef struct {
    const unsigned char *mapping;
    size_t file_size;
    int file_descriptor;
    uint32_t descriptor_count;
    uint64_t operation_sequence;
    unsigned char digest[32];
} execution_plan_v4_operation_envelope_t;

int execution_plan_v4_operation_open(
    const char *session_root,
    const char *relative_path,
    const char *submitted_sha256,
    uint32_t expected_dpus,
    uint32_t expected_tasklets,
    execution_plan_v4_operation_envelope_t *operation,
    char **error_message
);

int execution_plan_v4_operation_descriptor(
    const execution_plan_v4_operation_envelope_t *operation,
    uint32_t index,
    execution_plan_v4_embedded_request_t *descriptor,
    char **error_message
);

void execution_plan_v4_operation_close(
    execution_plan_v4_operation_envelope_t *operation
);

#endif
