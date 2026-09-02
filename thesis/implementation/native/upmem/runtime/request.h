#ifndef UPMEM_PLAN_REQUEST_V4_H
#define UPMEM_PLAN_REQUEST_V4_H

#include <stddef.h>
#include <stdint.h>

#include "protocol.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
    execution_plan_v4_work_unit_t work_unit;
    char *a_path;
    char *b_path;
    char *c_path;
    char a_sha256[65];
    char b_sha256[65];
    const unsigned char *a_source;
    const unsigned char *b_source;
    unsigned char *a_payload;
    unsigned char *b_payload;
    unsigned char *c_payload;
} execution_plan_v4_request_item_t;

typedef struct {
    execution_plan_v4_header_t header;
    execution_plan_v4_work_unit_t *work_units;
    execution_plan_v4_request_item_t *items;
    uint32_t item_count;
    char *manifest_path;
    char *sidecar_path;
    char manifest_sha256[65];
    char sidecar_sha256[65];
    char root_path[PATH_MAX];
} execution_plan_v4_request_t;

typedef struct {
    const unsigned char *manifest;
    size_t manifest_bytes;
    const unsigned char *sidecar;
    size_t sidecar_bytes;
    const unsigned char *payload;
    size_t payload_bytes;
    uint64_t request_sequence;
    uint64_t request_output_elements;
    uint32_t work_unit_count;
    unsigned char manifest_sha256[32];
    unsigned char sidecar_sha256[32];
    unsigned char payload_sha256[32];
} execution_plan_v4_embedded_request_t;

int execution_plan_v4_request_load(
    const char *session_root,
    const char *manifest_relative_path,
    const char *submitted_manifest_sha256,
    uint32_t expected_dpus,
    uint32_t expected_tasklets,
    execution_plan_v4_request_t *request,
    char **error_message
);

int execution_plan_v4_request_load_payloads(
    execution_plan_v4_request_t *request,
    char **error_message
);

int execution_plan_v4_request_validate_embedded(
    const char *session_root,
    const execution_plan_v4_embedded_request_t *embedded,
    uint32_t expected_dpus,
    uint32_t expected_tasklets,
    char **error_message
);

/* Request outputs must be zero-initialized or returned by request_free(). */
int execution_plan_v4_request_prepare_embedded(
    const char *session_root,
    const execution_plan_v4_embedded_request_t *embedded,
    uint32_t expected_dpus,
    uint32_t expected_tasklets,
    execution_plan_v4_request_t *request,
    char **error_message
);

int execution_plan_v4_request_materialize_prepared(
    execution_plan_v4_request_t *request,
    char **error_message
);

void execution_plan_v4_request_release_payloads(
    execution_plan_v4_request_t *request
);

int execution_plan_v4_request_load_embedded(
    const char *session_root,
    const execution_plan_v4_embedded_request_t *embedded,
    uint32_t expected_dpus,
    uint32_t expected_tasklets,
    execution_plan_v4_request_t *request,
    char **error_message
);

int execution_plan_v4_request_write_output(
    const execution_plan_v4_request_item_t *item,
    char **error_message
);

void execution_plan_v4_request_free(execution_plan_v4_request_t *request);

#endif
