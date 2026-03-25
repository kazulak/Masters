#ifndef VERIFY_RUNNER_H
#define VERIFY_RUNNER_H

#include <quest.h>
#include <stdbool.h>

// Add this line so the universal_runner knows the function exists!
void run_test_suite(const char* mode);

// Keep your existing circuit verifier
bool verify_circuit(const char* algo, Qureg qubits, int n);

#endif // VERIFY_RUNNER_H