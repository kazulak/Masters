"""UPMEM target namespace.

Import public functions from their owning modules. Keeping this namespace inert
prevents an active executor import from loading historical TaskGraph adapters.
"""
