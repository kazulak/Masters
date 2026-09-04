from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_management_init_resets_once_for_all_tasklet_counts(tmp_path: Path) -> None:
    """Compile the real wrapper and pinned initializer against observable SDK stubs."""
    (tmp_path / "defs.h").write_text("unsigned int me(void);\n", encoding="ascii")
    (tmp_path / "alloc.h").write_text("void mem_reset(void);\n", encoding="ascii")
    for header in ("mram.h", "barrier.h"):
        (tmp_path / header).write_text("", encoding="ascii")
    harness = tmp_path / "harness.c"
    harness.write_text(
        """
#include <assert.h>
static unsigned int tasklet;
static unsigned int resets;
unsigned int me(void) { return tasklet; }
void mem_reset(void) { resets++; }
int management_init_entry(void);
int main(void) {
    for (unsigned int count = 1; count <= 24; count++) {
        resets = 0;
        /* Exercise non-owner returns before the owner runs. */
        for (unsigned int i = count; i > 0; i--) {
            tasklet = i - 1;
            assert(management_init_entry() == 0);
            assert(resets == (tasklet == 0 ? 1u : 0u));
        }
        assert(resets == 1);
    }
    return 0;
}
""",
        encoding="ascii",
    )
    # Rename only the exported entry point after preprocessing its upstream include.
    preprocessed = subprocess.run(
        [
            "gcc", "-E", "-P", "-I", str(tmp_path),
            "-I", str(ROOT / "external/SimplePIM/lib/management"),
            str(ROOT / "native/upmem/runtime/management_init.c"),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    wrapper = tmp_path / "wrapper.c"
    wrapper.write_text(preprocessed.replace("int main(void)", "int management_init_entry(void)"), encoding="ascii")
    executable = tmp_path / "check-init"
    subprocess.run(
        ["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", str(wrapper), str(harness), "-o", str(executable)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
