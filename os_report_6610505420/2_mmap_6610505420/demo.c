/*
 * demo.c - a small target program whose memory map changes step by step.
 * Watch it with:  python3 mmap.py --run ./demo --watch 0.5
 *
 * OS Report 2026, Part 2 (Memory Map Visualization) - student ID 6610505420
 * Made for education only. This is not production software.
 * AI disclosure: written with the help of Claude Code (Anthropic, Claude Opus 5.5).
 *
 * Safety: uses at most about 130 MiB of RAM, creates one temporary file that is
 * deleted immediately (before it is even mapped), and frees everything at the end.
 *
 * Usage: ./demo [-s SECONDS_PER_STEP] [-i]     (-i waits for Enter between steps)
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

#ifndef PR_SET_VMA
#define PR_SET_VMA 0x53564d41
#define PR_SET_VMA_ANON_NAME 0
#endif

#define MiB ((size_t)1 << 20)
#define N_THREADS 4
#define N_SMALL 4000

static char data_buf[256 * 1024] = {1};    /* .data : initialised global (256 KiB)   */
static char bss_buf[4 * 1024 * 1024];      /* .bss  : zero-initialised global (4 MiB) */
static const char rodata_msg[] = "this string lives in .rodata";

static double step_seconds = 2.0;
static int interactive = 0;
static volatile int stop_threads = 0;
static int step_no = 0;

static void step(const char *what) {
    printf("[demo pid %d] step %d: %s\n", getpid(), step_no++, what);
    fflush(stdout);
    if (interactive) {
        printf("    press Enter to continue...");
        fflush(stdout);
        int c;
        while ((c = getchar()) != '\n' && c != EOF) {}
    } else {
        usleep((useconds_t)(step_seconds * 1e6));
    }
}

/* Give an anonymous mapping a name that shows up in /proc/PID/maps as [anon:NAME]
 * (Linux >= 5.17 with CONFIG_ANON_VMA_NAME). Errors are ignored on older kernels. */
static int name_region(void *p, size_t len, const char *name) {
    return prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, (unsigned long)p, len, (unsigned long)name) == 0;
}

/* Write one byte per page so that the kernel really allocates the page (first touch). */
static void touch(char *p, size_t len) {
    for (size_t i = 0; i < len; i += 4096) p[i] = 1;
}

static void *worker(void *arg) {
    long id = (long)arg;
    volatile char local[64 * 1024];            /* 64 KiB on this thread's own stack */
    memset((char *)local, (int)id, sizeof local);
    /* 16 x 64 KiB: each block is below the mmap threshold (128 KiB), so it comes
     * from this thread's own malloc arena, which grows inside its 64 MiB reservation. */
    char *m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = malloc(64 * 1024);
        if (m[i]) memset(m[i], (int)id, 64 * 1024);
    }
    while (!stop_threads) usleep(50 * 1000);
    for (int i = 0; i < 16; i++) free(m[i]);
    return NULL;
}

/* Each call puts 16 KiB on the main stack; the deepest call pauses so the
 * grown [stack] region can be observed. */
static int grow_stack(int depth) {
    volatile char frame[16 * 1024];
    memset((char *)frame, depth, sizeof frame);
    if (depth == 0) {
        step("main stack grown by ~2 MiB with recursion ([stack] RSS increases)");
        return frame[0];
    }
    return grow_stack(depth - 1) + frame[1];
}

int main(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "s:ih")) != -1) {
        if (opt == 's') step_seconds = atof(optarg);
        else if (opt == 'i') interactive = 1;
        else {
            printf("usage: %s [-s seconds_per_step] [-i]\n", argv[0]);
            return opt == 'h' ? 0 : 1;
        }
    }
    if (step_seconds < 0.1) step_seconds = 0.1;
    if (step_seconds > 60) step_seconds = 60;

    step("program started: code, .rodata, .data, .bss and shared libraries are mapped");

    memset(bss_buf, 7, sizeof bss_buf);
    data_buf[0] = rodata_msg[0];
    step("touched the 4 MiB .bss array (its RSS goes from 0 to 4 MiB)");

    static char *small[N_SMALL];
    for (int i = 0; i < N_SMALL; i++) {
        small[i] = malloc(512);
        if (small[i]) memset(small[i], i, 512);
    }
    step("4000 small malloc(512): the [heap] grows with brk()");

    char *big = malloc(64 * MiB);
    if (!big) { perror("malloc"); return 1; }
    step("malloc(64 MiB): served by mmap(), virtual size +64 MiB but RSS unchanged");

    for (int q = 1; q <= 4; q++) {
        touch(big + (size_t)(q - 1) * 16 * MiB, 16 * MiB);
        char msg[96];
        snprintf(msg, sizeof msg, "touched %d/4 of the 64 MiB block (demand paging, RSS +16 MiB)", q);
        step(msg);
    }

    char *named = mmap(NULL, 32 * MiB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (named == MAP_FAILED) { perror("mmap"); return 1; }
    int named_ok = name_region(named, 32 * MiB, "demo-buffer");
    touch(named, 8 * MiB);
    step(named_ok ? "mmap(32 MiB) named [anon:demo-buffer], only the first 8 MiB touched"
                  : "mmap(32 MiB), only the first 8 MiB touched (kernel cannot name it, so it may merge "
                    "with a neighbouring anonymous region)");

    pthread_t th[N_THREADS];
    for (long i = 0; i < N_THREADS; i++) pthread_create(&th[i], NULL, worker, (void *)i);
    step("4 threads started: each gets a stack (with a guard page) and a malloc arena");

    char path[] = "/tmp/mmap_demo_XXXXXX";
    int fd = mkstemp(path);
    char *fmap = MAP_FAILED;
    if (fd >= 0) {
        unlink(path);                          /* the file disappears from /tmp right away */
        if (ftruncate(fd, (off_t)(8 * MiB)) == 0) {
            fmap = mmap(NULL, 8 * MiB, PROT_READ, MAP_PRIVATE, fd, 0);
            if (fmap != MAP_FAILED) {
                volatile char sum = 0;
                for (size_t i = 0; i < 8 * MiB; i += 4096) sum += fmap[i];
                (void)sum;
            }
        }
        close(fd);                             /* the mapping keeps the file alive */
    }
    step("mmap of an 8 MiB file (already deleted from /tmp): file-backed region");

    char *shared = mmap(NULL, 16 * MiB, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared != MAP_FAILED) touch(shared, 16 * MiB);
    step("mmap(MAP_SHARED | MAP_ANONYMOUS, 16 MiB): shared memory (shows as /dev/zero)");

    /* Changing the protection of the middle of a region splits one VMA into three. */
    mprotect(named + 8 * MiB, 8 * MiB, PROT_READ);
    step("mprotect(PROT_READ) on the middle 8 MiB of the 32 MiB buffer: one region is split into three");

    grow_stack(128);

    stop_threads = 1;
    for (int i = 0; i < N_THREADS; i++) pthread_join(th[i], NULL);
    step("threads joined (glibc may keep their stacks cached for reuse)");

    free(big);
    munmap(named, 32 * MiB);
    if (fmap != MAP_FAILED) munmap(fmap, 8 * MiB);
    if (shared != MAP_FAILED) munmap(shared, 16 * MiB);
    step("free(64 MiB) and munmap(): those regions disappear from the map");

    for (int i = 0; i < N_SMALL; i++) free(small[i]);
    malloc_trim(0);
    step("free() of the small blocks + malloc_trim(0): the [heap] shrinks");

    printf("[demo pid %d] done\n", getpid());
    return 0;
}
