/*
 * parallel.c - Parallel dense matrix multiplication C = A x B with OpenMP (C11)
 *
 * OS Report 2026, Part 1 (Intro to Parallel Programming) - student ID 6610505420
 * Made for education only. This is not production software.
 *
 * AI disclosure: this source file was written with the help of an AI coding
 * assistant (Claude Code, Anthropic, model Claude Opus 5.5). The student
 * reviewed, compiled, tested and benchmarked it.
 *
 * Build : make                 (or: gcc -O3 -march=native -fopenmp parallel.c -o parallel -lm)
 * Run   : ./parallel -n 800 -t 8 -a blocked -b 64
 * Help  : ./parallel -h
 *
 * Algorithms (all compute C = A x B for square n x n row-major double matrices):
 *   naive     - textbook i-j-k triple loop, inner loop walks B by column (stride n)
 *   ikj       - loop interchange i-k-j, inner loop walks B and C by row (stride 1, SIMD friendly)
 *   transpose - copy B into B^T first, then every C[i][j] is a dot product of two rows
 *   blocked   - cache tiling: (bs x bs) tiles of C are distributed over threads, i-k-j inside a tile
 *
 * Every program phase is timed separately so the serial part and the parallel
 * part can be measured for the Amdahl's-law analysis:
 *   alloc  (serial)   - allocate A, B, C (+ B^T)
 *   spawn  (overhead) - first OpenMP parallel region, creates the thread team
 *   init   (serial, or parallel with --par-init) - fill A and B, first touch of pages
 *   compute (parallel) - the multiplication itself (best of -r repetitions)
 *   checksum (serial) - sum of all elements of C
 * Verification (Freivalds or full) is timed separately and not part of the total.
 *
 * Safety: the program only allocates memory and computes. It refuses sizes that
 * would need more than 50% of physical RAM and limits the thread count, so it
 * cannot exhaust the machine. It does not change any system setting.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#define MAX_N 20000
#define MAX_THREADS 256
#define MAX_REPS 1000
#define MEM_LIMIT_FRACTION 0.5          /* never use more than 50% of physical RAM */
#define HUGE_PAGE_SIZE ((size_t)2 << 20) /* 2 MiB transparent huge page on x86-64 */

#define SEED_A 0x1234567ULL
#define SEED_B 0x89ABCDEULL
#define SEED_X 0x5EED5EEDULL

typedef enum { ALG_NAIVE, ALG_IKJ, ALG_TRANSPOSE, ALG_BLOCKED, ALG_COUNT } algo_t;
static const char *ALGO_NAMES[ALG_COUNT] = {"naive", "ikj", "transpose", "blocked"};

typedef enum { THP_DEFAULT, THP_ON, THP_OFF } thp_t;
static const char *THP_NAMES[] = {"default", "on", "off"};

typedef enum { VERIFY_NONE, VERIFY_FREIVALDS, VERIFY_FULL } verify_t;
static const char *VERIFY_NAMES[] = {"none", "freivalds", "full"};

typedef struct {
    int n;
    int threads;
    algo_t algo;
    int block;
    omp_sched_t sched;
    int chunk;
    int reps;
    int par_init;
    thp_t thp;
    verify_t verify;
    int csv;
    int header;
    int verbose;
} config_t;

typedef struct {
    double t_alloc, t_spawn, t_init, t_compute_min, t_compute_avg, t_checksum, t_total, t_verify;
    double gflops, checksum, verify_err;
    int verify_ok;
    int threads_actual;
    double mem_theory_mb;
    double vm_hwm_mb, vm_peak_mb, anon_huge_mb;
    long minflt, majflt, nvcsw, nivcsw;
} result_t;

/* ------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* ------------------------------------------------------------------------- */

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(EXIT_FAILURE);
}

/* Deterministic pseudo-random value in [-1, 1) for element idx (splitmix64 hash).
 * A hash (not a sequential generator) lets serial and parallel init produce
 * exactly the same matrices. */
static inline double value_at(uint64_t seed, uint64_t idx) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (idx + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return (double)(z >> 11) * 0x1.0p-53 * 2.0 - 1.0;
}

/* Read a "Key:   value kB" line from a /proc file, return value in kB or -1. */
static long read_proc_kb(const char *path, const char *key) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    size_t klen = strlen(key);
    long val = -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, klen) == 0 && line[klen] == ':') {
            val = strtol(line + klen + 1, NULL, 10);
            break;
        }
    }
    fclose(f);
    return val;
}

static double physical_ram_bytes(void) {
    long pages = sysconf(_SC_PHYS_PAGES);
    long psize = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || psize <= 0) return 0.0;
    return (double)pages * (double)psize;
}

/* Allocate an n_elems array of doubles. Large arrays are aligned to 2 MiB so a
 * transparent huge page can back them; with --thp on/off we ask the kernel
 * (madvise) to use or avoid huge pages for this range only. */
static double *alloc_matrix(size_t n_elems, thp_t thp, int verbose) {
    size_t bytes = n_elems * sizeof(double);
    size_t align = bytes >= HUGE_PAGE_SIZE ? HUGE_PAGE_SIZE : 64;
    void *p = NULL;
    if (posix_memalign(&p, align, bytes) != 0) return NULL;
    if (thp != THP_DEFAULT && bytes >= HUGE_PAGE_SIZE) {
        int advice = (thp == THP_ON) ? MADV_HUGEPAGE : MADV_NOHUGEPAGE;
        if (madvise(p, bytes, advice) != 0 && verbose)
            fprintf(stderr, "warning: madvise(%s) failed: %s\n", THP_NAMES[thp], strerror(errno));
    }
    return (double *)p;
}

/* ------------------------------------------------------------------------- */
/* Initialisation and the four multiplication kernels                         */
/* ------------------------------------------------------------------------- */

static void init_matrices(int n, double *restrict A, double *restrict B, int parallel) {
    size_t nn = (size_t)n * n;
    /* if(parallel): with --par-init the loop runs on all threads, otherwise on one. */
#pragma omp parallel for schedule(static) if (parallel)
    for (size_t i = 0; i < nn; i++) {
        A[i] = value_at(SEED_A, i);
        B[i] = value_at(SEED_B, i);
    }
}

/* naive i-j-k: C[i][j] = sum_k A[i][k] * B[k][j]; B accessed with stride n. */
static void matmul_naive(int n, const double *restrict A, const double *restrict B, double *restrict C) {
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += A[(size_t)i * n + k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = s;
        }
    }
}

/* i-k-j: row i of C accumulates a * row k of B; all inner accesses are stride 1. */
static void matmul_ikj(int n, const double *restrict A, const double *restrict B, double *restrict C) {
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++) {
        double *restrict ci = C + (size_t)i * n;
        const double *restrict ai = A + (size_t)i * n;
        memset(ci, 0, (size_t)n * sizeof(double));
        for (int k = 0; k < n; k++) {
            const double a = ai[k];
            const double *restrict bk = B + (size_t)k * n;
            for (int j = 0; j < n; j++) ci[j] += a * bk[j];
        }
    }
}

/* transpose: BT = B^T (parallel), then C[i][j] = dot(row i of A, row j of BT). */
static void matmul_transpose(int n, const double *restrict A, const double *restrict B,
                             double *restrict BT, double *restrict C) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) BT[(size_t)j * n + i] = B[(size_t)i * n + j];

#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++) {
        const double *restrict ai = A + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double *restrict bj = BT + (size_t)j * n;
            double s = 0.0;
            /* omp simd allows the compiler to reorder the sum and vectorise it */
#pragma omp simd reduction(+ : s)
            for (int k = 0; k < n; k++) s += ai[k] * bj[k];
            C[(size_t)i * n + j] = s;
        }
    }
}

/* blocked (tiled) i-k-j. Each (ib, jb) tile of C is owned by exactly one thread,
 * so no synchronisation is needed. The k dimension is also tiled so that the
 * working set (a bs x bs tile of A, B and C) stays in the L1/L2 cache. */
static void matmul_blocked(int n, int bs, const double *restrict A, const double *restrict B,
                           double *restrict C) {
    const int nb = (n + bs - 1) / bs;
#pragma omp parallel for collapse(2) schedule(runtime)
    for (int ib = 0; ib < nb; ib++) {
        for (int jb = 0; jb < nb; jb++) {
            const int i0 = ib * bs, i1 = (i0 + bs < n) ? i0 + bs : n;
            const int j0 = jb * bs, j1 = (j0 + bs < n) ? j0 + bs : n;
            for (int i = i0; i < i1; i++)
                memset(C + (size_t)i * n + j0, 0, (size_t)(j1 - j0) * sizeof(double));
            for (int k0 = 0; k0 < n; k0 += bs) {
                const int k1 = (k0 + bs < n) ? k0 + bs : n;
                for (int i = i0; i < i1; i++) {
                    double *restrict ci = C + (size_t)i * n;
                    const double *restrict ai = A + (size_t)i * n;
                    for (int k = k0; k < k1; k++) {
                        const double a = ai[k];
                        const double *restrict bk = B + (size_t)k * n;
                        for (int j = j0; j < j1; j++) ci[j] += a * bk[j];
                    }
                }
            }
        }
    }
}

static void run_kernel(const config_t *cfg, const double *A, const double *B, double *BT, double *C) {
    switch (cfg->algo) {
    case ALG_NAIVE: matmul_naive(cfg->n, A, B, C); break;
    case ALG_IKJ: matmul_ikj(cfg->n, A, B, C); break;
    case ALG_TRANSPOSE: matmul_transpose(cfg->n, A, B, BT, C); break;
    case ALG_BLOCKED: matmul_blocked(cfg->n, cfg->block, A, B, C); break;
    default: die("unknown algorithm");
    }
}

/* ------------------------------------------------------------------------- */
/* Verification                                                               */
/* ------------------------------------------------------------------------- */

/* Freivalds' check: for a random vector x, A(Bx) must equal Cx. O(n^2) work
 * instead of O(n^3), so it can be used for every n. Returns relative error. */
static double verify_freivalds(int n, const double *A, const double *B, const double *C) {
    double *x = malloc((size_t)n * sizeof(double));
    double *bx = malloc((size_t)n * sizeof(double));
    double *abx = malloc((size_t)n * sizeof(double));
    double *cx = malloc((size_t)n * sizeof(double));
    if (!x || !bx || !abx || !cx) die("out of memory in verification");
    for (int j = 0; j < n; j++) x[j] = value_at(SEED_X, (uint64_t)j);
    for (int i = 0; i < n; i++) {
        double s1 = 0.0, s2 = 0.0;
        for (int k = 0; k < n; k++) {
            s1 += B[(size_t)i * n + k] * x[k];
            s2 += C[(size_t)i * n + k] * x[k];
        }
        bx[i] = s1;
        cx[i] = s2;
    }
    double max_diff = 0.0, max_ref = 0.0;
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int k = 0; k < n; k++) s += A[(size_t)i * n + k] * bx[k];
        abx[i] = s;
        double d = fabs(abx[i] - cx[i]);
        if (d > max_diff) max_diff = d;
        if (fabs(abx[i]) > max_ref) max_ref = fabs(abx[i]);
    }
    free(x); free(bx); free(abx); free(cx);
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

/* Full check against a single-threaded i-k-j reference. O(n^3), use for small n. */
static double verify_full(int n, const double *A, const double *B, const double *C) {
    size_t nn = (size_t)n * n;
    double *R = calloc(nn, sizeof(double));
    if (!R) die("out of memory in verification");
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            const double a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++) R[(size_t)i * n + j] += a * B[(size_t)k * n + j];
        }
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < nn; i++) {
        double d = fabs(R[i] - C[i]);
        if (d > max_diff) max_diff = d;
        if (fabs(R[i]) > max_ref) max_ref = fabs(R[i]);
    }
    free(R);
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

/* ------------------------------------------------------------------------- */
/* Command line                                                               */
/* ------------------------------------------------------------------------- */

static void usage(const char *prog) {
    printf("Usage: %s [options]\n"
           "  -n N            matrix size n (n x n), 1..%d             [default 800]\n"
           "  -t T            number of OpenMP threads, 1..%d          [default: all CPUs]\n"
           "  -a ALGO         naive | ikj | transpose | blocked          [default blocked]\n"
           "  -b BS           tile size for 'blocked'                    [default 64]\n"
           "  -s KIND         OpenMP schedule: static | dynamic | guided [default static]\n"
           "  -c CHUNK        schedule chunk size, 0 = runtime default   [default 0]\n"
           "  -r R            repeat the multiplication R times, report best and mean [default 1]\n"
           "  -p, --par-init  initialise A and B in parallel (default: serial)\n"
           "  --thp MODE      transparent huge pages via madvise: default | on | off\n"
           "  --verify MODE   none | freivalds | full                    [default freivalds]\n"
           "  --csv           print one CSV line instead of the text report\n"
           "  --header        print the CSV header line and exit\n"
           "  -v              verbose (show which CPU each thread starts on)\n"
           "  -h              this help\n"
           "OpenMP environment variables such as OMP_PROC_BIND, OMP_PLACES and\n"
           "OMP_WAIT_POLICY are honoured (they are set outside the program).\n",
           prog, MAX_N, MAX_THREADS);
}

static int parse_int(const char *s, const char *what, int lo, int hi) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end != '\0' || v < lo || v > hi) {
        fprintf(stderr, "error: %s must be an integer in [%d, %d], got '%s'\n", what, lo, hi, s);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

static void parse_args(int argc, char **argv, config_t *cfg) {
    enum { OPT_THP = 1000, OPT_VERIFY, OPT_CSV, OPT_HEADER };
    static struct option longopts[] = {
        {"par-init", no_argument, NULL, 'p'},  {"thp", required_argument, NULL, OPT_THP},
        {"verify", required_argument, NULL, OPT_VERIFY}, {"csv", no_argument, NULL, OPT_CSV},
        {"header", no_argument, NULL, OPT_HEADER}, {"help", no_argument, NULL, 'h'},
        {NULL, 0, NULL, 0}};
    int c;
    while ((c = getopt_long(argc, argv, "n:t:a:b:s:c:r:pvh", longopts, NULL)) != -1) {
        switch (c) {
        case 'n': cfg->n = parse_int(optarg, "n", 1, MAX_N); break;
        case 't': cfg->threads = parse_int(optarg, "threads", 1, MAX_THREADS); break;
        case 'b': cfg->block = parse_int(optarg, "block size", 1, 4096); break;
        case 'c': cfg->chunk = parse_int(optarg, "chunk", 0, 1 << 20); break;
        case 'r': cfg->reps = parse_int(optarg, "reps", 1, MAX_REPS); break;
        case 'p': cfg->par_init = 1; break;
        case 'v': cfg->verbose = 1; break;
        case 'a': {
            int found = 0;
            for (int i = 0; i < ALG_COUNT; i++)
                if (strcmp(optarg, ALGO_NAMES[i]) == 0) { cfg->algo = (algo_t)i; found = 1; }
            if (!found) die("algorithm must be naive, ikj, transpose or blocked");
            break;
        }
        case 's':
            if (strcmp(optarg, "static") == 0) cfg->sched = omp_sched_static;
            else if (strcmp(optarg, "dynamic") == 0) cfg->sched = omp_sched_dynamic;
            else if (strcmp(optarg, "guided") == 0) cfg->sched = omp_sched_guided;
            else die("schedule must be static, dynamic or guided");
            break;
        case OPT_THP:
            if (strcmp(optarg, "default") == 0) cfg->thp = THP_DEFAULT;
            else if (strcmp(optarg, "on") == 0) cfg->thp = THP_ON;
            else if (strcmp(optarg, "off") == 0) cfg->thp = THP_OFF;
            else die("--thp must be default, on or off");
            break;
        case OPT_VERIFY:
            if (strcmp(optarg, "none") == 0) cfg->verify = VERIFY_NONE;
            else if (strcmp(optarg, "freivalds") == 0) cfg->verify = VERIFY_FREIVALDS;
            else if (strcmp(optarg, "full") == 0) cfg->verify = VERIFY_FULL;
            else die("--verify must be none, freivalds or full");
            break;
        case OPT_CSV: cfg->csv = 1; break;
        case OPT_HEADER: cfg->header = 1; break;
        case 'h': usage(argv[0]); exit(EXIT_SUCCESS);
        default: usage(argv[0]); exit(EXIT_FAILURE);
        }
    }
    if (optind < argc) die("unexpected extra argument (see -h)");
}

static const char *sched_name(omp_sched_t s) {
    switch (s & ~omp_sched_monotonic) {
    case omp_sched_static: return "static";
    case omp_sched_dynamic: return "dynamic";
    case omp_sched_guided: return "guided";
    default: return "auto";
    }
}

static const char *bind_name(omp_proc_bind_t b) {
    switch (b) {
    case omp_proc_bind_false: return "false";
    case omp_proc_bind_true: return "true";
    case omp_proc_bind_master: return "master";
    case omp_proc_bind_close: return "close";
    case omp_proc_bind_spread: return "spread";
    default: return "unknown";
    }
}

static const char *CSV_HEADER =
    "algo,n,threads,schedule,chunk,block,par_init,thp,reps,proc_bind,places,wait_policy,"
    "t_alloc,t_spawn,t_init,t_compute_min,t_compute_avg,t_checksum,t_total,gflops,"
    "mem_theory_mb,vm_hwm_mb,vm_peak_mb,anon_huge_mb,minflt,majflt,nvcsw,nivcsw,"
    "checksum,verify,verify_err,verify_ok,t_verify";

static const char *env_or(const char *name, const char *fallback) {
    const char *v = getenv(name);
    return (v && *v) ? v : fallback;
}

static void print_csv(const config_t *cfg, const result_t *r) {
    printf("%s,%d,%d,%s,%d,%d,%d,%s,%d,%s,%s,%s,"
           "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.4f,"
           "%.3f,%.3f,%.3f,%.3f,%ld,%ld,%ld,%ld,"
           "%.17g,%s,%.3e,%d,%.6f\n",
           ALGO_NAMES[cfg->algo], cfg->n, r->threads_actual, sched_name(cfg->sched), cfg->chunk,
           cfg->algo == ALG_BLOCKED ? cfg->block : 0, cfg->par_init, THP_NAMES[cfg->thp], cfg->reps,
           bind_name(omp_get_proc_bind()), env_or("OMP_PLACES", "unset"),
           env_or("OMP_WAIT_POLICY", "unset"), r->t_alloc, r->t_spawn, r->t_init, r->t_compute_min,
           r->t_compute_avg, r->t_checksum, r->t_total, r->gflops, r->mem_theory_mb, r->vm_hwm_mb,
           r->vm_peak_mb, r->anon_huge_mb, r->minflt, r->majflt, r->nvcsw, r->nivcsw, r->checksum,
           VERIFY_NAMES[cfg->verify], r->verify_err, r->verify_ok, r->t_verify);
}

static void print_report(const config_t *cfg, const result_t *r) {
    double serial = r->t_alloc + r->t_checksum + (cfg->par_init ? 0.0 : r->t_init);
    printf("=== Parallel matrix multiplication (OpenMP) ===\n");
    printf("n = %d, threads = %d, algorithm = %s", cfg->n, r->threads_actual, ALGO_NAMES[cfg->algo]);
    if (cfg->algo == ALG_BLOCKED) printf(" (tile %d)", cfg->block);
    printf(", schedule = %s,%d, init = %s, THP = %s\n", sched_name(cfg->sched), cfg->chunk,
           cfg->par_init ? "parallel" : "serial", THP_NAMES[cfg->thp]);
    printf("OMP_PROC_BIND = %s, OMP_PLACES = %s, OMP_WAIT_POLICY = %s\n",
           bind_name(omp_get_proc_bind()), env_or("OMP_PLACES", "unset"), env_or("OMP_WAIT_POLICY", "unset"));
    printf("\n-- time (seconds) --\n");
    printf("  alloc      %12.6f  serial\n", r->t_alloc);
    printf("  spawn      %12.6f  thread-team start-up\n", r->t_spawn);
    printf("  init       %12.6f  %s\n", r->t_init, cfg->par_init ? "parallel" : "serial");
    printf("  compute    %12.6f  parallel, best of %d (mean %.6f)\n", r->t_compute_min, cfg->reps, r->t_compute_avg);
    printf("  checksum   %12.6f  serial\n", r->t_checksum);
    printf("  total      %12.6f  (serial share %.2f%%)\n", r->t_total, 100.0 * serial / r->t_total);
    printf("  performance %11.3f  GFLOP/s (2n^3 / compute)\n", r->gflops);
    printf("\n-- memory --\n");
    printf("  matrices (theory)  %10.2f MiB\n", r->mem_theory_mb);
    printf("  peak RSS (VmHWM)   %10.2f MiB\n", r->vm_hwm_mb);
    printf("  peak virtual       %10.2f MiB\n", r->vm_peak_mb);
    printf("  AnonHugePages      %10.2f MiB\n", r->anon_huge_mb);
    printf("  page faults        minor %ld, major %ld\n", r->minflt, r->majflt);
    printf("  context switches   voluntary %ld, involuntary %ld\n", r->nvcsw, r->nivcsw);
    printf("\n-- result --\n");
    printf("  checksum (sum of C) = %.17g\n", r->checksum);
    if (cfg->verify == VERIFY_NONE)
        printf("  verification: skipped\n");
    else
        printf("  verification (%s): relative error %.3e -> %s (%.3f s)\n", VERIFY_NAMES[cfg->verify],
               r->verify_err, r->verify_ok ? "PASS" : "FAIL", r->t_verify);
}

/* ------------------------------------------------------------------------- */
/* main                                                                       */
/* ------------------------------------------------------------------------- */

int main(int argc, char **argv) {
    config_t cfg = {.n = 800, .threads = 0, .algo = ALG_BLOCKED, .block = 64,
                    .sched = omp_sched_static, .chunk = 0, .reps = 1, .par_init = 0,
                    .thp = THP_DEFAULT, .verify = VERIFY_FREIVALDS};
    parse_args(argc, argv, &cfg);
    if (cfg.header) { puts(CSV_HEADER); return EXIT_SUCCESS; }
    if (cfg.threads == 0) cfg.threads = omp_get_num_procs();
    if (cfg.threads > MAX_THREADS) cfg.threads = MAX_THREADS;

    /* Safety check: refuse jobs that would use more than half of the RAM. */
    const size_t nn = (size_t)cfg.n * cfg.n;
    int n_mats = 3 + (cfg.algo == ALG_TRANSPOSE) + (cfg.verify == VERIFY_FULL);
    double need = (double)n_mats * (double)nn * sizeof(double);
    double ram = physical_ram_bytes();
    if (ram > 0 && need > MEM_LIMIT_FRACTION * ram) {
        fprintf(stderr, "error: n=%d needs %.0f MiB but the limit is %.0f MiB (50%% of RAM)\n",
                cfg.n, need / 1048576.0, MEM_LIMIT_FRACTION * ram / 1048576.0);
        return EXIT_FAILURE;
    }

    omp_set_num_threads(cfg.threads);
    omp_set_schedule(cfg.sched, cfg.chunk);
    result_t r = {0};
    r.mem_theory_mb = (double)(3 + (cfg.algo == ALG_TRANSPOSE)) * nn * sizeof(double) / 1048576.0;

    /* Phase 1: allocation (serial) */
    double t0 = omp_get_wtime();
    double *A = alloc_matrix(nn, cfg.thp, cfg.verbose);
    double *B = alloc_matrix(nn, cfg.thp, cfg.verbose);
    double *C = alloc_matrix(nn, cfg.thp, cfg.verbose);
    double *BT = cfg.algo == ALG_TRANSPOSE ? alloc_matrix(nn, cfg.thp, cfg.verbose) : NULL;
    if (!A || !B || !C || (cfg.algo == ALG_TRANSPOSE && !BT)) die("out of memory");
    double t1 = omp_get_wtime();

    /* Phase 2: start the thread team (first parallel region creates the threads) */
    int cpu_of[MAX_THREADS];
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        if (id < MAX_THREADS) cpu_of[id] = sched_getcpu();
#pragma omp single
        r.threads_actual = omp_get_num_threads();
    }
    double t2 = omp_get_wtime();

    /* Phase 3: initialise A and B (serial by default) */
    init_matrices(cfg.n, A, B, cfg.par_init);
    double t3 = omp_get_wtime();

    /* Phase 4: the multiplication, repeated cfg.reps times */
    double best = 1e300, sum = 0.0;
    for (int rep = 0; rep < cfg.reps; rep++) {
        double s = omp_get_wtime();
        run_kernel(&cfg, A, B, BT, C);
        double dt = omp_get_wtime() - s;
        sum += dt;
        if (dt < best) best = dt;
    }
    double t4 = omp_get_wtime();

    /* Phase 5: checksum (serial) */
    double cs = 0.0;
    for (size_t i = 0; i < nn; i++) cs += C[i];
    double t5 = omp_get_wtime();

    r.t_alloc = t1 - t0;
    r.t_spawn = t2 - t1;
    r.t_init = t3 - t2;
    r.t_compute_min = best;
    r.t_compute_avg = sum / cfg.reps;
    r.t_checksum = t5 - t4;
    r.t_total = r.t_alloc + r.t_spawn + r.t_init + r.t_compute_min + r.t_checksum;
    r.gflops = 2.0 * (double)cfg.n * cfg.n * cfg.n / best / 1e9;
    r.checksum = cs;

    /* Memory statistics are read now, while all matrices are still allocated. */
    r.vm_hwm_mb = read_proc_kb("/proc/self/status", "VmHWM") / 1024.0;
    r.vm_peak_mb = read_proc_kb("/proc/self/status", "VmPeak") / 1024.0;
    r.anon_huge_mb = read_proc_kb("/proc/self/smaps_rollup", "AnonHugePages") / 1024.0;
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    r.minflt = ru.ru_minflt;
    r.majflt = ru.ru_majflt;
    r.nvcsw = ru.ru_nvcsw;
    r.nivcsw = ru.ru_nivcsw;

    /* Verification (not part of the timed program) */
    double tv = omp_get_wtime();
    if (cfg.verify == VERIFY_FREIVALDS) r.verify_err = verify_freivalds(cfg.n, A, B, C);
    else if (cfg.verify == VERIFY_FULL) r.verify_err = verify_full(cfg.n, A, B, C);
    r.t_verify = omp_get_wtime() - tv;
    r.verify_ok = cfg.verify == VERIFY_NONE ? 1 : (r.verify_err < 1e-9);

    if (cfg.csv) {
        print_csv(&cfg, &r);
    } else {
        print_report(&cfg, &r);
        if (cfg.verbose) {
            printf("\n-- thread placement at start-up (thread -> CPU) --\n  ");
            for (int i = 0; i < r.threads_actual && i < MAX_THREADS; i++) printf("%d->%d ", i, cpu_of[i]);
            printf("\n");
        }
    }

    free(A); free(B); free(C); free(BT);
    return r.verify_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
