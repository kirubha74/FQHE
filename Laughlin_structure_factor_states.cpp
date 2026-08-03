
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int N = 16;
static constexpr int Q = N - 1;
static constexpr int X = 2 * N - 1;

static constexpr int TABLE_LOG2 = 33;

static constexpr int PREFIX_DEPTH = 4;

static constexpr int DUMP_STATES = 1;
static constexpr int DUMP_TEXT   = 0;

static constexpr std::uint64_t TABLE_SIZE = 1ull << TABLE_LOG2;
static constexpr std::uint64_t TABLE_MASK = TABLE_SIZE - 1;
static constexpr std::uint64_t OCCUPIED   = 1ull << 63;

static_assert(N >= PREFIX_DEPTH, "N must be at least PREFIX_DEPTH");
static_assert(N + X - 1 <= 63, "packed key must fit in 63 bits");

static std::atomic<std::uint64_t>* g_keys = nullptr;
static std::atomic<std::int64_t>*  g_vals = nullptr;

static inline std::uint64_t mix64(std::uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}

static inline void table_add(std::uint64_t key, std::int64_t delta) {
    const std::uint64_t marked = key | OCCUPIED;
    std::uint64_t i = mix64(key) & TABLE_MASK;
    for (;;) {
        std::uint64_t cur = g_keys[i].load(std::memory_order_relaxed);
        if (cur == marked) {
            g_vals[i].fetch_add(delta, std::memory_order_relaxed);
            return;
        }
        if (cur == 0) {
            std::uint64_t expected = 0;
            if (g_keys[i].compare_exchange_strong(expected, marked,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {
                g_vals[i].fetch_add(delta, std::memory_order_relaxed);
                return;
            }
            continue;
        }
        i = (i + 1) & TABLE_MASK;
    }
}

static inline std::uint64_t encode(const int* occ) {
    std::uint64_t key = 0;
    int bit = 0;
    for (int m = 0; m < X; ++m) {
        for (int k = 0; k < occ[m]; ++k) key |= (1ull << bit++);
        ++bit;
    }
    return key;
}

static inline void decode(std::uint64_t key, int* occ) {
    for (int m = 0; m < X; ++m) occ[m] = 0;
    int m = 0;
    for (int bit = 0; bit < N + X - 1; ++bit) {
        if (key & (1ull << bit)) ++occ[m];
        else ++m;
    }
}

struct Worker {
    int perm[N];
    int occ[X];
    int root[N];
    int sign;

    void dfs(int d) {
        if (d == N) {
            table_add(encode(occ), sign);
            return;
        }
        for (int i = d; i < N; ++i) {
            if (i != d) { std::swap(perm[d], perm[i]); sign = -sign; }
            const int m = root[d] + perm[d];
            ++occ[m];
            dfs(d + 1);
            --occ[m];
            if (i != d) { std::swap(perm[d], perm[i]); sign = -sign; }
        }
    }
};

static double lfact(int n) {
    static double c[64];
    static bool init = false;
    if (!init) { c[0] = 0.0; for (int i = 1; i < 64; ++i) c[i] = c[i-1] + std::log((double)i); init = true; }
    return c[n];
}

static double lbinom(int n, int k) {
    return lfact(n) - lfact(k) - lfact(n - k);
}

static double wigner3j(int j1, int j2, int j3, int m1, int m2, int m3) {
    if (m1 + m2 + m3 != 0) return 0.0;
    if (j3 < std::abs(j1 - j2) || j3 > j1 + j2) return 0.0;
    if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(m3) > j3) return 0.0;

    const double ldelta = 0.5 * (lfact(j1 + j2 - j3) + lfact(j1 - j2 + j3) +
                                 lfact(-j1 + j2 + j3) - lfact(j1 + j2 + j3 + 1));
    const double lpre = ldelta + 0.5 * (lfact(j1 + m1) + lfact(j1 - m1) +
                                        lfact(j2 + m2) + lfact(j2 - m2) +
                                        lfact(j3 + m3) + lfact(j3 - m3));

    const int kmin = std::max({0, j2 - j3 - m1, j1 - j3 + m2});
    const int kmax = std::min({j1 + j2 - j3, j1 - m1, j2 + m2});

    double sum = 0.0;
    for (int k = kmin; k <= kmax; ++k) {
        const double ld = lfact(k) + lfact(j1 + j2 - j3 - k) +
                          lfact(j1 - m1 - k) + lfact(j2 + m2 - k) +
                          lfact(j3 - j2 + m1 + k) + lfact(j3 - j1 - m2 + k);
        const double t = std::exp(lpre - ld);
        sum += (k % 2 == 0) ? t : -t;
    }
    const int pe = j1 - j2 - m3;
    return ((((pe % 2) + 2) % 2) == 0 ? 1.0 : -1.0) * sum;
}

int main() {
    const auto t0 = std::chrono::steady_clock::now();

#ifdef _OPENMP
    std::printf("threads = %d\n", omp_get_max_threads());
#endif
    std::printf("N = %d, orbitals = %d, table slots = %llu (%.1f GB)\n",
                N, X, (unsigned long long)TABLE_SIZE,
                16.0 * (double)TABLE_SIZE / 1e9);
    std::fflush(stdout);

    g_keys = new (std::nothrow) std::atomic<std::uint64_t>[TABLE_SIZE];
    g_vals = new (std::nothrow) std::atomic<std::int64_t>[TABLE_SIZE];
    if (!g_keys || !g_vals) { std::fprintf(stderr, "allocation failed\n"); return 1; }

#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < (std::int64_t)TABLE_SIZE; ++i) {
        g_keys[i].store(0, std::memory_order_relaxed);
        g_vals[i].store(0, std::memory_order_relaxed);
    }

    std::int64_t ntasks = 1;
    for (int d = 0; d < PREFIX_DEPTH; ++d) ntasks *= (N - d);
    std::printf("tasks = %lld\n", (long long)ntasks);
    std::fflush(stdout);

    std::atomic<std::int64_t> done{0};

#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t t = 0; t < ntasks; ++t) {
        Worker w;
        for (int i = 0; i < N; ++i) w.root[i] = N - 1 - i;
        for (int i = 0; i < N; ++i) w.perm[i] = w.root[i];
        for (int m = 0; m < X; ++m) w.occ[m] = 0;
        w.sign = 1;

        std::int64_t r = t;
        for (int d = 0; d < PREFIX_DEPTH; ++d) {
            const int span = N - d;
            const int i = d + (int)(r % span);
            r /= span;
            if (i != d) { std::swap(w.perm[d], w.perm[i]); w.sign = -w.sign; }
            ++w.occ[w.root[d] + w.perm[d]];
        }
        w.dfs(PREFIX_DEPTH);

        const std::int64_t d = done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (ntasks >= 100 && d % (ntasks / 100) == 0) {
            const double el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "  progress %3lld%%  elapsed %.0f s  eta %.0f s\n",
                         (long long)(100 * d / ntasks), el,
                         el * (double)(ntasks - d) / (double)d);
            std::fflush(stderr);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();

    std::vector<double> lbin(X);
    for (int m = 0; m < X; ++m) lbin[m] = lbinom(2 * Q, m);

    double lref = 0.5 * lfact(N);
    for (int m = 0; m < X; m += 2) lref -= 0.5 * lbin[m];

    const int Lmax = 2 * Q;
    std::vector<double> cLm((Lmax + 1) * X);
    for (int L = 0; L <= Lmax; ++L) {
        const double w2 = wigner3j(Q, Q, L, -Q, Q, 0);
        for (int im = 0; im < X; ++im) {
            const int m = im - Q;
            const int pe = 3 * Q + L + m;
            const double sgn = ((((pe % 2) + 2) % 2) == 0) ? 1.0 : -1.0;
            const double pref = sgn * std::sqrt((double)(2*Q+1) * (2*Q+1) * (2*L+1))
                                / std::sqrt(4.0 * M_PI);
            cLm[L * X + im] = pref * wigner3j(Q, Q, L, -m, m, 0) * w2;
        }
    }

    long double norm2 = 0.0L;
    std::vector<long double> expect(Lmax + 1, 0.0L);
    std::int64_t dim = 0;

#pragma omp parallel
    {
        long double loc_norm = 0.0L;
        std::vector<long double> loc_exp(Lmax + 1, 0.0L);
        std::int64_t loc_dim = 0;
        int occ[X];

#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < (std::int64_t)TABLE_SIZE; ++i) {
            const std::uint64_t k = g_keys[i].load(std::memory_order_relaxed);
            if (k == 0) continue;
            const std::int64_t c = g_vals[i].load(std::memory_order_relaxed);
            if (c == 0) continue;
            ++loc_dim;

            decode(k & ~OCCUPIED, occ);

            double la = 0.0;
            double lw = 0.5 * lfact(N);
            for (int m = 0; m < X; ++m) {
                if (!occ[m]) continue;
                la += lfact(occ[m]);
                lw -= 0.5 * lfact(occ[m]);
                lw -= 0.5 * occ[m] * lbin[m];
            }
            const double C = (double)c * std::exp(la + lw - lref);
            const long double C2 = (long double)C * C;
            loc_norm += C2;

            for (int L = 0; L <= Lmax; ++L) {
                const double* cm = &cLm[L * X];
                double diag = 0.0;
                for (int m = 0; m < X; ++m) diag += occ[m] * cm[m];
                loc_exp[L] += C2 * (long double)diag * diag;
            }
        }

#pragma omp critical
        {
            norm2 += loc_norm;
            dim += loc_dim;
            for (int L = 0; L <= Lmax; ++L) expect[L] += loc_exp[L];
        }
    }

    const auto t2 = std::chrono::steady_clock::now();

    if (DUMP_STATES) {
        const auto td0 = std::chrono::steady_clock::now();
        char dname[64];
        std::snprintf(dname, sizeof(dname), DUMP_TEXT ? "states_N%d.txt"
                                                      : "states_N%d.bin", N);
        std::FILE* dp = std::fopen(dname, DUMP_TEXT ? "w" : "wb");
        if (!dp) { std::fprintf(stderr, "cannot open %s\n", dname); return 1; }

        if (DUMP_TEXT) {
            std::fprintf(dp, "# occupation-basis decomposition of Delta^2\n");
            std::fprintf(dp, "# N = %d   orbitals = %d   states = %lld\n",
                         N, X, (long long)dim);
            std::fprintf(dp, "# columns: n_0 ... n_%d  a_lambda  C_normalised\n",
                         X - 1);
        } else {
            const std::uint64_t hdr[4] = { 0x4C415547484C4E31ull,
                                           (std::uint64_t)N,
                                           (std::uint64_t)X,
                                           (std::uint64_t)dim };
            std::fwrite(hdr, sizeof(std::uint64_t), 4, dp);
        }

        std::int64_t written = 0;
        int occ[X];
        for (std::uint64_t i = 0; i < TABLE_SIZE; ++i) {
            const std::uint64_t k = g_keys[i].load(std::memory_order_relaxed);
            if (k == 0) continue;
            const std::int64_t c = g_vals[i].load(std::memory_order_relaxed);
            if (c == 0) continue;

            decode(k & ~OCCUPIED, occ);

            double la = 0.0;
            double lw = 0.5 * lfact(N);
            for (int m = 0; m < X; ++m) {
                if (!occ[m]) continue;
                la += lfact(occ[m]);
                lw -= 0.5 * lfact(occ[m]);
                lw -= 0.5 * occ[m] * lbin[m];
            }
            std::int64_t mult = 1;
            for (int m = 0; m < X; ++m)
                for (int t = 2; t <= occ[m]; ++t) mult *= t;
            const std::int64_t a_lambda = c * mult;
            const double Cn = (double)c * std::exp(la + lw - lref);

            if (DUMP_TEXT) {
                for (int m = 0; m < X; ++m) std::fprintf(dp, "%d ", occ[m]);
                std::fprintf(dp, "%lld %.16e\n", (long long)a_lambda, Cn);
            } else {
                const std::uint64_t rk = k & ~OCCUPIED;
                std::fwrite(&rk, sizeof(std::uint64_t), 1, dp);
                std::fwrite(&a_lambda, sizeof(std::int64_t), 1, dp);
                std::fwrite(&Cn, sizeof(double), 1, dp);
            }
            ++written;
        }
        std::fclose(dp);
        std::fprintf(stderr, "wrote %lld states to %s (%.2f GB, %.1f s)\n",
                     (long long)written, dname,
                     (DUMP_TEXT ? 0.0 : 24.0 * (double)written / 1e9),
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - td0).count());
    }

    char fname[64];
    std::snprintf(fname, sizeof(fname), "S_L_N%d.dat", N);
    std::FILE* fp = std::fopen(fname, "w");
    if (!fp) { std::fprintf(stderr, "cannot open %s\n", fname); return 1; }

    std::fprintf(fp, "# nu=1/2 bosonic Laughlin static structure factor\n");
    std::fprintf(fp, "# N = %d   Q = %d   orbitals = %d\n", N, Q, X);
    std::fprintf(fp, "# basis dimension = %lld\n", (long long)dim);
    std::fprintf(fp, "# contraction %.3f s   S(L) %.3f s\n",
                 std::chrono::duration<double>(t1 - t0).count(),
                 std::chrono::duration<double>(t2 - t1).count());
    std::fprintf(fp, "#\n#%7s %24s\n", "L", "S(L)");

    for (int L = 0; L <= Lmax; ++L) {
        const long double S = 4.0L * (long double)M_PI / N * (expect[L] / norm2);
        std::fprintf(fp, "%8d %24.16e\n", L, (double)S);
        std::printf("S(%d) = %.10f\n", L, (double)S);
    }
    std::fclose(fp);
    std::fprintf(stderr, "results written to %s\n", fname);

    std::fprintf(stderr, "\n[N = %d] basis dimension = %lld\n", N, (long long)dim);
    std::fprintf(stderr, "[timing] contraction %.3f s, S(L) %.3f s\n",
                 std::chrono::duration<double>(t1 - t0).count(),
                 std::chrono::duration<double>(t2 - t1).count());

    delete[] g_keys;
    delete[] g_vals;
    return 0;
}
