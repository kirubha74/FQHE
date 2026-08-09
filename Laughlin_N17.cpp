
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
#include <functional>
#include <string>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int N = 14;
static constexpr int Q = N - 1;
static constexpr int X = 2 * N - 1;

static constexpr int PAIR_TASKS = 2;    // pairs fixed to form parallel tasks

// 0 = use every core the scheduler actually granted (recommended).
// Any other value forces exactly that many threads.
static constexpr int FORCE_THREADS = 0;

static constexpr int DUMP_STATES = 0;   // 1 = write states_N*.bin, 0 = skip
static constexpr int DUMP_TEXT   = 0;   // 0 = binary, 1 = plain text

static_assert(N >= 4, "N must be at least 4");
static_assert(PAIR_TASKS <= (N + 1) / 2 - 1, "PAIR_TASKS too large for this N");

static constexpr int MAXP = 2 * (N - 1);
static constexpr int TOT  = N * (N - 1);

static std::int64_t* g_cnt = nullptr;
static std::int64_t* g_cum = nullptr;
static std::atomic<std::int64_t>* g_tally = nullptr;
static int g_pre[N + 1];
static std::int64_t g_dim = 0;

static inline std::int64_t& CNT(int k, int prev, int s) {
    return g_cnt[(std::size_t)(k * (MAXP + 1) + prev) * (TOT + 1) + s];
}

// cumulative table: CUM(k,s,v) = sum over v' >= v of CNT(k+1,v',s+v'),
// omitting the v <= prev cap (which enters as a subtraction bound instead).
static inline std::int64_t& CUM(int k, int s, int v) {
    return g_cum[((std::size_t)k * (TOT + 1) + s) * (MAXP + 2) + v];
}

static inline bool feasible_nocap(int k, int s, int v) {
    const int rem = TOT - s;
    if (v > rem) return false;
    if (s + v > g_pre[k + 1]) return false;
    if ((std::int64_t)(rem - v) > (std::int64_t)v * (N - k - 1)) return false;
    return true;
}

static inline bool feasible(int k, int prev, int s, int v) {
    const int rem = TOT - s;
    if (v > prev || v > rem) return false;
    if (s + v > g_pre[k + 1]) return false;
    if ((std::int64_t)(rem - v) > (std::int64_t)v * (N - k - 1)) return false;
    return true;
}

static void build_counts() {
    g_pre[0] = 0;
    for (int i = 0; i < N; ++i) g_pre[i + 1] = g_pre[i] + 2 * (N - 1 - i);
    const std::size_t sz = (std::size_t)(N + 1) * (MAXP + 1) * (TOT + 1);
    g_cnt = new std::int64_t[sz]();
    for (int prev = 0; prev <= MAXP; ++prev) CNT(N, prev, TOT) = 1;
    for (int k = N - 1; k >= 0; --k)
        for (int prev = 0; prev <= MAXP; ++prev)
            for (int s = 0; s <= TOT; ++s) {
                std::int64_t r = 0;
                for (int v = std::min(prev, TOT - s); v >= 0; --v)
                    if (feasible(k, prev, s, v)) r += CNT(k + 1, v, s + v);
                CNT(k, prev, s) = r;
            }
    g_dim = CNT(0, MAXP, 0);

    const std::size_t csz = (std::size_t)N * (TOT + 1) * (MAXP + 2);
    g_cum = new std::int64_t[csz]();
    for (int k = 0; k < N; ++k)
        for (int s = 0; s <= TOT; ++s) {
            CUM(k, s, MAXP + 1) = 0;
            for (int v = MAXP; v >= 0; --v)
                CUM(k, s, v) = CUM(k, s, v + 1)
                             + (feasible_nocap(k, s, v) ? CNT(k + 1, v, s + v) : 0);
        }
}

static inline std::int64_t rank_of(const int* mu) {
    std::int64_t r = 0;
    int s = 0, prev = MAXP;
    for (int k = 0; k < N; ++k) {
        const int hi = std::min(prev, TOT - s);
        r += CUM(k, s, mu[k] + 1) - CUM(k, s, hi + 1);
        s += mu[k]; prev = mu[k];
    }
    return r;
}

static void unrank(std::int64_t r, int* mu) {
    int s = 0, prev = MAXP;
    for (int k = 0; k < N; ++k)
        for (int v = std::min(prev, TOT - s); v >= 0; --v) {
            if (!feasible(k, prev, s, v)) continue;
            const std::int64_t c = CNT(k + 1, v, s + v);
            if (r < c) { mu[k] = v; s += v; prev = v; break; }
            r -= c;
        }
}

static inline void occ_to_mu(const int* occ, int* mu) {
    int t = 0;
    for (int m = X - 1; m >= 0; --m)
        for (int c = 0; c < occ[m]; ++c) mu[t++] = m;
}

static inline std::uint64_t encode_occ(const int* occ) {
    std::uint64_t k = 0;
    int b = 0;
    for (int m = 0; m < X; ++m) {
        for (int t = 0; t < occ[m]; ++t) k |= (1ull << b++);
        ++b;
    }
    return k;
}

static inline void mu_to_occ(const int* mu, int* occ) {
    for (int m = 0; m < X; ++m) occ[m] = 0;
    for (int k = 0; k < N; ++k) ++occ[mu[k]];
}

static constexpr int NPAIRS = (N + 1) / 2;

static inline std::uint64_t bitrev64(std::uint64_t v) {
    v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
    v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
    v = ((v >> 8) & 0x00FF00FF00FF00FFull) | ((v & 0x00FF00FF00FF00FFull) << 8);
    v = ((v >> 16) & 0x0000FFFF0000FFFFull) | ((v & 0x0000FFFF0000FFFFull) << 16);
    return (v >> 32) | (v << 32);
}

struct Worker {
    int perm[N];
    int occ[X];
    int root[N];
    bool used[N];
    int placed[N];
    int nplaced;
    int inv;

    // ---- place / unplace for levels above the last ----
    inline void place(int pos, int v) {
        int acc = 0;
        for (int t = 0; t < nplaced; ++t) {
            const int q = placed[t];
            acc += ((q < pos) == (perm[q] > v));      // branchless
        }
        inv += acc;
        perm[pos] = v; used[v] = true; placed[nplaced++] = pos;
        ++occ[root[pos] + v];
    }
    inline void unplace(int pos, int v) {
        --occ[root[pos] + v];
        --nplaced; used[v] = false;
        int acc = 0;
        for (int t = 0; t < nplaced; ++t) {
            const int q = placed[t];
            acc += ((q < pos) == (perm[q] > v));
        }
        inv -= acc;
    }

    // ---- emit: canonical mirror key -> rank -> tally ----
    inline void emit(int cmp_state, int inv_total) {
        // partition (descending) straight from the occupation array
        int mu[N], mv[N];
        int t = 0;
        for (int m = X - 1; m >= 0; --m)
            for (int c = 0; c < occ[m]; ++c) mu[t++] = m;
        // mirror partition: reverse and complement
        for (int k = 0; k < N; ++k) mv[k] = (X - 1) - mu[N - 1 - k];
        // pick the lexicographically smaller as canonical
        const int* can = mu;
        for (int k = 0; k < N; ++k) {
            if (mu[k] != mv[k]) { can = (mu[k] < mv[k]) ? mu : mv; break; }
        }
        const std::int64_t sgn = (inv_total & 1) ? -1 : 1;
        const std::int64_t add = (cmp_state == 0) ? sgn : 2 * sgn;
        g_tally[rank_of(can)].fetch_add(add, std::memory_order_relaxed);
    }

    // ---- last level: O(1) sign via precomputed cost tables ----
    inline void last_level(int cmp_state) {
        const int i = NPAIRS - 1, j = N - NPAIRS;
        int costA[N], costB[N];
        for (int v = 0; v < N; ++v) {
            if (used[v]) continue;
            int a = 0, b2 = 0;
            for (int t = 0; t < nplaced; ++t) {
                const int q = placed[t];
                a  += ((q < i) == (perm[q] > v));
                b2 += ((q < j) == (perm[q] > v));
            }
            costA[v] = a; costB[v] = b2;
        }

        if (i == j) {                       // odd N: single middle position
            for (int v = 0; v < N; ++v) {
                if (used[v]) continue;
                int cs = cmp_state;
                if (cs == 0) {
                    const int w = N - 1 - v;
                    if (v < w) cs = -1; else if (v > w) cs = 1;
                }
                if (cs == 1) continue;
                ++occ[root[i] + v];
                emit(cs, inv + costA[v]);
                --occ[root[i] + v];
            }
            return;
        }

        for (int a = 0; a < N; ++a) {
            if (used[a]) continue;
            ++occ[root[i] + a];
            for (int b = 0; b < N; ++b) {
                if (b == a || used[b]) continue;
                int cs = cmp_state;
                if (cs == 0) {
                    const int ya = N - 1 - b, yb = N - 1 - a;
                    if (a < ya || (a == ya && b < yb)) cs = -1;
                    else if (a > ya || (a == ya && b > yb)) cs = 1;
                }
                if (cs == 1) continue;
                ++occ[root[j] + b];
                emit(cs, inv + costA[a] + costB[b] + (a > b ? 1 : 0));
                --occ[root[j] + b];
            }
            --occ[root[i] + a];
        }
    }

    void dfs(int k, int cmp_state) {
        if (k == NPAIRS - 1) { last_level(cmp_state); return; }
        const int i = k, j = N - 1 - k;
        for (int a = 0; a < N; ++a) {
            if (used[a]) continue;
            place(i, a);
            for (int b = 0; b < N; ++b) {
                if (used[b]) continue;
                int cs = cmp_state;
                if (cs == 0) {
                    const int ya = N - 1 - b, yb = N - 1 - a;
                    if (a < ya || (a == ya && b < yb)) cs = -1;
                    else if (a > ya || (a == ya && b > yb)) cs = 1;
                }
                if (cs == 1) continue;
                place(j, b);
                dfs(k + 1, cs);
                unplace(j, b);
            }
            unplace(i, a);
        }
    }
};

static constexpr int LFACT_MAX = 4 * N + 16;

static double lfact(int n) {
    static double c[LFACT_MAX];
    static bool init = false;
    if (!init) {
        c[0] = 0.0;
        for (int i = 1; i < LFACT_MAX; ++i) c[i] = c[i-1] + std::log((double)i);
        init = true;
    }
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

static void write_partial(const char* path) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", path); std::exit(1); }
    const std::uint64_t hdr[3] = { 0x4C4155504152543Aull, (std::uint64_t)N,
                                   (std::uint64_t)g_dim };
    std::fwrite(hdr, sizeof(std::uint64_t), 3, f);
    const std::size_t CH = 1u << 22;
    std::vector<std::int64_t> buf(CH);
    for (std::int64_t i = 0; i < g_dim; i += CH) {
        const std::size_t n = (std::size_t)std::min<std::int64_t>(CH, g_dim - i);
        for (std::size_t j = 0; j < n; ++j)
            buf[j] = g_tally[i + j].load(std::memory_order_relaxed);
        std::fwrite(buf.data(), sizeof(std::int64_t), n, f);
    }
    std::fclose(f);
}

static void add_partial(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(1); }
    std::uint64_t hdr[3];
    if (std::fread(hdr, sizeof(std::uint64_t), 3, f) != 3 ||
        hdr[0] != 0x4C4155504152543Aull || (int)hdr[1] != N ||
        (std::int64_t)hdr[2] != g_dim) {
        std::fprintf(stderr, "%s: not a matching partial file\n", path);
        std::exit(1);
    }
    const std::size_t CH = 1u << 22;
    std::vector<std::int64_t> buf(CH);
    for (std::int64_t i = 0; i < g_dim; i += CH) {
        const std::size_t n = (std::size_t)std::min<std::int64_t>(CH, g_dim - i);
        if (std::fread(buf.data(), sizeof(std::int64_t), n, f) != n) {
            std::fprintf(stderr, "%s: truncated\n", path); std::exit(1);
        }
        for (std::size_t j = 0; j < n; ++j)
            if (buf[j]) g_tally[i + j].fetch_add(buf[j], std::memory_order_relaxed);
    }
    std::fclose(f);
    std::fprintf(stderr, "merged %s\n", path);
}

int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();

#ifdef _OPENMP
    const int ncores = omp_get_num_procs();
    if (FORCE_THREADS > 0) omp_set_num_threads(FORCE_THREADS);
    else                   omp_set_num_threads(ncores);
    const int nthr = omp_get_max_threads();
    std::printf("cores visible = %d, threads = %d\n", ncores, nthr);
    if (nthr > ncores)
        std::fprintf(stderr,
            "WARNING: %d threads on %d cores -- oversubscribed, this will be SLOWER.\n",
            nthr, ncores);
    if (nthr < 8)
        std::fprintf(stderr,
            "WARNING: only %d thread(s). Check --cpus-per-task and that the binary\n"
            "         was compiled with -fopenmp.\n", nthr);
#else
    std::printf("threads = 1 (compiled WITHOUT -fopenmp)\n");
    std::fprintf(stderr, "WARNING: no OpenMP. Recompile with -fopenmp for parallelism.\n");
#endif
    build_counts();
    std::printf("N = %d, orbitals = %d, dimension = %lld (%.1f GB)\n",
                N, X, (long long)g_dim, 8.0 * (double)g_dim / 1e9);
    std::fflush(stdout);

    g_tally = new (std::nothrow) std::atomic<std::int64_t>[g_dim];
    if (!g_tally) { std::fprintf(stderr, "allocation failed\n"); return 1; }

#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < g_dim; ++i) g_tally[i].store(0, std::memory_order_relaxed);

    // ------------------------------------------------------------------
    // Build the task list: all surviving choices for the first
    // PAIR_TASKS pairs. Generated serially (cheap), then run in parallel.
    // ------------------------------------------------------------------
    struct Task { int val[2 * PAIR_TASKS]; int cs; };
    std::vector<Task> tasks;
    {
        Task cur{};
        bool used[N] = {};
        std::function<void(int,int)> gen = [&](int k, int cs) {
            if (k == PAIR_TASKS) { cur.cs = cs; tasks.push_back(cur); return; }
            for (int a = 0; a < N; ++a) {
                if (used[a]) continue;
                used[a] = true;
                for (int b = 0; b < N; ++b) {
                    if (used[b]) continue;
                    int c2 = cs;
                    if (c2 == 0) {
                        const int ya = N - 1 - b, yb = N - 1 - a;
                        if (a < ya || (a == ya && b < yb)) c2 = -1;
                        else if (a > ya || (a == ya && b > yb)) c2 = 1;
                    }
                    if (c2 == 1) continue;
                    used[b] = true;
                    cur.val[2 * k] = a; cur.val[2 * k + 1] = b;
                    gen(k + 1, c2);
                    used[b] = false;
                }
                used[a] = false;
            }
        };
        gen(0, 0);
    }
    const std::int64_t ntasks = (std::int64_t)tasks.size();
    std::printf("tasks = %lld\n", (long long)ntasks);
    std::fflush(stdout);

    // ---- work-splitting arguments ----
    std::int64_t part_begin = 0, part_end = ntasks;
    const char* part_out = nullptr;
    bool merge_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--range" && i + 2 < argc) {
            part_begin = std::atoll(argv[i + 1]);
            part_end   = std::atoll(argv[i + 2]);
            i += 2;
        } else if (std::string(argv[i]) == "--out" && i + 1 < argc) {
            part_out = argv[++i];
        } else if (std::string(argv[i]) == "--merge") {
            merge_mode = true;
        } else if (merge_mode) {
            add_partial(argv[i]);
        }
    }
    if (part_begin < 0) part_begin = 0;
    if (part_end > ntasks) part_end = ntasks;
    if (!merge_mode && (part_begin != 0 || part_end != ntasks))
        std::printf("running task range [%lld, %lld) of %lld\n",
                    (long long)part_begin, (long long)part_end, (long long)ntasks);

    std::atomic<std::int64_t> done{0};

#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t t = part_begin; t < (merge_mode ? part_begin : part_end); ++t) {
        Worker w;
        for (int i2 = 0; i2 < N; ++i2) w.root[i2] = N - 1 - i2;
        for (int i2 = 0; i2 < N; ++i2) w.used[i2] = false;
        for (int m = 0; m < X; ++m) w.occ[m] = 0;
        w.nplaced = 0; w.inv = 0;

        for (int k = 0; k < PAIR_TASKS; ++k) {
            w.place(k, tasks[t].val[2 * k]);
            w.place(N - 1 - k, tasks[t].val[2 * k + 1]);
        }
        if (PAIR_TASKS == NPAIRS - 1) w.last_level(tasks[t].cs);
        else w.dfs(PAIR_TASKS, tasks[t].cs);

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

    if (part_out) {
        write_partial(part_out);
        std::fprintf(stderr, "partial written to %s (contraction %.1f s)\n",
                     part_out, std::chrono::duration<double>(t1 - t0).count());
        delete[] g_tally; delete[] g_cnt; delete[] g_cum;
        return 0;
    }


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

#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < g_dim; ++i) {
            const std::int64_t raw = g_tally[i].load(std::memory_order_relaxed);
            if (raw == 0) continue;
            int mu[N], occA[X], occB[X];
            unrank(i, mu);
            mu_to_occ(mu, occA);
            for (int m = 0; m < X; ++m) occB[m] = occA[X - 1 - m];

            bool self = true;
            for (int m = 0; m < X; ++m) if (occA[m] != occB[m]) { self = false; break; }
            const std::int64_t c = self ? raw : raw / 2;
            const int copies = self ? 1 : 2;

            for (int rep = 0; rep < copies; ++rep) {
                const int* occ = (rep == 0) ? occA : occB;
                ++loc_dim;
                double la = 0.0, lw = 0.5 * lfact(N);
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
            std::fprintf(dp, "# columns: n_0 ... n_%d  a_lambda  C_normalised\n", X - 1);
        } else {
            const std::uint64_t hdr[4] = { 0x4C415547484C4E31ull,
                                           (std::uint64_t)N,
                                           (std::uint64_t)X,
                                           (std::uint64_t)dim };
            std::fwrite(hdr, sizeof(std::uint64_t), 4, dp);
        }

        std::int64_t written = 0;
        for (std::int64_t i = 0; i < g_dim; ++i) {
            const std::int64_t raw = g_tally[i].load(std::memory_order_relaxed);
            if (raw == 0) continue;

            int mu[N], occA[X], occB[X];
            unrank(i, mu);
            mu_to_occ(mu, occA);
            for (int m = 0; m < X; ++m) occB[m] = occA[X - 1 - m];

            bool self = true;
            for (int m = 0; m < X; ++m) if (occA[m] != occB[m]) { self = false; break; }
            const std::int64_t c = self ? raw : raw / 2;
            const int copies = self ? 1 : 2;

            for (int rep = 0; rep < copies; ++rep) {
                const int* occ = (rep == 0) ? occA : occB;
                double la = 0.0, lw = 0.5 * lfact(N);
                std::int64_t mult = 1;
                for (int m = 0; m < X; ++m) {
                    if (!occ[m]) continue;
                    la += lfact(occ[m]);
                    lw -= 0.5 * lfact(occ[m]);
                    lw -= 0.5 * occ[m] * lbin[m];
                    for (int t = 2; t <= occ[m]; ++t) mult *= t;
                }
                const std::int64_t a_lambda = c * mult;
                const double Cn = (double)c * std::exp(la + lw - lref);

                if (DUMP_TEXT) {
                    for (int m = 0; m < X; ++m) std::fprintf(dp, "%d ", occ[m]);
                    std::fprintf(dp, "%lld %.16e\n", (long long)a_lambda, Cn);
                } else {
                    const std::uint64_t key = encode_occ(occ);
                    std::fwrite(&key, sizeof(std::uint64_t), 1, dp);
                    std::fwrite(&a_lambda, sizeof(std::int64_t), 1, dp);
                    std::fwrite(&Cn, sizeof(double), 1, dp);
                }
                ++written;
            }
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

    delete[] g_tally;
    delete[] g_cnt;
    delete[] g_cum;
    return 0;
}
