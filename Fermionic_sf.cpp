#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif
#include <chrono>
static inline int omp_get_max_threads() { return 1; }
static inline double omp_get_wtime() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
#endif

using u64 = uint64_t;
using i64 = int64_t;
using i128 = __int128;

static constexpr int N_PART = 9;
static constexpr double PROGRESS_SEC = 10.0;

static constexpr bool SAVE_STATES_BIN = false;
static constexpr bool SAVE_STATES_TXT = true;
static constexpr bool SAVE_SBAR = true;

static constexpr int TWO_Q = 3 * (N_PART - 1);
static constexpr int MAXORB = TWO_Q;
static constexpr int NBITS = 3 * N_PART - 3;

static std::vector<double> LOGF;

static void init_logf(int n) {
    LOGF.assign(n + 1, 0.0);
    for (int i = 1; i <= n; ++i) LOGF[i] = LOGF[i - 1] + std::log((double)i);
}

static bool tri_ok(int ta, int tb, int tc) {
    if (((ta + tb + tc) & 1) != 0) return false;
    if (tc < std::abs(ta - tb)) return false;
    if (tc > ta + tb) return false;
    return true;
}

static double wigner3j(int tj1, int tj2, int tj3, int tm1, int tm2, int tm3) {
    if (tm1 + tm2 + tm3 != 0) return 0.0;
    if (!tri_ok(tj1, tj2, tj3)) return 0.0;
    if (std::abs(tm1) > tj1 || std::abs(tm2) > tj2 || std::abs(tm3) > tj3) return 0.0;
    if (((tj1 + tm1) & 1) || ((tj2 + tm2) & 1) || ((tj3 + tm3) & 1)) return 0.0;

    int a = (tj1 + tj2 - tj3) / 2;
    int b = (tj1 - tj2 + tj3) / 2;
    int c = (-tj1 + tj2 + tj3) / 2;
    int d = (tj1 + tj2 + tj3) / 2 + 1;

    int p1 = (tj1 + tm1) / 2, p2 = (tj1 - tm1) / 2;
    int p3 = (tj2 + tm2) / 2, p4 = (tj2 - tm2) / 2;
    int p5 = (tj3 + tm3) / 2, p6 = (tj3 - tm3) / 2;

    double logdelta = LOGF[a] + LOGF[b] + LOGF[c] - LOGF[d];
    double lognum = LOGF[p1] + LOGF[p2] + LOGF[p3] + LOGF[p4] + LOGF[p5] + LOGF[p6];

    int x = (tj1 - tm1) / 2;
    int y = (tj2 + tm2) / 2;
    int z = (tj3 - tj2 + tm1) / 2;
    int w = (tj3 - tj1 - tm2) / 2;

    int kmin = 0;
    if (-z > kmin) kmin = -z;
    if (-w > kmin) kmin = -w;
    int kmax = a;
    if (x < kmax) kmax = x;
    if (y < kmax) kmax = y;
    if (kmin > kmax) return 0.0;

    std::vector<double> lt(kmax - kmin + 1);
    double lmax = -1e300;
    for (int k = kmin; k <= kmax; ++k) {
        double t = -(LOGF[k] + LOGF[a - k] + LOGF[x - k] + LOGF[y - k] + LOGF[z + k] + LOGF[w + k]);
        lt[k - kmin] = t;
        if (t > lmax) lmax = t;
    }
    long double s = 0.0L;
    for (int k = kmin; k <= kmax; ++k) {
        long double term = std::exp((long double)(lt[k - kmin] - lmax));
        s += ((k & 1) ? -term : term);
    }
    if (s == 0.0L) return 0.0;

    int ph = (tj1 - tj2 - tm3) / 2;
    long double sign = (ph & 1) ? -1.0L : 1.0L;
    long double mag = std::exp((long double)(0.5 * (logdelta + lognum) + lmax));
    return (double)(sign * mag * s);
}

static void fmt_hms(double s, char* out, size_t n) {
    if (!(s >= 0.0) || s > 3.15e10) { std::snprintf(out, n, "--:--:--"); return; }
    long t = (long)(s + 0.5);
    long h = t / 3600;
    if (h > 999999) { std::snprintf(out, n, "--:--:--"); return; }
    long m = (t / 60) % 60, sec = t % 60;
    std::snprintf(out, n, "%02d:%02d:%02d", (int)h, (int)m, (int)sec);
}

static inline u64 mix64(u64 k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33; return k;
}

struct TableI64 {
    size_t cap = 0, mask = 0;
    std::atomic<u64>* keys = nullptr;
    std::atomic<i64>* vals = nullptr;

    void init(int log2cap) {
        cap = (size_t)1 << log2cap;
        mask = cap - 1;
        keys = new std::atomic<u64>[cap];
        vals = new std::atomic<i64>[cap];
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cap; ++i) {
            keys[i].store(0, std::memory_order_relaxed);
            vals[i].store(0, std::memory_order_relaxed);
        }
    }
    void destroy() { delete[] keys; delete[] vals; keys = nullptr; vals = nullptr; }

    inline void add(u64 k, i64 v) {
        size_t i = mix64(k) & mask;
        for (;;) {
            u64 cur = keys[i].load(std::memory_order_acquire);
            if (cur == k) { vals[i].fetch_add(v, std::memory_order_relaxed); return; }
            if (cur == 0) {
                u64 exp = 0;
                if (keys[i].compare_exchange_strong(exp, k, std::memory_order_acq_rel) || exp == k) {
                    vals[i].fetch_add(v, std::memory_order_relaxed); return;
                }
            }
            i = (i + 1) & mask;
        }
    }
};

struct TableI128 {
    size_t cap = 0, mask = 0;
    std::atomic<u64>* keys = nullptr;
    i128* vals = nullptr;
    std::atomic<unsigned char>* lk = nullptr;

    void init(int log2cap) {
        cap = (size_t)1 << log2cap;
        mask = cap - 1;
        keys = new std::atomic<u64>[cap];
        vals = new i128[cap];
        lk = new std::atomic<unsigned char>[cap];
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < cap; ++i) {
            keys[i].store(0, std::memory_order_relaxed);
            vals[i] = 0;
            lk[i].store(0, std::memory_order_relaxed);
        }
    }
    void destroy() { delete[] keys; delete[] vals; delete[] lk; keys = nullptr; vals = nullptr; lk = nullptr; }

    inline void add(u64 k, i128 v) {
        size_t i = mix64(k) & mask;
        for (;;) {
            u64 cur = keys[i].load(std::memory_order_acquire);
            if (cur == 0) {
                u64 exp = 0;
                if (!keys[i].compare_exchange_strong(exp, k, std::memory_order_acq_rel) && exp != k) {
                    i = (i + 1) & mask; continue;
                }
                cur = k;
            }
            if (cur == k) {
                unsigned char e = 0;
                while (!lk[i].compare_exchange_weak(e, 1, std::memory_order_acquire)) e = 0;
                vals[i] += v;
                lk[i].store(0, std::memory_order_release);
                return;
            }
            i = (i + 1) & mask;
        }
    }
};

static TableI64 T1;
static TableI128 T2;

static inline u64 key_from_counts(const int* cnt, int hi) {
    u64 key = 0;
    int t = 0;
    for (int v = 0; v <= hi; ++v) {
        int c = cnt[v];
        if (c) { key |= (((u64)1 << c) - 1) << (v + t); t += c; }
    }
    return key;
}

static inline u64 mirror_key(u64 k) {
    u64 r = 0;
    while (k) { int b = __builtin_ctzll(k); r |= (u64)1 << (MAXORB - b); k &= k - 1; }
    return r;
}

static void step1_dfs(int j, u64 used, int inv, int* cnt) {
    constexpr int N = N_PART;
    if (j == N) {
        T1.add(key_from_counts(cnt, 2 * N - 2), (inv & 1) ? -1 : 1);
        return;
    }
    int lj = N - 1 - j;
    for (int r = 0; r < N; ++r) {
        u64 bit = (u64)1 << r;
        if (used & bit) continue;
        int add = __builtin_popcountll(used >> (r + 1));
        int d = lj + (N - 1 - r);
        cnt[d]++;
        step1_dfs(j + 1, used | bit, inv + add, cnt);
        cnt[d]--;
    }
}

struct Prefix { u64 used; int inv; std::vector<int> dvals; };

static void gen_prefixes(int j, int depth, u64 used, int inv, std::vector<int>& dv, std::vector<Prefix>& out) {
    constexpr int N = N_PART;
    if (j == depth) { out.push_back(Prefix{used, inv, dv}); return; }
    int lj = N - 1 - j;
    for (int r = 0; r < N; ++r) {
        u64 bit = (u64)1 << r;
        if (used & bit) continue;
        int add = __builtin_popcountll(used >> (r + 1));
        dv.push_back(lj + (N - 1 - r));
        gen_prefixes(j + 1, depth, used | bit, inv + add, dv, out);
        dv.pop_back();
    }
}

static void step2_dfs(int j, u64 used, int inv, const int* dval, int* rc, int K, i128 coeff, bool dual) {
    constexpr int N = N_PART;
    if (j == N) {
        i128 c = (inv & 1) ? -coeff : coeff;
        T2.add(used, c);
        if (dual) T2.add(mirror_key(used), c);
        return;
    }
    int lj = N - 1 - j;
    for (int t = 0; t < K; ++t) {
        if (rc[t] == 0) continue;
        int v = dval[t] + lj;
        u64 bit = (u64)1 << v;
        if (used & bit) continue;
        int add = __builtin_popcountll(used & (bit - 1));
        rc[t]--;
        step2_dfs(j + 1, used | bit, inv + add, dval, rc, K, coeff, dual);
        rc[t]++;
    }
}

static void decode_bkey(u64 key, int* part) {
    constexpr int N = N_PART;
    int nu[64], c = 0;
    u64 k = key;
    while (k) { nu[c++] = __builtin_ctzll(k); k &= k - 1; }
    for (int j = 0; j < N; ++j) part[j] = nu[N - 1 - j] - (N - 1 - j);
}

int main(int argc, char** argv) {
    static_assert(N_PART >= 2 && N_PART <= 21, "N_PART out of supported range");
    static_assert(NBITS < 64, "key does not fit in 64 bits");
    constexpr int N = N_PART;

    int lc1 = (argc > 1) ? std::atoi(argv[1]) : 0;
    int lc2 = (argc > 2) ? std::atoi(argv[2]) : 0;
    if (lc1 == 0) { lc1 = 16; while (lc1 < 32 && ((size_t)1 << lc1) < (size_t)1 << std::min(31, 2 * N)) ++lc1; }
    if (lc2 == 0) lc2 = std::min(33, lc1 + 2);

    init_logf(8 * N + 64);

    double tstart = omp_get_wtime();

    T1.init(lc1);
    {
        int depth = 1;
        long long tasks = N;
        int nt = omp_get_max_threads();
        long long want = (long long)nt * 64;
        while (depth < N - 1 && tasks < want) { ++depth; tasks *= (N - depth + 1); }
        std::vector<Prefix> pre;
        std::vector<int> dv;
        gen_prefixes(0, depth, 0, 0, dv, pre);
        const size_t ntask = pre.size();
        std::atomic<size_t> done{0};
        std::atomic<double> lastp{omp_get_wtime()};
#pragma omp parallel
        {
            std::vector<int> cnt(2 * N - 1, 0);
#pragma omp for schedule(dynamic, 1)
            for (size_t t = 0; t < ntask; ++t) {
                std::fill(cnt.begin(), cnt.end(), 0);
                for (int d : pre[t].dvals) cnt[d]++;
                step1_dfs(depth, pre[t].used, pre[t].inv, cnt.data());
                size_t p = ++done;
                double now = omp_get_wtime();
                if (now - lastp.load(std::memory_order_relaxed) > PROGRESS_SEC) {
#pragma omp critical (progress)
                    {
                        if (now - lastp.load(std::memory_order_relaxed) > PROGRESS_SEC) {
                            lastp.store(now, std::memory_order_relaxed);
                            double el = now - tstart;
                            double frac = (double)p / (double)ntask;
                            char e[24], r[24];
                            fmt_hms(el, e, sizeof(e));
                            fmt_hms(frac > 0 ? el * (1.0 - frac) / frac : -1.0, r, sizeof(r));
                            std::printf("%6.2f%%   elapsed %s   remaining ~%s\n", 100.0 * frac, e, r);
                            std::fflush(stdout);
                        }
                    }
                }
            }
        }
    }

    std::vector<u64> bkeys;
    std::vector<i64> braw;
    for (size_t i = 0; i < T1.cap; ++i) {
        u64 k = T1.keys[i].load(std::memory_order_relaxed);
        if (!k) continue;
        i64 v = T1.vals[i].load(std::memory_order_relaxed);
        if (!v) continue;
        bkeys.push_back(k);
        braw.push_back(v);
    }
    T1.destroy();
    size_t NB = bkeys.size();
    std::vector<std::vector<int>> dval(NB), dcnt(NB);
    std::vector<i128> bcoef(NB);
    std::vector<unsigned char> mode(NB, 0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < NB; ++i) {
        std::vector<int> part(N);
        decode_bkey(bkeys[i], part.data());
        std::vector<int> dv, dc;
        for (int a = 0; a < N; ++a) {
            if (!dv.empty() && dv.back() == part[a]) dc.back()++;
            else { dv.push_back(part[a]); dc.push_back(1); }
        }
        i128 f = 1;
        for (int c : dc) for (int k = 2; k <= c; ++k) f *= k;
        dval[i] = dv;
        dcnt[i] = dc;
        bcoef[i] = (i128)braw[i] * f;

        std::vector<int> cnt(2 * N - 1, 0);
        for (int a = 0; a < N; ++a) cnt[(2 * N - 2) - part[a]]++;
        u64 mk = key_from_counts(cnt.data(), 2 * N - 2);
        mode[i] = (mk == bkeys[i]) ? 1 : (bkeys[i] < mk ? 2 : 0);
    }

    std::vector<size_t> work;
    work.reserve(NB);
    for (size_t i = 0; i < NB; ++i) if (mode[i]) work.push_back(i);
    std::sort(work.begin(), work.end(), [&](size_t a, size_t b) {
        return dval[a].size() > dval[b].size();
    });

    std::vector<double> wt(work.size());
    double wtot = 0.0;
    for (size_t w = 0; w < work.size(); ++w) {
        double x = 1.0;
        for (int a = 2; a <= N; ++a) x *= (double)a;
        for (int c : dcnt[work[w]]) for (int a = 2; a <= c; ++a) x /= (double)a;
        wt[w] = x;
        wtot += x;
    }
    double wdone = 0.0;
    std::atomic<double> lastp2{omp_get_wtime()};

    T2.init(lc2);

#pragma omp parallel
    {
        std::vector<int> rc;
#pragma omp for schedule(dynamic, 8)
        for (size_t w = 0; w < work.size(); ++w) {
            size_t i = work[w];
            rc.assign(dcnt[i].begin(), dcnt[i].end());
            step2_dfs(0, 0, 0, dval[i].data(), rc.data(), (int)dval[i].size(), bcoef[i], mode[i] == 2);
#pragma omp atomic
            wdone += wt[w];
            double now = omp_get_wtime();
            if (now - lastp2.load(std::memory_order_relaxed) > PROGRESS_SEC) {
#pragma omp critical (progress)
                {
                    if (now - lastp2.load(std::memory_order_relaxed) > PROGRESS_SEC) {
                        lastp2.store(now, std::memory_order_relaxed);
                        double snap;
#pragma omp atomic read
                        snap = wdone;
                        double el = now - tstart;
                        double frac = snap / wtot;
                        char e[24], r[24];
                        fmt_hms(el, e, sizeof(e));
                        fmt_hms(frac > 1e-9 ? el * (1.0 - frac) / frac : -1.0, r, sizeof(r));
                        std::printf("%6.2f%%   elapsed %s   remaining ~%s\n", 100.0 * frac, e, r);
                        std::fflush(stdout);
                    }
                }
            }
        }
    }

    std::vector<u64> fkeys;
    std::vector<i128> fcoef;
    for (size_t i = 0; i < T2.cap; ++i) {
        u64 k = T2.keys[i].load(std::memory_order_relaxed);
        if (!k) continue;
        i128 v = T2.vals[i];
        if (v == 0) continue;
        fkeys.push_back(k);
        fcoef.push_back(v);
    }
    T2.destroy();
    size_t D = fkeys.size();

    std::vector<size_t> ord(D);
    for (size_t i = 0; i < D; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return fkeys[a] > fkeys[b]; });

    std::vector<double> logbin(TWO_Q + 1);
    for (int j = 0; j <= TWO_Q; ++j) logbin[j] = LOGF[TWO_Q] - LOGF[j] - LOGF[TWO_Q - j];

    std::vector<double> logmag(D);
    std::vector<int> sg(D);
    double lognf = LOGF[N];

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < D; ++i) {
        i128 c = fcoef[i];
        int s = 1;
        if (c < 0) { s = -1; c = -c; }
        double lc = (double)std::log((long double)c);
        double acc = 0.0;
        u64 k = fkeys[i];
        while (k) { int b = __builtin_ctzll(k); acc += logbin[b]; k &= k - 1; }
        logmag[i] = lc + 0.5 * lognf - 0.5 * acc;
        sg[i] = s;
    }

    double lmax = -1e300;
    for (size_t i = 0; i < D; ++i) if (logmag[i] > lmax) lmax = logmag[i];

    std::vector<double> amp(D);
    long double nrm2 = 0.0L;
#pragma omp parallel for schedule(static) reduction(+ : nrm2)
    for (size_t i = 0; i < D; ++i) {
        double a = sg[i] * std::exp(logmag[i] - lmax);
        amp[i] = a;
        nrm2 += (long double)a * a;
    }
    double invn = 1.0 / std::sqrt((double)nrm2);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < D; ++i) amp[i] *= invn;

    const int tQ = TWO_Q;
    std::vector<double> sbar(TWO_Q + 1, 0.0);
    for (int L = 0; L <= TWO_Q; ++L) {
        double w2 = wigner3j(tQ, tQ, 2 * L, -tQ, tQ, 0);
        std::vector<double> rho(TWO_Q + 1, 0.0);
        if (w2 != 0.0) {
            double pref = std::sqrt((double)(tQ + 1) * (double)(tQ + 1) * (double)(2 * L + 1) / (4.0 * M_PI));
            for (int j = 0; j <= TWO_Q; ++j) {
                int tm = 2 * j - tQ;
                double w1 = wigner3j(tQ, tQ, 2 * L, -tm, tm, 0);
                if (w1 == 0.0) continue;
                rho[j] = (((tQ + L + j) & 1) ? -1.0 : 1.0) * pref * w1 * w2;
            }
        }
        long double acc = 0.0L;
#pragma omp parallel for schedule(static) reduction(+ : acc)
        for (size_t i = 0; i < D; ++i) {
            u64 k = fkeys[i];
            double d = 0.0;
            while (k) { int b = __builtin_ctzll(k); d += rho[b]; k &= k - 1; }
            acc += (long double)amp[i] * amp[i] * (long double)d * d;
        }
        sbar[L] = (4.0 * M_PI / N) * (double)acc;
    }

    char fn[256];

    if (SAVE_STATES_BIN) {
    std::snprintf(fn, sizeof(fn), "nu13_N%d_states.bin", N);
    {
        FILE* f = std::fopen(fn, "wb");
        i64 hdr[4] = {(i64)N, (i64)TWO_Q, (i64)D, (i64)NBITS};
        std::fwrite(hdr, sizeof(i64), 4, f);
        std::vector<char> buf;
        buf.reserve(1u << 22);
        for (size_t t = 0; t < D; ++t) {
            size_t i = ord[t];
            u64 k = fkeys[i];
            i128 c = fcoef[i];
            i64 lo = (i64)(u64)c, hi = (i64)(c >> 64);
            double a = amp[i];
            const char* p;
            p = (const char*)&k;  buf.insert(buf.end(), p, p + 8);
            p = (const char*)&lo; buf.insert(buf.end(), p, p + 8);
            p = (const char*)&hi; buf.insert(buf.end(), p, p + 8);
            p = (const char*)&a;  buf.insert(buf.end(), p, p + 8);
            if (buf.size() >= (1u << 22)) { std::fwrite(buf.data(), 1, buf.size(), f); buf.clear(); }
        }
        if (!buf.empty()) std::fwrite(buf.data(), 1, buf.size(), f);
        std::fclose(f);
    }

    }

    if (SAVE_STATES_TXT) {
    std::snprintf(fn, sizeof(fn), "nu13_N%d_states.txt", N);
    {
        FILE* f = std::fopen(fn, "w");
        std::setvbuf(f, nullptr, _IOFBF, 1 << 22);
        std::fprintf(f, "# N=%d 2Q=%d D=%zu\n", N, TWO_Q, D);
        std::fprintf(f, "# occupation partition coefficient amplitude\n");
        std::vector<char> occ(TWO_Q + 2, '0');
        occ[TWO_Q + 1] = 0;
        int part[64];
        char buf[64];
        for (size_t t = 0; t < D; ++t) {
            size_t i = ord[t];
            for (int a = 0; a <= TWO_Q; ++a) occ[a] = '0';
            u64 k = fkeys[i];
            int np = 0;
            while (k) { int b = __builtin_ctzll(k); occ[b] = '1'; part[np++] = b; k &= k - 1; }
            std::fputs(occ.data(), f);
            std::fputc(' ', f);
            std::fputc('[', f);
            for (int a = np - 1; a >= 0; --a) std::fprintf(f, "%d%s", part[a], a ? "," : "");
            std::fputs("] ", f);
            i128 c = fcoef[i];
            int s = 1;
            if (c < 0) { s = -1; c = -c; }
            int p = 63;
            buf[p] = 0;
            if (c == 0) buf[--p] = '0';
            while (c > 0) { buf[--p] = (char)('0' + (int)(c % 10)); c /= 10; }
            std::fprintf(f, "%s%s %.17e\n", s < 0 ? "-" : "", buf + p, amp[i]);
        }
        std::fclose(f);
    }

    }

    if (SAVE_SBAR) {
    std::snprintf(fn, sizeof(fn), "nu13_N%d_sbar.txt", N);
    {
        FILE* f = std::fopen(fn, "w");
        std::fprintf(f, "# nu=1/3 Laughlin  N=%d  2Q=%d  D=%zu\n", N, TWO_Q, D);
        std::fprintf(f, "# L  Sbar(L)\n");
        for (int L = 0; L <= TWO_Q; ++L) std::fprintf(f, "%d %.17e\n", L, sbar[L]);
        std::fclose(f);
    }

    }

    std::printf("\n  L        Sbar(L)\n");
    for (int L = 0; L <= TWO_Q; ++L) std::printf("%3d   %.12f\n", L, sbar[L]);
    std::printf("\n");

    {
        char e[24];
        fmt_hms(omp_get_wtime() - tstart, e, sizeof(e));
        std::printf("[done] N=%d  D=%zu  threads=%d  total time %s\n", N, D, omp_get_max_threads(), e);
    }
    return 0;
}
