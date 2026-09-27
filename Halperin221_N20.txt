#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

constexpr int K = 10;
constexpr int MM = 2;
constexpr int NN = 1;

constexpr int N = 2 * K;
constexpr int Q2 = MM * (K - 1) + NN * K;
constexpr int NORB = Q2 + 1;
constexpr int NL = Q2 + 1;
constexpr int LEXTRA = 2;
constexpr int SMAX = NN * K * K;
constexpr int SHALF = SMAX / 2;

constexpr int bits_for(int x) { int b = 1; while ((1 << b) <= x) ++b; return b; }
constexpr int PBITS = bits_for(Q2);
static_assert(K * PBITS < 64, "packed partition must leave the top bit free");
static_assert(MM % 2 == 0, "MM must be even");

using Part = std::array<uint8_t, K>;
using Key = uint64_t;
using Term = std::pair<Key, long long>;
using Vec = std::vector<Term>;

static long double FACT[256];
static long long FACTL[K + 1];

static void init_fact() {
    FACT[0] = 1.0L;
    for (int i = 1; i < 256; ++i) FACT[i] = FACT[i - 1] * (long double)i;
    FACTL[0] = 1;
    for (int i = 1; i <= K; ++i) FACTL[i] = FACTL[i - 1] * i;
}

static inline Key pack(const Part &p) {
    Key w = 0;
    for (int i = 0; i < K; ++i) w |= (Key)p[i] << (i * PBITS);
    return w;
}

static inline Part unpack(Key w) {
    Part p;
    for (int i = 0; i < K; ++i) p[i] = (uint8_t)((w >> (i * PBITS)) & ((Key(1) << PBITS) - 1));
    return p;
}

static inline void sort_desc(Part &p) {
    for (int i = 1; i < K; ++i) {
        uint8_t x = p[i];
        int j = i - 1;
        while (j >= 0 && p[j] < x) { p[j + 1] = p[j]; --j; }
        p[j + 1] = x;
    }
}

static inline long long hash_mult(const Part &p) {
    long long f = 1;
    int i = 0;
    while (i < K) {
        int j = i + 1;
        while (j < K && p[j] == p[i]) ++j;
        f *= FACTL[j - i];
        i = j;
    }
    return f;
}

struct FlatMap {
    static constexpr Key EMPTY = ~Key(0);
    std::vector<Key> k;
    std::vector<long long> v;
    size_t mask = 0, n = 0;

    explicit FlatMap(size_t hint = 16) {
        size_t c = 16;
        while (c < 2 * hint) c <<= 1;
        k.assign(c, EMPTY);
        v.assign(c, 0);
        mask = c - 1;
    }
    static inline size_t hsh(Key x) {
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return (size_t)x;
    }
    void grow() {
        std::vector<Key> ok;
        std::vector<long long> ov;
        ok.swap(k);
        ov.swap(v);
        size_t c = (mask + 1) * 2;
        k.assign(c, EMPTY);
        v.assign(c, 0);
        mask = c - 1;
        n = 0;
        for (size_t i = 0; i < ok.size(); ++i)
            if (ok[i] != EMPTY) add(ok[i], ov[i]);
    }
    inline void add(Key key, long long val) {
        if (2 * (n + 1) > mask + 1) grow();
        size_t i = hsh(key) & mask;
        for (;;) {
            if (k[i] == key) { v[i] += val; return; }
            if (k[i] == EMPTY) { k[i] = key; v[i] = val; ++n; return; }
            i = (i + 1) & mask;
        }
    }
    inline bool insert(Key key, long long val) {
        if (2 * (n + 1) > mask + 1) grow();
        size_t i = hsh(key) & mask;
        for (;;) {
            if (k[i] == key) return false;
            if (k[i] == EMPTY) { k[i] = key; v[i] = val; ++n; return true; }
            i = (i + 1) & mask;
        }
    }
    inline long long find(Key key) const {
        size_t i = hsh(key) & mask;
        for (;;) {
            if (k[i] == key) return v[i];
            if (k[i] == EMPTY) return -1;
            i = (i + 1) & mask;
        }
    }
    Vec to_vec() const {
        Vec out;
        out.reserve(n);
        for (size_t i = 0; i <= mask; ++i)
            if (k[i] != EMPTY && v[i] != 0) out.emplace_back(k[i], v[i]);
        return out;
    }
};

struct RadixAgg {
    static constexpr int RB = 10;
    static constexpr size_t NP = size_t(1) << RB;
    static constexpr size_t THRESH = size_t(1) << 16;
    static constexpr size_t BUFN = 2048;
    FlatMap single;
    bool part = false;
    std::vector<FlatMap> tab;
    std::vector<std::vector<Term>> buf;

    explicit RadixAgg(size_t hint) : single(std::min(hint, THRESH)) {}

    void to_partitioned() {
        part = true;
        tab.reserve(NP);
        for (size_t p = 0; p < NP; ++p) tab.emplace_back(2 * THRESH / NP + 16);
        buf.assign(NP, Vec{});
        for (auto &b : buf) b.reserve(BUFN);
        FlatMap old(16);
        std::swap(old, single);
        for (size_t i = 0; i <= old.mask; ++i)
            if (old.k[i] != FlatMap::EMPTY) route(old.k[i], old.v[i]);
    }
    inline void route(Key key, long long val) {
        const size_t p = FlatMap::hsh(key) >> (64 - RB);
        auto &b = buf[p];
        b.emplace_back(key, val);
        if (b.size() >= BUFN) flush(p);
    }
    void flush(size_t p) {
        for (const auto &t : buf[p]) tab[p].add(t.first, t.second);
        buf[p].clear();
    }
    inline void add(Key key, long long val) {
        if (!part) {
            single.add(key, val);
            if (single.n > THRESH) to_partitioned();
            return;
        }
        route(key, val);
    }
    Vec to_vec() {
        if (!part) return single.to_vec();
        size_t tot = 0;
        for (size_t p = 0; p < NP; ++p) { flush(p); tot += tab[p].n; }
        Vec out;
        out.reserve(tot);
        for (size_t p = 0; p < NP; ++p)
            for (size_t i = 0; i <= tab[p].mask; ++i)
                if (tab[p].k[i] != FlatMap::EMPTY && tab[p].v[i] != 0) out.emplace_back(tab[p].k[i], tab[p].v[i]);
        return out;
    }
};

static std::vector<Part> distinct_perms(const Part &p) {
    std::array<uint8_t, K> v = p;
    std::sort(v.begin(), v.end());
    std::vector<Part> out;
    do { out.push_back(v); } while (std::next_permutation(v.begin(), v.end()));
    return out;
}

[[noreturn]] static void die(const char *msg) {
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

static inline void sort_net(Part &p) {
    if constexpr (K == 10) {
#define CE(a, b) { const uint8_t x = p[a], y = p[b]; p[a] = x > y ? x : y; p[b] = x > y ? y : x; }
        CE(4,9) CE(3,8) CE(2,7) CE(1,6) CE(0,5) CE(1,4) CE(6,9) CE(0,3) CE(5,8) CE(0,2)
        CE(3,6) CE(7,9) CE(0,1) CE(2,4) CE(5,7) CE(8,9) CE(1,2) CE(4,6) CE(7,8) CE(3,5)
        CE(2,5) CE(6,8) CE(1,3) CE(4,7) CE(2,3) CE(6,7) CE(3,4) CE(5,6) CE(4,5)
#undef CE
    } else {
        sort_desc(p);
    }
}

static inline uint32_t block_mask(const Part &lam) {
    uint32_t m = 0;
    for (int p = 1; p < K; ++p)
        if (lam[p] == lam[p - 1]) m |= 1u << (p - 1);
    return m;
}

static Vec BB_canon(const Vec &P, const std::vector<Part> &perms) {
    std::vector<std::vector<uint32_t>> bucket(1u << (K - 1));
    for (uint32_t i = 0; i < (uint32_t)P.size(); ++i) bucket[block_mask(unpack(P[i].first))].push_back(i);
    RadixAgg out(P.size() * 4 + 16);
    std::vector<Part> canon;
    std::vector<long long> orb;
    for (uint32_t m = 0; m < (uint32_t)bucket.size(); ++m) {
        if (bucket[m].empty()) continue;
        canon.clear();
        orb.clear();
        for (const Part &pi : perms) {
            bool ok = true;
            for (int p = 1; p < K && ok; ++p)
                if ((m >> (p - 1)) & 1u) ok = pi[p - 1] >= pi[p];
            if (!ok) continue;
            long long o = 1;
            int i = 0;
            while (i < K) {
                int j = i + 1;
                while (j < K && ((m >> (j - 1)) & 1u)) ++j;
                o *= FACTL[j - i];
                int u = i;
                while (u < j) {
                    int w = u + 1;
                    while (w < j && pi[w] == pi[u]) ++w;
                    o /= FACTL[w - u];
                    u = w;
                }
                i = j;
            }
            canon.push_back(pi);
            orb.push_back(o);
        }
        long long maxorb = 0;
        for (long long o : orb) maxorb = std::max(maxorb, o);
        for (uint32_t idx : bucket[m]) {
            const Part lam = unpack(P[idx].first);
            const long long hl = hash_mult(lam);
            if (P[idx].second % hl) die("divisibility invariant violated");
            const long long q = P[idx].second / hl;
            long long guard;
            if (__builtin_mul_overflow(q < 0 ? -q : q, maxorb, &guard) || guard > (LLONG_MAX >> 12))
                die("coefficient overflow risk");
            for (size_t c = 0; c < canon.size(); ++c) {
                Part nu;
                for (int i = 0; i < K; ++i) nu[i] = (uint8_t)(lam[i] + canon[c][i]);
                sort_net(nu);
                out.add(pack(nu), q * orb[c]);
            }
        }
    }
    Vec r = out.to_vec();
    for (auto &t : r) {
        long long w;
        if (__builtin_mul_overflow(t.second, hash_mult(unpack(t.first)), &w)) die("coefficient overflow");
        t.second = w;
    }
    return r;
}

static Vec BB_exact(const Vec &P, const std::vector<Part> &perms) {
    FlatMap out(P.size() * perms.size() + 16);
    for (const auto &t : P) {
        const Part lam = unpack(t.first);
        const long long hl = hash_mult(lam);
        FlatMap tmp(perms.size() + 16);
        for (const Part &pi : perms) {
            Part nu;
            for (int i = 0; i < K; ++i) nu[i] = (uint8_t)(lam[i] + pi[i]);
            sort_desc(nu);
            tmp.add(pack(nu), hash_mult(nu));
        }
        for (size_t i = 0; i <= tmp.mask; ++i)
            if (tmp.k[i] != FlatMap::EMPTY && tmp.v[i]) {
                if (tmp.v[i] % hl) die("non-integer contraction in BB_exact");
                out.add(tmp.k[i], t.second * (tmp.v[i] / hl));
            }
    }
    return out.to_vec();
}

static Vec FF_square() {
    Part delta;
    for (int i = 0; i < K; ++i) delta[i] = (uint8_t)(K - 1 - i);
    std::array<uint8_t, K> v = delta;
    std::sort(v.begin(), v.end());
    FlatMap out(1 << 12);
    do {
        int s = 1;
        std::array<uint8_t, K> b = delta;
        for (int i = 0; i < K; ++i) {
            int j = i;
            while (j < K - 1 && b[j] != v[i]) ++j;
            if (j != i) { std::swap(b[i], b[j]); s = -s; }
        }
        Part nu;
        for (int i = 0; i < K; ++i) nu[i] = (uint8_t)(delta[i] + v[i]);
        sort_desc(nu);
        out.add(pack(nu), (long long)s * hash_mult(nu));
    } while (std::next_permutation(v.begin(), v.end()));
    return out.to_vec();
}

static long double wigner3j(int j1, int j2, int j3, int m1, int m2, int m3) {
    if (m1 + m2 + m3 != 0) return 0.0L;
    if (j1 + j2 < j3 || j2 + j3 < j1 || j3 + j1 < j2) return 0.0L;
    if (std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(m3) > j3) return 0.0L;
    if ((j1 + m1) % 2 || (j2 + m2) % 2 || (j3 + m3) % 2) return 0.0L;
    if ((j1 + j2 + j3) % 2) return 0.0L;
    const int a = (j1 + j2 - j3) / 2, b = (j1 - j2 + j3) / 2;
    const int c = (-j1 + j2 + j3) / 2, d = (j1 + j2 + j3) / 2 + 1;
    long double tri = FACT[a] * FACT[b] * FACT[c] / FACT[d];
    long double pre = FACT[(j1 + m1) / 2] * FACT[(j1 - m1) / 2] * FACT[(j2 + m2) / 2] *
                      FACT[(j2 - m2) / 2] * FACT[(j3 + m3) / 2] * FACT[(j3 - m3) / 2];
    int klo = std::max(0, std::max(-(j3 - j2 + m1) / 2, -(j3 - j1 - m2) / 2));
    int khi = std::min(a, std::min((j1 - m1) / 2, (j2 + m2) / 2));
    long double s = 0.0L;
    for (int k = klo; k <= khi; ++k) {
        long double t = FACT[k] * FACT[a - k] * FACT[(j1 - m1) / 2 - k] * FACT[(j2 + m2) / 2 - k] *
                        FACT[(j3 - j2 + m1) / 2 + k] * FACT[(j3 - j1 - m2) / 2 + k];
        s += ((k & 1) ? -1.0L : 1.0L) / t;
    }
    const int ph = (j1 - j2 - m3) / 2;
    return ((ph & 1) ? -1.0L : 1.0L) * std::sqrt(tri) * std::sqrt(pre) * s;
}

static double part_norm(const Part &p) {
    int n[NORB] = {0};
    for (int i = 0; i < K; ++i) ++n[p[i]];
    long double w = FACT[K];
    for (int e = 0; e < NORB; ++e) w /= FACT[n[e]];
    w = std::sqrt(w);
    for (int e = 0; e < NORB; ++e)
        if (n[e]) w /= std::pow(FACT[Q2] / (FACT[e] * FACT[Q2 - e]), 0.5L * n[e]);
    return (double)w;
}

static std::filesystem::path ckpt_path(const std::filesystem::path &dir, int s, bool tmp) {
    char buf[64];
    std::snprintf(buf, sizeof buf, tmp ? "s_%04d.tmp" : "s_%04d.txt", s);
    return dir / buf;
}

static void write_ckpt(const std::filesystem::path &dir, int s, long double nr,
                       const long double *uu, const long double *vv, const long double *ud) {
    const auto tp = ckpt_path(dir, s, true);
    FILE *f = std::fopen(tp.string().c_str(), "w");
    if (!f) die("cannot write checkpoint");
    std::fprintf(f, "%d %La\n", s, nr);
    for (int L = 0; L < NL; ++L) std::fprintf(f, "%La %La %La\n", uu[L], vv[L], ud[L]);
    std::fclose(f);
    std::filesystem::rename(tp, ckpt_path(dir, s, false));
}

static bool read_ckpt(const std::filesystem::path &dir, int s, long double &nr,
                      long double *uu, long double *vv, long double *ud) {
    FILE *f = std::fopen(ckpt_path(dir, s, false).string().c_str(), "r");
    if (!f) return false;
    int ss;
    bool ok = std::fscanf(f, "%d %La", &ss, &nr) == 2 && ss == s;
    for (int L = 0; ok && L < NL; ++L) ok = std::fscanf(f, "%La %La %La", &uu[L], &vv[L], &ud[L]) == 3;
    std::fclose(f);
    return ok;
}

int main(int argc, char **argv) {
    init_fact();

    int sb = 0, se = SHALF;
    if (argc >= 3) { sb = std::atoi(argv[1]); se = std::atoi(argv[2]); }
    sb = std::max(sb, 0);
    se = std::min(se, SHALF);

    char dname[96];
    std::snprintf(dname, sizeof dname, "ckpt2_Halperin_%d%d%d_N%d", MM, MM, NN, N);
    const std::filesystem::path dir(dname);
    std::filesystem::create_directories(dir);

    std::vector<int> todo;
    for (int s = sb; s <= se; ++s)
        if (!std::filesystem::exists(ckpt_path(dir, s, false))) todo.push_back(s);

    if (!todo.empty()) {
        Vec LU = FF_square();
        for (int r = 0; r < MM / 2 - 1; ++r) {
            Vec base = FF_square();
            FlatMap acc(LU.size() * 4);
            for (const auto &b : base) {
                Vec t = BB_canon(LU, distinct_perms(unpack(b.first)));
                for (const auto &x : t) acc.add(x.first, b.second * x.second);
            }
            LU = acc.to_vec();
        }

        std::vector<Vec> cd(NN * K + 1);
        {
            std::vector<Vec> cur(1);
            cur[0].emplace_back(pack(Part{}), 1LL);
            for (int r = 0; r < NN; ++r) {
                std::vector<FlatMap> nxt(cur.size() + K);
                for (size_t d = 0; d < cur.size(); ++d) {
                    if (cur[d].empty()) continue;
                    for (int j = 0; j <= K; ++j) {
                        Part ej{};
                        for (int i = 0; i < j; ++i) ej[i] = 1;
                        const long long sg = (j & 1) ? -1LL : 1LL;
                        for (const auto &t : BB_exact(cur[d], distinct_perms(ej)))
                            nxt[d + j].add(t.first, sg * t.second);
                    }
                }
                cur.assign(nxt.size(), Vec{});
                for (size_t d = 0; d < nxt.size(); ++d) cur[d] = nxt[d].to_vec();
            }
            for (size_t d = 0; d < cur.size() && d < cd.size(); ++d) cd[d] = cur[d];
        }
        std::vector<std::vector<std::vector<Part>>> cdPerms(cd.size());
        for (size_t d = 0; d < cd.size(); ++d)
            for (const auto &t : cd[d]) cdPerms[d].push_back(distinct_perms(unpack(t.first)));

        std::vector<std::vector<std::array<int, K>>> bucket(SMAX + 1);
        {
            std::array<int, K> cur{};
            auto rec = [&](auto &&self, int start, int depth, int sum) -> void {
                if (depth == K) { bucket[sum].push_back(cur); return; }
                for (int d = start; d <= NN * K; ++d) { cur[depth] = d; self(self, d, depth + 1, sum + d); }
            };
            rec(rec, 0, 0, 0);
        }

        const double FOURPI = (double)(4.0L * std::acos(-1.0L));
        std::vector<double> cLe((size_t)NL * NORB);
        for (int L = 0; L < NL; ++L) {
            long double w2 = wigner3j(Q2, 2 * L, Q2, -Q2, 0, Q2);
            for (int e = 0; e < NORB; ++e) {
                int m2 = 2 * e - Q2;
                long double w1 = wigner3j(Q2, 2 * L, Q2, -m2, 0, m2);
                long double sg = ((Q2 + e + L) & 1) ? -1.0L : 1.0L;
                cLe[(size_t)L * NORB + e] =
                    (double)(sg * std::sqrt((long double)((Q2 + 1) * (Q2 + 1) * (2 * L + 1)) / FOURPI) * w1 * w2);
            }
        }

        for (int s : todo) {
            auto ids = bucket[s];
            std::sort(ids.begin(), ids.end(), [](const std::array<int, K> &a, const std::array<int, K> &b) {
                auto cost = [](const std::array<int, K> &m) {
                    long long p = FACTL[K];
                    int i = 0;
                    while (i < K) { int j = i + 1; while (j < K && m[j] == m[i]) ++j; p /= FACTL[j - i]; i = j; }
                    return p;
                };
                return cost(a) > cost(b);
            });
            const int nm = (int)ids.size();

            std::vector<Vec> A(nm), B(nm);
#pragma omp parallel for schedule(dynamic, 1)
            for (int r = 0; r < nm; ++r) {
                Part kap;
                for (int i = 0; i < K; ++i) kap[i] = (uint8_t)(NN * K - ids[r][i]);
                sort_desc(kap);
                A[r] = BB_canon(LU, distinct_perms(kap));
                Vec V = LU;
                for (int i = 0; i < K; ++i) {
                    const int d = ids[r][i];
                    FlatMap acc(V.size() * 4 + 16);
                    for (size_t t = 0; t < cd[d].size(); ++t) {
                        const Vec W = BB_canon(V, cdPerms[d][t]);
                        const long long c = cd[d][t].second;
                        for (const auto &x : W) acc.add(x.first, c * x.second);
                    }
                    V = acc.to_vec();
                }
                B[r] = std::move(V);
            }

            std::vector<Key> LA, LB;
            FlatMap ia(1 << 16), ib(1 << 16);
            for (int r = 0; r < nm; ++r) {
                for (const auto &t : A[r]) if (ia.insert(t.first, (long long)LA.size())) LA.push_back(t.first);
                for (const auto &t : B[r]) if (ib.insert(t.first, (long long)LB.size())) LB.push_back(t.first);
            }
            const size_t na = LA.size(), nb = LB.size();

            long double tN = 0.0L, tUU[NL] = {0.0L}, tVV[NL] = {0.0L}, tUD[NL] = {0.0L};

            if (na && nb) {
                std::vector<double> fa(na), fb(nb), duT(na * NL), dvT(nb * NL);
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < na; ++i) {
                    const Part p = unpack(LA[i]);
                    fa[i] = part_norm(p);
                    for (int L = 0; L < NL; ++L) {
                        double t = 0.0;
                        for (int q = 0; q < K; ++q) t += cLe[(size_t)L * NORB + p[q]];
                        duT[i * NL + L] = t;
                    }
                }
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < nb; ++i) {
                    const Part p = unpack(LB[i]);
                    fb[i] = part_norm(p);
                    for (int L = 0; L < NL; ++L) {
                        double t = 0.0;
                        for (int q = 0; q < K; ++q) t += cLe[(size_t)L * NORB + p[q]];
                        dvT[i * NL + L] = t;
                    }
                }

                std::vector<double> Ad((size_t)nm * na, 0.0), Bd((size_t)nm * nb, 0.0);
#pragma omp parallel for schedule(dynamic, 1)
                for (int r = 0; r < nm; ++r) {
                    for (const auto &t : A[r]) { size_t i = (size_t)ia.find(t.first); Ad[(size_t)r * na + i] = (double)t.second * fa[i]; }
                    for (const auto &t : B[r]) { size_t i = (size_t)ib.find(t.first); Bd[(size_t)r * nb + i] = (double)t.second * fb[i]; }
                    Vec().swap(A[r]);
                    Vec().swap(B[r]);
                }

#pragma omp parallel
                {
                    long double lN = 0.0L, lUU[NL] = {0.0L}, lVV[NL] = {0.0L}, lUD[NL] = {0.0L};
                    double gA1[NL], gA2[NL], hB[NL], hB2[NL];
#pragma omp for schedule(dynamic, 1) nowait
                    for (int rr = 0; rr < nm; ++rr) {
                        const int r = nm - 1 - rr;
                        const double *ar = &Ad[(size_t)r * na];
                        const double *br = &Bd[(size_t)r * nb];
                        for (int q = 0; q <= r; ++q) {
                            const double *aq = &Ad[(size_t)q * na];
                            const double *bq = &Bd[(size_t)q * nb];
                            double g1 = 0.0, gb = 0.0;
                            for (int L = 0; L < NL; ++L) { gA1[L] = 0.0; gA2[L] = 0.0; hB[L] = 0.0; hB2[L] = 0.0; }
                            for (size_t i = 0; i < na; ++i) {
                                const double x = ar[i] * aq[i];
                                if (x == 0.0) continue;
                                g1 += x;
                                const double *d = &duT[i * NL];
                                for (int L = 0; L < NL; ++L) { const double xd = x * d[L]; gA1[L] += xd; gA2[L] += xd * d[L]; }
                            }
                            for (size_t j = 0; j < nb; ++j) {
                                const double y = br[j] * bq[j];
                                if (y == 0.0) continue;
                                gb += y;
                                const double *e = &dvT[j * NL];
                                for (int L = 0; L < NL; ++L) { const double ye = y * e[L]; hB[L] += ye; hB2[L] += ye * e[L]; }
                            }
                            const long double w = (r == q) ? 1.0L : 2.0L;
                            lN += w * (long double)g1 * gb;
                            for (int L = 0; L < NL; ++L) {
                                lUU[L] += w * (long double)gA2[L] * gb;
                                lVV[L] += w * (long double)g1 * hB2[L];
                                lUD[L] += w * (long double)gA1[L] * hB[L];
                            }
                        }
                    }
#pragma omp critical
                    {
                        tN += lN;
                        for (int L = 0; L < NL; ++L) { tUU[L] += lUU[L]; tVV[L] += lVV[L]; tUD[L] += lUD[L]; }
                    }
                }
            }
            write_ckpt(dir, s, tN, tUU, tVV, tUD);
        }
    }

    long double nrm2 = 0.0L, NUU[NL] = {0.0L}, NUD[NL] = {0.0L};
    int done = 0;
    for (int s = 0; s <= SHALF; ++s) {
        long double nr, uu[NL], vv[NL], ud[NL];
        if (!read_ckpt(dir, s, nr, uu, vv, ud)) continue;
        ++done;
        const bool self = (s == SMAX - s);
        nrm2 += self ? nr : 2.0L * nr;
        for (int L = 0; L < NL; ++L) {
            NUU[L] += self ? uu[L] : uu[L] + vv[L];
            NUD[L] += self ? ud[L] : 2.0L * ud[L];
        }
    }
    if (done != SHALF + 1) {
        std::printf("%d of %d blocks complete\n", done, SHALF + 1);
        return 0;
    }

    const long double FOURPI = 4.0L * std::acos(-1.0L);
    std::printf("Halperin (%d,%d,%d)  Nu=Nv=%d  N=%d  2Q=%d\n", MM, MM, NN, K, N, Q2);
    std::printf("%4s %16s %16s %16s %16s %16s %16s %16s\n",
                "L", "Sbar_uu", "Sbar_ud", "Sbar", "O(L)", "S_uu", "S_ud", "S");
    const long double frac = (long double)K / N;
    for (int L = 0; L <= Q2 + LEXTRA; ++L) {
        long double buu = (L <= Q2) ? FOURPI / N * NUU[L] / nrm2 : 0.0L;
        long double bud = (L <= Q2) ? FOURPI / N * NUD[L] / nrm2 : 0.0L;
        long double w2 = wigner3j(Q2, 2 * L, Q2, -Q2, 0, Q2);
        long double O = 1.0L - (long double)(Q2 + 1) * w2 * w2;
        long double suu = buu + frac * O;
        std::printf("%4d %16.10Lf %16.10Lf %16.10Lf %16.10Lf %16.10Lf %16.10Lf %16.10Lf\n",
                    L, buu, bud, 2.0L * (buu + bud), O, suu, bud, 2.0L * (suu + bud));
    }
    return 0;
}
