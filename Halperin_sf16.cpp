#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

constexpr int K = 8;
constexpr int MM = 2;
constexpr int NN = 1;

constexpr int N = 2 * K;
constexpr int Q2 = MM * (K - 1) + NN * K;
constexpr int NORB = Q2 + 1;
constexpr int LEXTRA = 2;


using Part = std::array<uint8_t, K>;

struct PartHash {
    size_t operator()(const Part &p) const noexcept {
        size_t h = 1469598103934665603ull;
        for (int i = 0; i < K; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        return h;
    }
};

using Poly = std::unordered_map<Part, long long, PartHash>;

static long double FACT[256];

static void init_fact() {
    FACT[0] = 1.0L;
    for (int i = 1; i < 256; ++i) FACT[i] = FACT[i - 1] * (long double)i;
}

static inline Part sort_desc(Part p) {
    for (int i = 1; i < K; ++i) {
        uint8_t x = p[i];
        int j = i - 1;
        while (j >= 0 && p[j] < x) { p[j + 1] = p[j]; --j; }
        p[j + 1] = x;
    }
    return p;
}

static long long hash_mult(const Part &p) {
    long long f = 1;
    int i = 0;
    while (i < K) {
        int j = i;
        while (j < K && p[j] == p[i]) ++j;
        for (int t = 2; t <= j - i; ++t) f *= t;
        i = j;
    }
    return f;
}

static std::vector<Part> distinct_perms(const Part &p) {
    std::vector<uint8_t> v(p.begin(), p.end());
    std::sort(v.begin(), v.end());
    std::vector<Part> out;
    do {
        Part q;
        for (int i = 0; i < K; ++i) q[i] = v[i];
        out.push_back(q);
    } while (std::next_permutation(v.begin(), v.end()));
    return out;
}

static void add_term(Poly &P, const Part &k, long long c) {
    if (!c) return;
    auto it = P.find(k);
    if (it == P.end()) P.emplace(k, c);
    else { it->second += c; if (!it->second) P.erase(it); }
}

static Poly BB(const Poly &P, const Part &lam2) {
    std::vector<Part> perms = distinct_perms(lam2);
    Poly out;
    Poly tmp;
    for (const auto &kv : P) {
        const long long hl = hash_mult(kv.first);
        tmp.clear();
        for (const Part &pi : perms) {
            Part nu;
            for (int i = 0; i < K; ++i) nu[i] = kv.first[i] + pi[i];
            nu = sort_desc(nu);
            add_term(tmp, nu, hash_mult(nu));
        }
        for (const auto &t : tmp) add_term(out, t.first, kv.second * (t.second / hl));
    }
    return out;
}

static Poly FF_square() {
    Part delta;
    for (int i = 0; i < K; ++i) delta[i] = K - 1 - i;
    std::vector<uint8_t> v(delta.begin(), delta.end());
    std::sort(v.begin(), v.end());
    Poly out;
    do {
        int s = 1;
        std::vector<uint8_t> b(delta.begin(), delta.end());
        for (int i = 0; i < K; ++i) {
            int j = i;
            while (b[j] != v[i]) ++j;
            if (j != i) { std::swap(b[i], b[j]); s = -s; }
        }
        Part nu;
        for (int i = 0; i < K; ++i) nu[i] = delta[i] + v[i];
        nu = sort_desc(nu);
        add_term(out, nu, (long long)s * hash_mult(nu));
    } while (std::next_permutation(v.begin(), v.end()));
    return out;
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

static long double part_norm(const Part &p, std::array<double, NORB> &nocc) {
    int n[NORB] = {0};
    for (int i = 0; i < K; ++i) ++n[p[i]];
    long double w = FACT[K];
    for (int e = 0; e < NORB; ++e) w /= FACT[n[e]];
    w = std::sqrt(w);
    for (int e = 0; e < NORB; ++e) {
        nocc[e] = n[e];
        if (n[e]) w /= std::pow(FACT[Q2] / (FACT[e] * FACT[Q2 - e]), 0.5L * n[e]);
    }
    return w;
}

int main() {
    init_fact();

    Poly LU = FF_square();
    for (int r = 0; r < MM / 2 - 1; ++r) {
        Poly base = FF_square(), acc;
        for (const auto &kv : base)
            for (const auto &t : BB(LU, kv.first)) add_term(acc, t.first, kv.second * t.second);
        LU.swap(acc);
    }

    std::vector<Poly> cd(NN * K + 1);
    {
        std::vector<Poly> cur(1);
        Part zero{};
        cur[0].emplace(zero, 1LL);
        for (int r = 0; r < NN; ++r) {
            std::vector<Poly> nxt(cur.size() + K);
            for (size_t d = 0; d < cur.size(); ++d) {
                if (cur[d].empty()) continue;
                for (int j = 0; j <= K; ++j) {
                    Part ej{};
                    for (int i = 0; i < j; ++i) ej[i] = 1;
                    long long sg = (j & 1) ? -1LL : 1LL;
                    for (const auto &t : BB(cur[d], ej)) add_term(nxt[d + j], t.first, sg * t.second);
                }
            }
            cur.swap(nxt);
        }
        for (size_t d = 0; d < cur.size() && d < cd.size(); ++d) cd[d] = cur[d];
    }

    std::vector<std::array<int, K>> msets;
    {
        std::array<int, K> cur{};
        auto rec = [&](auto &&self, int start, int depth) -> void {
            if (depth == K) { msets.push_back(cur); return; }
            for (int d = start; d <= NN * K; ++d) { cur[depth] = d; self(self, d, depth + 1); }
        };
        rec(rec, 0, 0);
    }

    const int NMS = (int)msets.size();
    std::vector<Poly> Am(NMS), Bm(NMS);
    std::vector<int> sval(NMS);

#pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < NMS; ++t) {
        int s = 0;
        for (int i = 0; i < K; ++i) s += msets[t][i];
        sval[t] = s;

        Part kap;
        for (int i = 0; i < K; ++i) kap[i] = (uint8_t)(NN * K - msets[t][i]);
        kap = sort_desc(kap);
        Am[t] = BB(LU, kap);

        Poly V = LU;
        for (int i = 0; i < K; ++i) {
            Poly acc;
            for (const auto &kv : cd[msets[t][i]])
                for (const auto &u : BB(V, kv.first)) add_term(acc, u.first, kv.second * u.second);
            V.swap(acc);
        }
        Bm[t] = std::move(V);
    }

    int smax = 0;
    for (int t = 0; t < NMS; ++t) smax = std::max(smax, sval[t]);

    const long double FOURPI = 4.0L * std::acos(-1.0L);
    const int LMAX = Q2 + LEXTRA;
    std::vector<long double> NUU(LMAX + 1, 0.0L), NUD(LMAX + 1, 0.0L);
    long double nrm2 = 0.0L;

    for (int s = 0; s <= smax; ++s) {
        std::vector<int> ids;
        for (int t = 0; t < NMS; ++t)
            if (sval[t] == s && !Am[t].empty() && !Bm[t].empty()) ids.push_back(t);
        if (ids.empty()) continue;
        const int nm = (int)ids.size();

        std::vector<Part> LA, LB;
        std::unordered_map<Part, int, PartHash> ia, ib;
        for (int t : ids) {
            for (const auto &kv : Am[t]) if (ia.emplace(kv.first, (int)LA.size()).second) LA.push_back(kv.first);
            for (const auto &kv : Bm[t]) if (ib.emplace(kv.first, (int)LB.size()).second) LB.push_back(kv.first);
        }
        const int na = (int)LA.size(), nb = (int)LB.size();

        std::vector<double> A((size_t)nm * na, 0.0), B((size_t)nm * nb, 0.0);
        std::vector<std::array<double, NORB>> oA(na), oB(nb);
        std::vector<double> fa(na), fb(nb);
        for (int i = 0; i < na; ++i) fa[i] = (double)part_norm(LA[i], oA[i]);
        for (int i = 0; i < nb; ++i) fb[i] = (double)part_norm(LB[i], oB[i]);
        for (int r = 0; r < nm; ++r) {
            for (const auto &kv : Am[ids[r]]) { int i = ia[kv.first]; A[(size_t)r * na + i] = (double)kv.second * fa[i]; }
            for (const auto &kv : Bm[ids[r]]) { int i = ib[kv.first]; B[(size_t)r * nb + i] = (double)kv.second * fb[i]; }
        }

        std::vector<double> GB((size_t)nm * nm, 0.0);
        for (int r = 0; r < nm; ++r)
            for (int q = 0; q <= r; ++q) {
                double h = 0.0;
                for (int i = 0; i < nb; ++i) h += B[(size_t)r * nb + i] * B[(size_t)q * nb + i];
                GB[(size_t)r * nm + q] = GB[(size_t)q * nm + r] = h;
                double g = 0.0;
                for (int i = 0; i < na; ++i) g += A[(size_t)r * na + i] * A[(size_t)q * na + i];
                nrm2 += (long double)g * h * (r == q ? 1.0L : 2.0L);
            }

#pragma omp parallel for schedule(dynamic)
        for (int L = 0; L <= Q2; ++L) {
            double c[NORB];
            long double w2 = wigner3j(Q2, 2 * L, Q2, -Q2, 0, Q2);
            for (int e = 0; e < NORB; ++e) {
                int m2 = 2 * e - Q2;
                long double w1 = wigner3j(Q2, 2 * L, Q2, -m2, 0, m2);
                long double sg = ((Q2 + e + L) & 1) ? -1.0L : 1.0L;
                c[e] = (double)(sg * std::sqrt((long double)((Q2 + 1) * (Q2 + 1) * (2 * L + 1)) / FOURPI) * w1 * w2);
            }
            std::vector<double> du(na), dv(nb);
            for (int i = 0; i < na; ++i) { double t = 0; for (int e = 0; e < NORB; ++e) t += c[e] * oA[i][e]; du[i] = t; }
            for (int i = 0; i < nb; ++i) { double t = 0; for (int e = 0; e < NORB; ++e) t += c[e] * oB[i][e]; dv[i] = t; }
            long double auu = 0.0L, aud = 0.0L;
            for (int r = 0; r < nm; ++r)
                for (int q = 0; q < nm; ++q) {
                    double g2 = 0.0, g1 = 0.0, hb = 0.0;
                    for (int i = 0; i < na; ++i) {
                        double x = A[(size_t)r * na + i] * du[i], y = A[(size_t)q * na + i];
                        g2 += x * y * du[i];
                        g1 += x * y;
                    }
                    for (int i = 0; i < nb; ++i) hb += B[(size_t)r * nb + i] * B[(size_t)q * nb + i] * dv[i];
                    auu += (long double)g2 * GB[(size_t)r * nm + q];
                    aud += (long double)g1 * hb;
                }
#pragma omp critical
            { NUU[L] += auu; NUD[L] += aud; }
        }
    }

    std::printf("Halperin (%d,%d,%d)  Nu=Nv=%d  N=%d  2Q=%d\n", MM, MM, NN, K, N, Q2);
    std::printf("%4s %16s %16s %16s %16s %16s %16s %16s\n",
                "L", "Sbar_uu", "Sbar_ud", "Sbar", "O(L)", "S_uu", "S_ud", "S");
    const long double frac = (long double)K / N;
    for (int L = 0; L <= LMAX; ++L) {
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
