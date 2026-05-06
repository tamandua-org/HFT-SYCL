#include "consts.hpp"
#include "pair_info.hpp"
#include "tick.hpp"
#include "run.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <sycl/sycl.hpp>


void constructFinalPositionFromAllInputs(
    const std::vector<PairInfo> &h_pairs,
    std::array<uint8_t, N_STOCKS> &positions) {

    std::array<std::array<int, 3>, N_STOCKS> sumPositions{};

    for (const auto pair: h_pairs){
        int i = pair.i;
        int j = pair.j;

        if(pair.position == BUY){
            sumPositions[i][BUY]++;
            sumPositions[j][SELL]++;
        }
        else if(pair.position == SELL){
            sumPositions[j][BUY]++;
            sumPositions[i][SELL]++;
        }
        else{
            sumPositions[j][HOLD]++;
            sumPositions[i][HOLD]++;
        }
    }

    for (int i = 0; i < N_STOCKS; ++i){
        if (sumPositions[i][BUY] > N_STOCKS / 4)
            positions[i] = BUY;
        else if (sumPositions[i][SELL] > N_STOCKS / 4)
            positions[i] = SELL;
        else
            positions[i] = HOLD;
    }

 
}

static void warmup_cpu(const std::vector<Tick>& ticks,
                       std::vector<PairInfo>& flat)
{
    int idx = 0;
    for (uint32_t i = 0; i < N_STOCKS; ++i) {
        for (uint32_t j = i + 1; j < N_STOCKS; ++j, ++idx) {
            // Pasada 1: meanI, meanJ, varJ, covIJ con Welford
            float meanI = 0.f, meanJ = 0.f, varJ = 0.f, covIJ = 0.f;
            for (uint32_t w = 0; w < WINDOW_SIZE; ++w) {
                const float xi = ticks[w].prices[i];
                const float xj = ticks[w].prices[j];
                const float oldI = meanI, oldJ = meanJ;
                meanI += (xi - meanI) / (w + 1);
                meanJ += (xj - meanJ) / (w + 1);
                varJ  += (xj - oldJ) * (xj - meanJ);
                covIJ += (xi - oldI) * (xj - meanJ);
            }
            varJ  /= WINDOW_SIZE;
            covIJ /= WINDOW_SIZE;
            const float beta = covIJ / std::max(varJ, EPSILON);

            // Pasada 2: meanSpread, varSpread con el beta recién calculado
            float meanSpread = 0.f, varSpread = 0.f;
            for (uint32_t w = 0; w < WINDOW_SIZE; ++w) {
                const float spread = ticks[w].prices[i] - beta * ticks[w].prices[j];
                const float oldM = meanSpread;
                meanSpread += (spread - meanSpread) / (w + 1);
                varSpread  += (spread - oldM) * (spread - meanSpread);
            }
            varSpread /= WINDOW_SIZE;

            PairInfo& p = flat[idx];
            p.meanI = meanI;        p.meanJ = meanJ;
            p.varJ  = varJ;         p.covIJ = covIJ;
            p.beta  = beta;
            p.meanSpread = meanSpread;  p.varSpread = varSpread;
            p.i = static_cast<uint16_t>(i);
            p.j = static_cast<uint16_t>(j);
            p.position = HOLD;
        }
    }
}


// ---- Parámetros del kernel ------------------------------------------------
constexpr uint32_t WG_SIZE          = 128;
constexpr uint32_t SLM_TARGET_BYTES = 32u * 1024u;          // headroom para casi cualquier GPU
constexpr uint32_t CHUNK_RAW        = SLM_TARGET_BYTES / (N_STOCKS * sizeof(float));
constexpr uint32_t CHUNK_SIZE       = (CHUNK_RAW < 32)  ? 32
                                    : (CHUNK_RAW > 512) ? 512
                                                        : CHUNK_RAW;


// ---- Kernel ---------------------------------------------------------------
static sycl::event z_score_kernel(sycl::queue& Q,
                                  const float* d_ticks,    // tick-major: tick*N_STOCKS + stock
                                  PairInfo* d_pairs,
                                  int n_ticks)
{
    const size_t n_groups = (N_PAIRS + WG_SIZE - 1) / WG_SIZE;
    const sycl::range<1> global_size{ n_groups * WG_SIZE };
    const sycl::range<1> local_size { WG_SIZE };

    return Q.submit([&](sycl::handler& cgh) {

        // Tile tick-major de CHUNK_SIZE ticks × N_STOCKS stocks
        sycl::local_accessor<float, 1> tile(sycl::range<1>(CHUNK_SIZE * N_STOCKS), cgh);

        cgh.parallel_for(sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> it) {
                const uint32_t lid = it.get_local_id(0);
                const uint32_t gid = it.get_global_id(0);
                const bool active  = gid < N_PAIRS;

                // Estado del par en registros (solo si el hilo es activo)
                PairInfo p;
                if (active) p = d_pairs[gid];

                for (int offset = 0; offset < n_ticks; offset += (int)CHUNK_SIZE) {
                    const int chunk = sycl::min((int)CHUNK_SIZE, n_ticks - offset);
                    const int total = chunk * (int)N_STOCKS;

                    // Carga colaborativa coalesced (tick-major → tick-major)
                    for (int t = (int)lid; t < total; t += (int)WG_SIZE) {
                        tile[t] = d_ticks[(size_t)offset * N_STOCKS + t];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (active) {
                        const uint32_t pi = p.i;
                        const uint32_t pj = p.j;

                        for (int k = 0; k < chunk; ++k) {
                            const float xi = tile[k * N_STOCKS + pi];
                            const float xj = tile[k * N_STOCKS + pj];

                            const float di = xi - p.meanI;
                            const float dj = xj - p.meanJ;
                            p.meanI = sycl::fma(ALPHA, di, p.meanI);
                            p.meanJ = sycl::fma(ALPHA, dj, p.meanJ);

                            const float omA = 1.0f - ALPHA;
                            p.varJ  = sycl::fma(omA, p.varJ,  ALPHA * dj * dj);
                            p.covIJ = sycl::fma(omA, p.covIJ, ALPHA * di * dj);

                            const float varJ_safe = sycl::fmax(p.varJ, EPSILON);
                            p.beta = p.covIJ / varJ_safe;

                            const float spread = sycl::fma(-p.beta, xj, xi);
                            const float ds     = spread - p.meanSpread;
                            p.meanSpread = sycl::fma(ALPHA, ds, p.meanSpread);
                            p.varSpread  = sycl::fma(omA, p.varSpread, ALPHA * ds * ds);

                            const float stddev = sycl::sqrt(sycl::fmax(p.varSpread, EPSILON));
                            const float z      = ds / stddev;

                            // Máquina de estados (compilador la convierte a select/predicate)
                            if (p.position == HOLD) {
                                if      (z >  THRESHOLD_ENTRY) p.position = SELL;
                                else if (z < -THRESHOLD_ENTRY) p.position = BUY;
                            } else if (p.position == BUY) {
                                if (z > -THRESHOLD_EXIT) p.position = HOLD;
                            } else { // SELL
                                if (z <  THRESHOLD_EXIT) p.position = HOLD;
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                if (active) d_pairs[gid] = p;
            });
    });
}

std::array<uint8_t, N_STOCKS> runTicks(const std::vector<Tick> &ticks) {
    const int total = static_cast<int>(ticks.size());
        if (total < (int)WINDOW_SIZE) return {};

        sycl::queue Q(sycl::gpu_selector_v,
                    sycl::property::queue::in_order{});

        // 1) Buffers en device
        float*         d_ticks = sycl::malloc_device<float>(
                                    static_cast<size_t>(total) * N_STOCKS, Q);
        PairInfo*  d_pairs = sycl::malloc_device<PairInfo>(N_PAIRS, Q);

        // 2) Una sola copia de TODOS los ticks (Tick ya es tick-major)
        auto copy_ticks = Q.memcpy(d_ticks, ticks.data(),
                                static_cast<size_t>(total) * sizeof(Tick));

        // 3) Warm-up en CPU en paralelo a la copia
        std::vector<PairInfo> h_pairs(N_PAIRS);
        warmup_cpu(ticks, h_pairs);

        // 4) Subir estado inicial de los pares y esperar a que los ticks estén
        auto copy_pairs = Q.memcpy(d_pairs, h_pairs.data(),
                                N_PAIRS * sizeof(PairInfo));
        copy_ticks.wait();
        copy_pairs.wait();

        // 5) UN solo kernel para todos los ticks post-warm-up
        const int n_post = total - (int)WINDOW_SIZE;
        if (n_post > 0) {
            z_score_kernel(Q,
                        d_ticks + (size_t)WINDOW_SIZE * N_STOCKS,
                        d_pairs,
                        n_post).wait();
        }

        // 6) Bajar resultados
        Q.memcpy(h_pairs.data(), d_pairs, N_PAIRS * sizeof(PairInfo)).wait();

        sycl::free(d_ticks, Q);
        sycl::free(d_pairs, Q);

        // h_pairs queda con el estado final; si necesitas el array 2D PairInfo,
        // reconviértelo aquí leyendo p.i, p.j de cada elemento.

        std::array<uint8_t, N_STOCKS> positions;
        constructFinalPositionFromAllInputs(h_pairs, positions);

        return positions;

}

