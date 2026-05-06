#include "asset.hpp"

#include "pair_info.hpp"
#include "tick.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include <sycl/sycl.hpp>

void convert(const std::vector<Tick> &ticks, std::array<Asset, N_STOCKS> &assets, int start, int size){

    int t = start;
    while (t < start + size){
        const auto &prices = ticks[t].prices;
        for (int i = 0; i < N_STOCKS; i++) {
            assets[i].ticks[t] = prices[i];
        }
        ++t;
    }
}

void convert(const std::vector<Tick>& ticks, float* h_ticks, int start, int size) {
    int t = start;
    for (int k = 0; k < size; ++k) {
        const auto& prices = ticks[t].prices;
        for (int i = 0; i < N_STOCKS; i++) {
            h_ticks[i * N_TICKS + k] = prices[i];
        }
        ++t;
    }
}

void flatten_pairs(
    const std::array<std::array<PairInfo, N_STOCKS>, N_STOCKS>& pairs,
    FlatPairInfo* h_pairs)
{
    int idx = 0;
    for (int i = 0; i < N_STOCKS; ++i)
        for (int j = i + 1; j < N_STOCKS; ++j) {
            const PairInfo& p = pairs[i][j];
            h_pairs[idx++] = {
                p.meanI, p.meanJ, p.varJ, p.covIJ,
                p.meanSpread, p.varSpread, p.beta,
                p.position, i, j
            };
        }
}

void unflatten_pairs(
    const FlatPairInfo* h_pairs,
    std::array<std::array<PairInfo, N_STOCKS>, N_STOCKS>& pairs)
{
    for (int idx = 0; idx < N_PAIRS; ++idx) {
        const FlatPairInfo& fp = h_pairs[idx];
        PairInfo& p = pairs[fp.i][fp.j];
        p.meanI      = fp.meanI;
        p.meanJ      = fp.meanJ;
        p.varJ       = fp.varJ;
        p.covIJ      = fp.covIJ;
        p.meanSpread = fp.meanSpread;
        p.varSpread  = fp.varSpread;
        p.beta       = fp.beta;
        p.position   = fp.position;
    }
}

void check_warmup_window(const std::array<Asset, N_STOCKS> assets, 
    std::array<std::array<PairInfo, N_STOCKS>, N_STOCKS> &pairs)
{   
    for (int i = 0; i < N_STOCKS; ++i){
        for (int j = i + 1; j < N_STOCKS; ++j){
            PairInfo &pair = pairs[i][j];
            for (int w = 0; w < WINDOW_SIZE; ++w){
                float x_i = assets[i].ticks[w];
                float x_j = assets[j].ticks[w];

                // Media Welford
                float oldMeanI = pair.meanI;
                float oldMeanJ = pair.meanJ;
                pair.meanI += (x_i - pair.meanI) / (w + 1);
                pair.meanJ += (x_j - pair.meanJ) / (w + 1);

                // Varianza y covarianza de Welford exactas
                // usa (x - oldMean) * (x - newMean) — fórmula online exacta
                pair.varJ  += (x_j - oldMeanJ) * (x_j - pair.meanJ);
                pair.covIJ += (x_i - oldMeanI) * (x_j - pair.meanJ);
            }
            pair.varJ  /= WINDOW_SIZE;
            pair.covIJ /= WINDOW_SIZE;

            float varJ_safe = std::max(pair.varJ, EPSILON);
            pair.beta = pair.covIJ / varJ_safe;
        }
    }

}

void z_score_batch(const std::array<Asset, N_STOCKS> assets,
std::array<std::array<PairInfo, N_STOCKS>, N_STOCKS> &pairs, int batch_size)
{
    for (int i = 0; i < N_STOCKS; ++i){
        for (int j = i + 1; j < N_STOCKS; ++j){
            PairInfo &pair = pairs[i][j];
            for (int k = 0; k < batch_size; ++k){
                float x_i = assets[i].ticks[k];
                float x_j = assets[j].ticks[k];

                float delta_i = x_i - pair.meanI;
                float delta_j = x_j - pair.meanJ;
                pair.meanI += ALPHA * delta_i;
                pair.meanJ += ALPHA * delta_j;

                pair.varJ = (1.0f - ALPHA) * pair.varJ + ALPHA * delta_j * delta_j;
                pair.covIJ = (1.0f - ALPHA) * pair.covIJ + ALPHA * delta_i * delta_j;


                float varJ_safe = std::max(pair.varJ, EPSILON);
                pair.beta = pair.covIJ / varJ_safe;

                float spread = x_i - pair.beta * x_j;

                float delta_s = spread - pair.meanSpread;

                pair.meanSpread += ALPHA * delta_s;
                pair.varSpread =
                    (1 - ALPHA) * pair.varSpread + ALPHA * delta_s * delta_s;

                float stddev = std::sqrt(std::max(pair.varSpread, EPSILON));

                float z = delta_s / stddev;


                if (pair.position == HOLD) {
                if (z > THRESHOLD_ENTRY)
                    pair.position = SELL;
                else if (z < -THRESHOLD_ENTRY)
                    pair.position = BUY;
                } else if (pair.position == BUY) {
                if (z > -THRESHOLD_EXIT)
                    pair.position = HOLD;
                } else if (pair.position == SELL) {
                if (z < THRESHOLD_EXIT)
                    pair.position = HOLD;
                }
            }
        }
    }
}

/* void z_score_kernel(sycl::queue& Q, const float* d_ticks, FlatPairInfo* d_pairs, int batch_size) {
    // N_PAIRS es 120 (16 * 15 / 2). 
    // Es una buena práctica en GPU que el tamaño del grupo de trabajo sea múltiplo de 32.
    constexpr int WG_SIZE = 128; 
    
    sycl::range<1> global_size{WG_SIZE};
    sycl::range<1> local_size{WG_SIZE};

    Q.submit([&](sycl::handler& cgh) {
        
        // 1. Declarar la memoria local (compartida por todos los hilos del work-group)
        // Tamaño necesario: N_STOCKS * batch_size floats
        sycl::local_accessor<float, 1> local_ticks(sycl::range<1>(N_STOCKS * batch_size), cgh);

        // Usamos nd_range para tener control explícito sobre el grupo de trabajo
        cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            
            int local_id = item.get_local_id(0);
            int global_id = item.get_global_id(0);

            // 2. CARGA COLABORATIVA (Global -> Local)
            // Los 128 hilos cooperan para cargar toda la matriz de ticks del batch actual.
            // Esto asegura que la lectura de memoria global sea coalesced (agrupada).
            int total_elements = N_STOCKS * batch_size;
            
            for (int i = local_id; i < total_elements; i += WG_SIZE) {
                int stock = i / batch_size;
                int tick = i % batch_size;
                
                // Nota: d_ticks tiene un salto de N_TICKS entre acciones según tu función convert
                local_ticks[stock * batch_size + tick] = d_ticks[stock * N_TICKS + tick];
            }

            // 3. BARRERA DE SINCRONIZACIÓN
            // Esperar a que todos los hilos terminen de copiar a la memoria local antes de continuar
            item.barrier();

            // 4. CÁLCULO INTENSIVO (Usando memoria local)
            // Solo procesamos los hilos reales (descartamos los 8 hilos extra de padding hasta 128)
            if (global_id < N_PAIRS) {
                FlatPairInfo pair = d_pairs[global_id];
                const int pi = pair.i;
                const int pj = pair.j;

                for (int k = 0; k < batch_size; ++k) {
                    
                    // ¡Lectura ultrarrápida desde la memoria local!
                    float x_i = local_ticks[pi * batch_size + k];
                    float x_j = local_ticks[pj * batch_size + k];

                    // Actualizar medias EMA
                    float delta_i = x_i - pair.meanI;
                    float delta_j = x_j - pair.meanJ;
                    pair.meanI += ALPHA * delta_i;
                    pair.meanJ += ALPHA * delta_j;

                    // Varianza y covarianza
                    pair.varJ  = (1.0f - ALPHA) * pair.varJ  + ALPHA * delta_j * delta_j;
                    pair.covIJ = (1.0f - ALPHA) * pair.covIJ + ALPHA * delta_i * delta_j;

                    float varJ_safe = sycl::max(pair.varJ, EPSILON);
                    pair.beta = pair.covIJ / varJ_safe;

                    // Spread y Z-Score
                    float spread  = x_i - pair.beta * x_j;
                    float delta_s = spread - pair.meanSpread;
                    pair.meanSpread += ALPHA * delta_s;
                    pair.varSpread   = (1.0f - ALPHA) * pair.varSpread + ALPHA * delta_s * delta_s;

                    float stddev = sycl::sqrt(sycl::max(pair.varSpread, EPSILON));
                    float z = delta_s / stddev;

                    // Máquina de estados para trading
                    if (pair.position == HOLD) {
                        if      (z >  THRESHOLD_ENTRY) pair.position = SELL;
                        else if (z < -THRESHOLD_ENTRY) pair.position = BUY;
                    } else if (pair.position == BUY) {
                        if (z > -THRESHOLD_EXIT) pair.position = HOLD;
                    } else if (pair.position == SELL) {
                        if (z <  THRESHOLD_EXIT) pair.position = HOLD;
                    }
                }
                
                // Escribir el estado final actualizado
                d_pairs[global_id] = pair;
            }
        });
    }).wait();
} */


void z_score_kernel(sycl::queue& Q, const float* d_ticks, FlatPairInfo* d_pairs, int batch_size) {
    constexpr int WG_SIZE = 128; 
    // Definimos el tamaño del bloque para no saturar la memoria local
    constexpr int CHUNK_SIZE = 256; 
    
    sycl::range<1> global_size{WG_SIZE};
    sycl::range<1> local_size{WG_SIZE};

    Q.submit([&](sycl::handler& cgh) {
        
        // 1. Memoria local ahora tiene un tamaño MÁXIMO FIJO y seguro
        sycl::local_accessor<float, 1> local_ticks(sycl::range<1>(N_STOCKS * CHUNK_SIZE), cgh);

        cgh.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
            int local_id = item.get_local_id(0);
            int global_id = item.get_global_id(0);

            // Cargamos el estado actual del par a los registros del hilo (memoria privada)
            FlatPairInfo pair;
            if (global_id < N_PAIRS) {
                pair = d_pairs[global_id];
            }

            // ==========================================
            // BUCLE EXTERNO: Procesamiento por bloques
            // ==========================================
            for (int offset = 0; offset < batch_size; offset += CHUNK_SIZE) {
                
                // Calculamos cuántos ticks quedan en este bloque (puede ser menor a CHUNK_SIZE al final)
                int current_chunk_size = sycl::min(CHUNK_SIZE, batch_size - offset);
                int total_elements = N_STOCKS * current_chunk_size;

                // 2. CARGA COLABORATIVA DEL BLOQUE ACTUAL (Global -> Local)
                for (int i = local_id; i < total_elements; i += WG_SIZE) {
                    int stock = i / current_chunk_size;
                    int tick = i % current_chunk_size;
                    
                    // Nota el desplazamiento (offset) en la memoria global
                    local_ticks[stock * current_chunk_size + tick] = 
                        d_ticks[stock * N_TICKS + (offset + tick)];
                }

                
                item.barrier();

                // 4. CÁLCULO INTENSIVO PARA EL BLOQUE ACTUAL
                if (global_id < N_PAIRS) {
                    const int pi = pair.i;
                    const int pj = pair.j;

                    for (int k = 0; k < current_chunk_size; ++k) {
                        float x_i = local_ticks[pi * current_chunk_size + k];
                        float x_j = local_ticks[pj * current_chunk_size + k];

                         // Actualizar medias EMA
                        float delta_i = x_i - pair.meanI;
                        float delta_j = x_j - pair.meanJ;
                        pair.meanI += ALPHA * delta_i;
                        pair.meanJ += ALPHA * delta_j;

                        // Varianza y covarianza
                        pair.varJ  = (1.0f - ALPHA) * pair.varJ  + ALPHA * delta_j * delta_j;
                        pair.covIJ = (1.0f - ALPHA) * pair.covIJ + ALPHA * delta_i * delta_j;

                        float varJ_safe = sycl::max(pair.varJ, EPSILON);
                        pair.beta = pair.covIJ / varJ_safe;

                        // Spread y Z-Score
                        float spread  = x_i - pair.beta * x_j;
                        float delta_s = spread - pair.meanSpread;
                        pair.meanSpread += ALPHA * delta_s;
                        pair.varSpread   = (1.0f - ALPHA) * pair.varSpread + ALPHA * delta_s * delta_s;

                        float stddev = sycl::sqrt(sycl::max(pair.varSpread, EPSILON));
                        float z = delta_s / stddev;

                        // Máquina de estados para trading
                        if (pair.position == HOLD) {
                            if      (z >  THRESHOLD_ENTRY) pair.position = SELL;
                            else if (z < -THRESHOLD_ENTRY) pair.position = BUY;
                        } else if (pair.position == BUY) {
                            if (z > -THRESHOLD_EXIT) pair.position = HOLD;
                        } else if (pair.position == SELL) {
                            if (z <  THRESHOLD_EXIT) pair.position = HOLD;
                        }
                    }
                }
                item.barrier();
            }
            // 6. Escribir el estado final actualizado de vuelta a memoria global
            if (global_id < N_PAIRS) {
                d_pairs[global_id] = pair;
            }
        });
    }).wait();
}

void runTicks(const std::vector<Tick> &ticks) {
    //Implementación en GPU
    /* std::array<Asset, N_STOCKS> assets;
    std::array<std::array<PairInfo, N_STOCKS>, N_STOCKS> pairs;
    int totalticks = ticks.size();
    int size = 0;

    convert(ticks, assets, 0, WINDOW_SIZE);
    check_warmup_window(assets, pairs);

    for (int w = WINDOW_SIZE; w < totalticks; w+= N_TICKS){
        size = std::min(int(N_TICKS), totalticks - w);
        convert(ticks, assets, w, size);
        //Parte paralelizable
        z_score_batch(assets, pairs, size);
        
    } */

    std::array<Asset, N_STOCKS> assets;
    std::array<std::array<PairInfo, N_STOCKS>, N_STOCKS> pairs;
    int totalticks = ticks.size();

    // Preferir la GPU si está disponible
    sycl::queue Q(sycl::gpu_selector_v);

    // 1. Usar malloc_shared: Memoria accesible por CPU y GPU sin memcpy
    float* shared_ticks = sycl::malloc_shared<float>(N_STOCKS * N_TICKS, Q);
    FlatPairInfo* shared_pairs = sycl::malloc_shared<FlatPairInfo>(N_PAIRS, Q);

    // Warmup en CPU
    convert(ticks, assets, 0, WINDOW_SIZE);      
    check_warmup_window(assets, pairs);          
    
    // La CPU escribe directamente en la memoria compartida
    flatten_pairs(pairs, shared_pairs);               

    for (int w = WINDOW_SIZE; w < totalticks; w += N_TICKS) {
        int size = std::min((int)N_TICKS, totalticks - w);

        // La CPU escribe los nuevos ticks directamente en la memoria compartida
        convert(ticks, shared_ticks, w, size); 
        
        // ¡Magia! Ya no necesitas Q.memcpy(). Lanzamos el kernel directamente.
        z_score_kernel(Q, shared_ticks, shared_pairs, size).wait();  
    }

    // La CPU lee directamente los resultados de la memoria compartida
    unflatten_pairs(shared_pairs, pairs);

    sycl::free(shared_ticks, Q);
    sycl::free(shared_pairs, Q);




void runTicks(const std::vector<Tick> &ticks) {
    //Implementación en GPU

    //1. Convertir la matriz de ticks por stock a stock por asset

    //2. Construir ventana inicial

    //3. Analizar los ticks nuevos




}