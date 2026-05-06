#include "run.hpp"
#include "tick.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <ranges>
#include <string>
#include <vector>

void readAsset(std::vector<float> &prices, const std::string &filename) {
  std::ifstream file(filename);

  float value;
  while (file >> value) {
    prices.push_back(value);
  }
}

void loadAllAssets(std::array<std::vector<float>, N_STOCKS> &assets,
                  std::array<std::string, N_STOCKS> &stockNames) {
  std::string folder = "../inputs"; // importante: la llamada a esta funcion hay
                                    // que hacerla desde un directorio de build
                                    // (o cualquier otra subcarpeta)
  int i = 0;
  int minAssetCount = INT_MAX;

  for (const auto &entry : std::filesystem::directory_iterator(folder)) {
    if (entry.is_regular_file()) {
      std::string path = entry.path().string();
      std::cout << "Loading: " << path << std::endl; // por ahora para debuggear

      std::vector<float> prices;
      readAsset(prices, path);
      stockNames[i] = path.substr(folder.length() + 1, path.length() - folder.length() + 1);
      assets[i++] = prices;

      minAssetCount = std::min(minAssetCount, static_cast<int>(prices.size()));

      if (i == N_STOCKS)
        break;
    }
  }

  for (auto &stock : assets) {
    stock.resize(minAssetCount);
  }

}

std::vector<Tick> buildTicks(std::array<std::vector<float>, N_STOCKS> &assets,
                             std::array<std::string, N_STOCKS> &stockNames) {
  std::vector<Tick> ticks;

  for (int t = 0; t < assets[0].size(); t++) {
    Tick tick;

    for (std::size_t a = 0; a < N_STOCKS; a++) {
      tick.prices[a] = assets[a][t];
    }

    ticks.push_back(tick);
  }

  return ticks;
}

int main() {
  std::array<std::vector<float>, N_STOCKS> assets;
  std::array<std::string, N_STOCKS> stockNames;

  loadAllAssets(assets, stockNames);

  auto ticks = buildTicks(assets, stockNames);

  auto positions = runTicks(ticks);

  for (const auto &[stock, pos] : std::views::zip(stockNames, positions)) {
    std::cout << stock << " " << pos << '\n';
  }

  return 0;
}