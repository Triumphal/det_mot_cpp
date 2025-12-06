#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <vector>

#define TQDM_MAJOR_VERSION 0
#define TQDM_MINOR_VERSION 1

namespace {

void print_gauge(double x) {
  // write percentage
  std::cout << std::setfill(' ') << std::right << std::setw(3) << int(x * 100)
            << "%|";

  // write gauge
  for (uint i = 0; i < 20; ++i) {
    if (double(i) / 20. < x) {
      std::cout << "█";
    } else {
      std::cout << " ";
    }
  }
  std::cout << "|";
}

class Pbar {
 public:
  Pbar(int size, const std::map<std::string, std::string>& kwargs)
      : _size(size), _start_time(std::chrono::system_clock::now()) {
    if (kwargs.find("desc") != kwargs.end()) _desc = kwargs.at("desc");
    if (kwargs.find("leave") != kwargs.end())
      std::istringstream(kwargs.at("leave")) >> std::boolalpha >> _leave;
  }

  Pbar(int size) : Pbar(size, {}) {}

  ~Pbar() {}

  void update(int n);

  static int _n_instance;

 private:
  int _size;

  // description for tqdm
  std::string _desc = "";
  bool _leave = true;

  int _counter{0};
  std::chrono::system_clock::time_point _start_time;
  std::chrono::system_clock::time_point _pre_time;
  std::time_t total_time{0};
};

int Pbar::_n_instance = 0;

void Pbar::update(int n) {
  if (_counter == 0) {
    _n_instance++;
    if (_n_instance > 1) std::cout << std::endl;
  }

  _counter = _counter + n;

  std::cout << "\r";

  // description
  if (!(_desc.empty())) std::cout << _desc << ": ";

  // gauge progress
  print_gauge(double(_counter) / double(_size));

  const auto curr_time = std::chrono::system_clock::now();

  // total time [ms] from beginning
  const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            curr_time - _start_time)
                            .count();
  // total time [s] from beginning
  const auto total_s =
      std::chrono::duration_cast<std::chrono::seconds>(curr_time - _start_time)
          .count();
  // total time [min] when it is devided into min and sec
  const auto total_time_m = int(total_s / 60);
  // total time [s] when it is devided into min and sec
  const auto total_time_s = total_s % 60;

  // estimated remaining time [s]
  const auto remain_s =
      int(total_s * double(_size - _counter) / double(_counter));
  // estimated remaining time [min] when it is devided into min and sec
  const auto remain_time_m = int(remain_s / 60);
  // estimated remaining time [s] when it is devided into min and sec
  const auto remain_time_s = remain_s % 60;

  // iteration number progress
  std::cout << " " << _counter << "/" << _size;
  // current time consumption
  std::cout << " [" << std::setfill('0') << std::right << std::setw(2)
            << total_time_m << ":" << std::setfill('0') << std::right
            << std::setw(2) << total_time_s << "<";
  // estimated remaining time consumption
  std::cout << std::setfill('0') << std::right << std::setw(2) << remain_time_m
            << ":" << std::setfill('0') << std::right << std::setw(2)
            << remain_time_s << ",  ";
  // average time consumption for each time consumptions
  std::cout << std::fixed << std::setprecision(2)
            << double(total_ms) * 1e-3 / double(_counter) << "s/it]";

  // 終端処理
  if (_counter == _size) {
    --_n_instance;

    if (_leave) {
      // 出力を残したまま改行する
      std::cout << std::endl;
    } else {
      std::cout << std::flush;

      if (_n_instance > 0) {
        // まだゲージが残っている場合はクリアして前ゲージの末端に移動
        std::cout << "\e[2K\e[1A\e[100C" << std::flush;
      } else {
        // ゲージが残っていない場合はクリアのみ
        std::cout << "\e[2K" << std::flush;
      }
    }
  } else {
    // 出力して改行しない
    std::cout << std::flush;
  }

  _pre_time = curr_time;
}

/**
 * override iterator syntax
 * (https://gist.github.com/jeetsukumaran/307264)
 */
template <typename T>
class TqdmIterator : public std::iterator<std::input_iterator_tag, T> {
 public:
  // corresponds to begin() and end() methods in Tqdm class
  TqdmIterator(typename std::vector<T>::iterator iterator, size_t size,
               const std::map<std::string, std::string>& kwargs)
      : _pbar(size, kwargs) {
    _iterator = iterator;
  }

  // The same with normal iterators
  T& operator*() { return *_iterator; }
  bool operator==(const TqdmIterator& rhs) {
    return _iterator == rhs._iterator;
  }
  bool operator!=(const TqdmIterator& rhs) {
    return _iterator != rhs._iterator;
  }

  TqdmIterator operator++() {
    // The same with usual iterators
    TqdmIterator i = *this;
    _iterator++;
    _pbar.update(1);
    return i;
  };

  TqdmIterator operator++(int junk) {
    _iterator++;
    _pbar.update(1);
    return *this;
  }

 private:
  typename std::vector<T>::iterator _iterator;
  Pbar _pbar;
};

/**
 * tqdm class by overriding "range-based for" syntax
 * (https://en.cppreference.com/w/cpp/language/range-for)
 */
template <typename T>
class Tqdm {
 public:
  Tqdm(const std::vector<T>& vec,
       const std::map<std::string, std::string>& kwargs)
      : _vec(vec), _kwargs(kwargs) {}

  using iterator = TqdmIterator<T>;

  // begin() and end() method defines the way to create iterator
  iterator begin() { return iterator(_vec.begin(), _vec.size(), _kwargs); }
  iterator end() { return iterator(_vec.end(), _vec.size(), _kwargs); }

 private:
  std::vector<T> _vec;
  std::map<std::string, std::string> _kwargs;
};
}  // namespace

namespace tqdm {

Pbar tqdm(int total) { return Pbar(total); }

template <typename T>
Tqdm<T> tqdm(const std::vector<T>& vec,
             const std::map<std::string, std::string>& kwargs) {
  return Tqdm<T>(vec, kwargs);
}

template <typename T>
Tqdm<T> tqdm(const std::vector<T>& vec) {
  return Tqdm<T>(vec, {});
}

Tqdm<int> trange(int num, const std::map<std::string, std::string>& kwargs) {
  std::vector<int> v(num);
  std::iota(v.begin(), v.end(), 0);
  return Tqdm<int>(v, kwargs);
}

Tqdm<int> trange(int num) { return trange(num, {}); }

}  // namespace tqdm