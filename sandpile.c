#include <errno.h>
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static void die(const char *s) {
  fprintf(stderr, "%s\n", s);
  exit(1);
}

int N = 801;
int **grid;
int *row_begin;
int *row_end;

// you can also use this function to debug print your sandpile
static void render(char *filename) {
  int chan = 3;
  unsigned char *pixels = (unsigned char *)malloc(N * N * chan * sizeof(char));
  int stride = N * chan;
  if (!pixels)
    die("malloc failed for PNG pixel buffer");
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      unsigned char r = 0, g = 0, b = 0;
      switch (grid[i][j]) {
      case 0: // obsidian
        r = 15;
        g = 15;
        b = 25;
        break;
      case 1: // electric cyan
        r = 0;
        g = 210;
        b = 255;
        break;
      case 2: // vibrant magenta
        r = 230;
        g = 30;
        b = 130;
        break;
      case 3: // bright gold
        r = 255;
        g = 200;
        b = 0;
        break;
      default: // white
        r = 0;
        g = 0;
        b = 0;
      }

      int pid = (i * N + j) * chan;
      pixels[pid] = r;
      pixels[pid + 1] = g;
      pixels[pid + 2] = b;
    }
  }
  if (stbi_write_png(filename, N, N, chan, pixels, stride)) {
    printf("Successfully saved sandpile identity to: %s\n", filename);
  } else {
    die("Error: stb_image_write failed to save png file\n");
  }
  free(pixels);
}

static void stabilize() {
  int threads = MIN(omp_get_max_threads(), N);
  int *shared_row = (int *)calloc(N, sizeof(int));
  int *special_row = (int *)calloc(N, sizeof(int));
  int *thread_changed = (int *)malloc(2 * threads * sizeof(int));
  int *thread_begin = (int *)malloc((threads + 1) * sizeof(int));
  if (!shared_row || !special_row || !thread_changed || !thread_begin)
    die("malloc failed for parallel stabilization");

  int64_t total_cells = 0;
  for (int i = 0; i < N; ++i)
    total_cells += row_end[i] - row_begin[i];
  thread_begin[0] = 0;
  thread_begin[threads] = N;
  int row = 0;
  int64_t cells = 0;
  for (int tid = 1; tid < threads; ++tid) {
    int64_t target = total_cells * tid / threads;
    while (row < N && 2 * cells + row_end[row] - row_begin[row] < 2 * target) {
      cells += row_end[row] - row_begin[row];
      ++row;
    }
    thread_begin[tid] = row;
    shared_row[row - 1] = 1;
    shared_row[row] = 1;
  }
  for (int i = 0; i < N; ++i) {
    if (i == 0 || i == N - 1 || row_begin[i] == 0 || row_end[i] == N)
      special_row[i] = 1;
    if (shared_row[i]) {
      special_row[i] = 1;
      if (i > 0)
        special_row[i - 1] = 1;
      if (i < N - 1)
        special_row[i + 1] = 1;
    }
  }

#pragma omp parallel num_threads(threads)
  {
    int tid = omp_get_thread_num();
    int begin = thread_begin[tid];
    int end = thread_begin[tid + 1];
    int round = 0;
    while (1) {
      int local_changed = 0;
      for (int i = begin; i < end; ++i) {
        if (!special_row[i]) {
          int first = row_begin[i];
          int last = row_end[i];
          int j = first & ~15;
          int vector_end = (last + 15) & ~15;
          __m512i one = _mm512_set1_epi32(1);
          __m512i three = _mm512_set1_epi32(3);
          __m512i four = _mm512_set1_epi32(4);
          for (; j < vector_end; j += 16) {
            __mmask16 valid = 0xffff;
            if (j < first)
              valid &= (__mmask16)(0xffffu << (first - j));
            if (j + 16 > last)
              valid &= (__mmask16)(0xffffu >> (j + 16 - last));
            __m512i value = _mm512_load_si512((__m512i *)&grid[i][j]);
            __mmask16 mask = _mm512_cmpgt_epi32_mask(value, three) & valid;
            if (!mask)
              continue;
            local_changed = 1;
            __mmask16 from_left = (__mmask16)(mask << 1);
            __mmask16 from_right = mask >> 1;
            value = _mm512_mask_sub_epi32(value, mask, value, four);
            value = _mm512_mask_add_epi32(value, from_left, value, one);
            value = _mm512_mask_add_epi32(value, from_right, value, one);
            _mm512_store_si512((__m512i *)&grid[i][j], value);
            value = _mm512_load_si512((__m512i *)&grid[i - 1][j]);
            _mm512_store_si512((__m512i *)&grid[i - 1][j],
                               _mm512_mask_add_epi32(value, mask, value, one));
            value = _mm512_load_si512((__m512i *)&grid[i + 1][j]);
            _mm512_store_si512((__m512i *)&grid[i + 1][j],
                               _mm512_mask_add_epi32(value, mask, value, one));
            if (mask & 1)
              grid[i][j - 1]++;
            if (mask & 0x8000)
              grid[i][j + 16]++;
          }
          continue;
        }
        for (int j = row_begin[i]; j < row_end[i]; ++j) {
          int grains;
          if (shared_row[i]) {
#pragma omp atomic read
            grains = grid[i][j];
          } else {
            grains = grid[i][j];
          }
          if (grains < 4)
            continue;
          local_changed = 1;
          if (shared_row[i]) {
#pragma omp atomic update
            grid[i][j] -= 4;
          } else {
            grid[i][j] -= 4;
          }
          if (i > 0) {
            if (shared_row[i - 1]) {
#pragma omp atomic update
              grid[i - 1][j]++;
            } else {
              grid[i - 1][j]++;
            }
          }
          if (j > 0) {
            if (shared_row[i]) {
#pragma omp atomic update
              grid[i][j - 1]++;
            } else {
              grid[i][j - 1]++;
            }
          }
          if (i < N - 1) {
            if (shared_row[i + 1]) {
#pragma omp atomic update
              grid[i + 1][j]++;
            } else {
              grid[i + 1][j]++;
            }
          }
          if (j < N - 1) {
            if (shared_row[i]) {
#pragma omp atomic update
              grid[i][j + 1]++;
            } else {
              grid[i][j + 1]++;
            }
          }
        }
      }
      int *changed = &thread_changed[(round & 1) * threads];
      changed[tid] = local_changed;
#pragma omp barrier
      int keep_going = 0;
      for (int t = 0; t < threads; ++t)
        keep_going |= changed[t];
      if (!keep_going)
        break;
      ++round;
    }
  }

  free(shared_row);
  free(special_row);
  free(thread_changed);
  free(thread_begin);
}

static void init_row_bounds() {
  int center = N / 2;
  int radius_squared = center * center;
  for (int i = 0; i < N; ++i) {
    int max_j = MAX(i, N - 1 - i);
    int vertical = i - center;
    row_begin[i] = max_j + 1;
    row_end[i] = max_j + 1;
    for (int j = 0; j <= max_j; ++j) {
      int horizontal = j - center;
      if (vertical * vertical + horizontal * horizontal <= radius_squared) {
        if (row_begin[i] > max_j)
          row_begin[i] = j;
        row_end[i] = j + 1;
      }
    }
  }
}

void find_identity() {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      grid[i][j] = 6;
    }
  }
  stabilize();
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      grid[i][j] = 6 - grid[i][j];
    }
  }
  stabilize();
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 2) {
    fprintf(stderr, "Usage: %s N\n", argv[0]);
    return EXIT_FAILURE;
  }
  char *end;
  uintmax_t parsed = strtoumax(argv[1], &end, 10);
  if (errno || argv[1][0] < '0' || argv[1][0] > '9' || *end || !parsed ||
      parsed > INT32_MAX)
    die("grid size must be a positive int");
  N = (int)parsed;

  grid = (int **)malloc(N * sizeof(int *));
  row_begin = (int *)malloc(N * sizeof(int));
  row_end = (int *)malloc(N * sizeof(int));
  size_t row_bytes = ((size_t)(N + 15) * sizeof(int) + 63) & ~(size_t)63;
  for (int i = 0; i < N; ++i) {
    grid[i] = (int *)aligned_alloc(64, row_bytes);
    memset(grid[i], 0, row_bytes);
  }
  init_row_bounds();

  find_identity();

  render("identity.png");
  for (int i = 0; i < N; ++i) {
    free(grid[i]);
  }
  free(grid);
  free(row_begin);
  free(row_end);
  return EXIT_SUCCESS;
}
