#include <errno.h>
#include <inttypes.h>
#include <math.h>
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
  int changed = 0;
  do {
    changed = 0;
    for (int i = 0; i < N; ++i) {
      for (int j = row_begin[i]; j < row_end[i]; ++j) {
        if (grid[i][j] < 4)
          continue;
        changed = 1; // topple it
        grid[i][j] -= 4;
        if (i > 0)
          grid[i - 1][j]++;
        if (j > 0)
          grid[i][j - 1]++;
        if (i < N - 1)
          grid[i + 1][j]++;
        if (j < N - 1)
          grid[i][j + 1]++;
      }
    }
  } while (changed);
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
  for (int i = 0; i < N; ++i) {
    grid[i] = (int *)malloc(N * sizeof(int));
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
