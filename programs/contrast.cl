// Copyright 2021 Jerónimo Sánchez <jeronimosg@hotmail.es>

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

kernel void threshold(read_only image2d_t src, write_only image2d_t dest,
                      uint th) {
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  uint4 pixel = read_imageui(src, pos);

  uint4 new_pixel = (uint4)(0);
  if (pixel.x > th) {
    new_pixel = (uint4)(255);
  }

  write_imageui(dest, pos, new_pixel);
}

kernel void threshold_mut(read_write image2d_t img, uint th) {
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  uint4 pixel = read_imageui(img, pos);

  uint4 new_pixel = (uint4)(0);
  if (pixel.x > th) {
    new_pixel = (uint4)(255);
  }

  write_imageui(img, pos, new_pixel);
}

kernel void adaptive_threshold(read_only image2d_t input,
                               write_only image2d_t output, int radius) {
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  int2 size = (int2)(get_global_size(0), get_global_size(1));

  int y_low = max(0, pos.y - radius);
  int y_high = min(size.y - 1, pos.y + radius);

  int x_low = max(0, pos.x - radius);
  int x_high = min(size.x - 1, pos.x + radius);

  int w = (y_high - y_low + 1) * (x_high - x_low + 1);

  uint4 pixel = read_imageui(input, pos);

  float mean = 0.0;
  for (int x = x_low; x <= x_high; x++) {
    for (int y = y_low; y <= y_high; y++) {
      uint4 curr_pixel = read_imageui(input, (int2)(x, y));
      mean += curr_pixel.x;
    }
  }

  mean = mean / w;

  uint4 new_pixel = (uint4)(0);
  if ((float)pixel.x >= mean) {
    new_pixel = (uint4)(255);
  }

  write_imageui(output, pos, new_pixel);
}

kernel void stretch_contrast(read_only image2d_t input,
                             write_only image2d_t output, uint lower,
                             uint upper) {
  int2 pos = (int2)(get_global_id(0), get_global_id(1));
  uint4 pixel = read_imageui(input, pos);

  uint len = upper - lower;

  uint4 new_pixel = (uint4)(0);
  if (pixel.x >= upper) {
    new_pixel = (uint4)(255);
  } else if (pixel.x <= lower) {
    new_pixel = (uint4)(0);
  } else {
    float scaled = (255 * (pixel.x - lower)) / len;
    new_pixel = (uint4)((uint)scaled);
  }

  write_imageui(output, pos, new_pixel);
}