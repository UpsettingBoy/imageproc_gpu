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

use image::GrayImage;

use crate::{Executor, Feature};

impl Executor {
    pub fn threshold(&self, img: &GrayImage, threshold: u8) -> GrayImage {
        let src = self.alloc_img(
            img,
            Some(
                ocl::flags::MEM_READ_ONLY
                    | ocl::flags::MEM_HOST_WRITE_ONLY
                    | ocl::flags::MEM_COPY_HOST_PTR,
            ),
        );

        let dest = self.alloc_img(
            img,
            Some(
                ocl::flags::MEM_WRITE_ONLY
                    | ocl::flags::MEM_HOST_READ_ONLY
                    | ocl::flags::MEM_COPY_HOST_PTR,
            ),
        );

        let dims = img.dimensions();

        let kernel = ocl::Kernel::builder()
            .program(self.get_program(&Feature::Contrast))
            .name("threshold")
            .queue(self.queue.clone())
            .global_work_size(&dims)
            .arg(&src)
            .arg(&dest)
            .arg(&(threshold as u32))
            .build()
            .expect("threshold kernel could not be loaded!");

        unsafe {
            kernel.enq().expect("Error while enqueueing the kernel!");
        }

        let mut output = image::ImageBuffer::new(dims.0, dims.1);

        dest.read(&mut output)
            .enq()
            .expect("Error while copying device mem to host!");

        output
    }

    pub fn threshold_mut(&self, img: &mut GrayImage, threshold: u8) {
        let output = self.alloc_img(&img, None);

        let kernel = ocl::Kernel::builder()
            .program(self.get_program(&Feature::Contrast))
            .name("threshold_mut")
            .queue(self.queue.clone())
            .global_work_size(&img.dimensions())
            .arg(&output)
            .arg(&(threshold as u32))
            .build()
            .expect("threshold_mut kernel could not be loaded!");

        unsafe {
            kernel.enq().expect("Error while enqueueing the kernel!");
        }

        output
            .read(img)
            .enq()
            .expect("Error while copying device mem to host!");
    }

    pub fn adaptive_threshold(&self, img: &GrayImage, block_radius: u32) -> GrayImage {
        assert!(block_radius > 0);

        let src = self.alloc_img(
            img,
            Some(
                ocl::flags::MEM_READ_ONLY
                    | ocl::flags::MEM_HOST_WRITE_ONLY
                    | ocl::flags::MEM_COPY_HOST_PTR,
            ),
        );

        let dest = self.alloc_img(
            img,
            Some(
                ocl::flags::MEM_WRITE_ONLY
                    | ocl::flags::MEM_HOST_READ_ONLY
                    | ocl::flags::MEM_COPY_HOST_PTR,
            ),
        );

        let dims = img.dimensions();

        let kernel = ocl::Kernel::builder()
            .program(self.get_program(&Feature::Contrast))
            .name("adaptive_threshold")
            .queue(self.queue.clone())
            .global_work_size(&dims)
            .arg(&src)
            .arg(&dest)
            .arg(&(block_radius as i32))
            .build()
            .expect("adaptive_threshold kernel could not be loaded!");

        unsafe {
            kernel.enq().expect("Error while enqueueing the kernel!");
        }

        let mut output = image::ImageBuffer::new(dims.0, dims.1);

        dest.read(&mut output)
            .enq()
            .expect("Error while copying device mem to host!");

        output
    }

    pub fn stretch_contrast(&self, img: &GrayImage, lower: u8, upper: u8) -> GrayImage {
        assert!(upper > lower, "upper must be strictly greater than lower");

        let src = self.alloc_img(
            img,
            Some(
                ocl::flags::MEM_READ_ONLY
                    | ocl::flags::MEM_HOST_WRITE_ONLY
                    | ocl::flags::MEM_COPY_HOST_PTR,
            ),
        );

        let dest = self.alloc_img(
            img,
            Some(
                ocl::flags::MEM_WRITE_ONLY
                    | ocl::flags::MEM_HOST_READ_ONLY
                    | ocl::flags::MEM_COPY_HOST_PTR,
            ),
        );

        let dims = img.dimensions();

        let kernel = ocl::Kernel::builder()
            .program(self.get_program(&Feature::Contrast))
            .name("stretch_contrast")
            .queue(self.queue.clone())
            .global_work_size(&dims)
            .arg(&src)
            .arg(&dest)
            .arg(&(lower as u32))
            .arg(&(upper as u32))
            .build()
            .expect("stretch_contrast kernel could not be loaded!");

        unsafe {
            kernel.enq().expect("Error while enqueueing the kernel!");
        }

        let mut output = image::ImageBuffer::new(dims.0, dims.1);

        dest.read(&mut output)
            .enq()
            .expect("Error while copying device mem to host!");

        output
    }
}

#[cfg(test)]
mod tests {
    use image::Luma;
    use imageproc::assert_pixels_eq;

    use super::*;

    #[test]
    fn adaptive_threshold_constant() {
        let executor = Executor::default();

        let image = GrayImage::from_pixel(3, 3, Luma([100u8]));
        let binary = executor.adaptive_threshold(&image, 1);
        let expected = GrayImage::from_pixel(3, 3, Luma([255u8]));
        assert_pixels_eq!(binary, expected);
    }

    #[test]
    fn adaptive_threshold_one_darker_pixel() {
        let executor = Executor::default();

        for y in 0..3 {
            for x in 0..3 {
                let mut image = GrayImage::from_pixel(3, 3, Luma([200u8]));
                image.put_pixel(x, y, Luma([100u8]));
                let binary = executor.adaptive_threshold(&image, 1);
                // All except the dark pixel have brightness >= their local mean
                let mut expected = GrayImage::from_pixel(3, 3, Luma([255u8]));
                expected.put_pixel(x, y, Luma([0u8]));
                assert_pixels_eq!(binary, expected);
            }
        }
    }

    #[test]
    fn adaptive_threshold_one_lighter_pixel() {
        let executor = Executor::default();

        for y in 0..5 {
            for x in 0..5 {
                let mut image = GrayImage::from_pixel(5, 5, Luma([100u8]));
                image.put_pixel(x, y, Luma([200u8]));

                let binary = executor.adaptive_threshold(&image, 1);

                for yb in 0..5 {
                    for xb in 0..5 {
                        let output_intensity = binary.get_pixel(xb, yb)[0];

                        let is_light_pixel = xb == x && yb == y;

                        let local_mean_includes_light_pixel =
                            (yb as i32 - y as i32).abs() <= 1 && (xb as i32 - x as i32).abs() <= 1;

                        if is_light_pixel {
                            assert_eq!(output_intensity, 255);
                        } else if local_mean_includes_light_pixel {
                            assert_eq!(output_intensity, 0);
                        } else {
                            assert_eq!(output_intensity, 255);
                        }
                    }
                }
            }
        }
    }

    fn constant_image(width: u32, height: u32, intensity: u8) -> GrayImage {
        GrayImage::from_pixel(width, height, Luma([intensity]))
    }

    #[test]
    fn test_threshold_0_image_0() {
        let executor = Executor::default();

        let expected = 0u8;
        let actual = executor.threshold(&constant_image(10, 10, 0), 0);
        assert_pixels_eq!(actual, constant_image(10, 10, expected));
    }

    #[test]
    fn test_threshold_0_image_1() {
        let executor = Executor::default();

        let expected = 255u8;
        let actual = executor.threshold(&constant_image(10, 10, 1), 0);
        assert_pixels_eq!(actual, constant_image(10, 10, expected));
    }

    #[test]
    fn test_threshold_threshold_255_image_255() {
        let executor = Executor::default();

        let expected = 0u8;
        let actual = executor.threshold(&constant_image(10, 10, 255), 255);
        assert_pixels_eq!(actual, constant_image(10, 10, expected));
    }

    #[test]
    fn test_threshold() {
        let executor = Executor::default();

        let original_contents = (0u8..26u8).map(|x| x * 10u8).collect();
        let original = GrayImage::from_raw(26, 1, original_contents).unwrap();

        let expected_contents = vec![0u8; 13].into_iter().chain(vec![255u8; 13]).collect();

        let expected = GrayImage::from_raw(26, 1, expected_contents).unwrap();

        let actual = executor.threshold(&original, 125u8);
        assert_pixels_eq!(expected, actual);
    }
}
