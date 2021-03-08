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

use log::info;
use ocl::Device;
use std::collections::HashMap;

pub mod contrast;
pub mod geometric_trans;

#[derive(PartialEq, Eq, Hash)]
pub(crate) enum Feature {
    Contrast,
    Geometric_Trans,
}

pub struct Executor {
    queue: ocl::Queue,
    programs: HashMap<Feature, ocl::Program>,
}

impl Default for Executor {
    fn default() -> Self {
        let platform = ocl::Platform::list()
            .pop()
            .expect("There are no available platforms!");
        let device = Device::first(platform).expect("There are no devices for this platform!");

        info!(
            "Using {} - {}",
            platform.name().unwrap(),
            device.name().unwrap()
        );

        Executor::new(device)
    }
}

impl Executor {
    pub fn new(device: Device) -> Self {
        let context = ocl::Context::builder()
            .devices(device)
            .build()
            .expect("Could not build the context!");

        let queue = ocl::Queue::new(&context, device, None).expect("Could not create the queue!");
        let mut programs = HashMap::new();

        //Create progams for each feature
        #[cfg(feature = "contrast")]
        {
            let contrast = ocl::Program::builder()
                .devices(device)
                .src_file("programs/contrast.cl")
                .build(&context)
                .expect("Could not build the contrast program!");

            programs.insert(Feature::Contrast, contrast);

            info!("Added contrast feature");
        }

        #[cfg(feature = "geometric_trans")]
        {
            let geometric = ocl::Program::builder()
                .devices(device)
                .src_file("programs/geometric_trans.cl")
                .build(&context)
                .expect("Could not build the geometric transformations program!");

            programs.insert(Feature::Geometric_Trans, geometric);

            info!("Added geometric transformations feature");
        }

        Self { queue, programs }
    }

    pub(crate) fn get_program(&self, f: &Feature) -> &ocl::Program {
        self.programs
            .get(f)
            .expect("This feature is not enabled/initialized!")
    }

    pub fn alloc_img<T, C>(
        &self,
        img: &image::ImageBuffer<T, C>,
        flags: Option<ocl::flags::MemFlags>,
    ) -> ocl::Image<T::Subpixel>
    where
        T: image::Pixel + 'static,
        T::Subpixel: ocl::traits::OclPrm + 'static,
        C: std::ops::Deref<Target = [T::Subpixel]>,
    {
        use ocl::{
            core::MemObjectType,
            enums::{ImageChannelDataType, ImageChannelOrder},
        };

        let dims = img.dimensions();
        let (order, c_type) = match T::COLOR_TYPE {
            image::ColorType::L8 => (
                ImageChannelOrder::Intensity,
                ImageChannelDataType::UnsignedInt8,
            ),
            image::ColorType::La8 => (
                ImageChannelOrder::Luminance,
                ImageChannelDataType::UnsignedInt8,
            ),
            image::ColorType::Rgb8 => (ImageChannelOrder::Rgb, ImageChannelDataType::UnsignedInt8),
            image::ColorType::Rgba8 => {
                (ImageChannelOrder::Rgba, ImageChannelDataType::UnsignedInt8)
            }
            image::ColorType::L16 => (
                ImageChannelOrder::Intensity,
                ImageChannelDataType::UnsignedInt16,
            ),
            image::ColorType::La16 => (
                ImageChannelOrder::Luminance,
                ImageChannelDataType::UnsignedInt16,
            ),
            image::ColorType::Rgb16 => {
                (ImageChannelOrder::Rgb, ImageChannelDataType::UnsignedInt16)
            }
            image::ColorType::Rgba16 => {
                (ImageChannelOrder::Rgba, ImageChannelDataType::UnsignedInt16)
            }
            image::ColorType::Bgr8 => panic!("Channel order BRG is not implemented!"),
            image::ColorType::Bgra8 => {
                (ImageChannelOrder::Bgra, ImageChannelDataType::UnsignedInt8)
            }
            image::ColorType::__NonExhaustive(_) => {
                panic!("This channel order and channel data type combo is not implemented!")
            }
        };

        let flags = match flags {
            Some(f) => f,
            None => ocl::flags::MEM_COPY_HOST_PTR | ocl::flags::MEM_READ_WRITE,
        };

        ocl::Image::<T::Subpixel>::builder()
            .channel_order(order)
            .channel_data_type(c_type)
            .image_type(MemObjectType::Image2d)
            .dims(&dims)
            .flags(flags)
            .copy_host_slice(&img)
            .queue(self.queue.clone())
            .build()
            .expect("Could not allocate image on GPU!")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_default() {
        let _ = Executor::default();
    }
}
