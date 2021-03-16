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

use futures::executor::block_on;
use image::ImageBuffer;
use log::info;
use std::{borrow::Cow, collections::HashMap, option};

pub mod contrast;
pub mod geometric_trans;

#[derive(PartialEq, Eq, Hash)]
pub(crate) enum Feature {
    Contrast,
    GeometricTrans,
}

pub struct Executor {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Default for Executor {
    fn default() -> Self {
        futures::executor::block_on(Self::new(wgpu::BackendBit::PRIMARY))
    }
}

impl Executor {
    pub async fn new(backend: wgpu::BackendBit) -> Self {
        let adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
        };

        let instance = wgpu::Instance::new(backend);
        let adapter = instance
            .request_adapter(&adapter_options)
            .await
            .expect("Could not find a valid adapter!");

        let features = wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::empty(),
            limits: wgpu::Limits::default(),
        };

        let (device, queue) = adapter
            .request_device(&features, None)
            .await
            .expect("Could not find a valid device!");

        log::info!(
            "Using {:?} - {}",
            adapter.get_info().backend,
            adapter.get_info().name
        );

        let mut flags = wgpu::ShaderFlags::VALIDATION;
        match adapter.get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal => {
                flags |= wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION;
            }
            _ => {}
        }

        #[cfg(feature = "contrast")]
        {
            // let module = wgpu::include_spirv!(env!("compute_shader.spv"));
            // let cs = device.create_shader_module(&src);
        }

        Self { device, queue }
    }

    // pub fn alloc_img_u8<P, C>(
    //     &self,
    //     img: &image::ImageBuffer<P, C>,
    //     usage: Option<wgpu::TextureUsage>,
    // ) where
    //     P: image::Pixel + 'static,
    //     P::Subpixel: u8,
    //     C: std::ops::Deref<Target = [P::Subpixel]>,
    // {
    //     todo!()
    // }

    // pub fn alloc_img_u16<P, C>(
    //     &self,
    //     img: &image::ImageBuffer<P, C>,
    //     usage: Option<wgpu::TextureUsage>,
    // ) where
    //     P: image::Pixel + 'static,
    //     P::Subpixel: u8,
    //     C: std::ops::Deref<Target = [P::Subpixel]>,
    // {
    //     todo!()
    // }

    pub fn alloc_img<P, C>(&self, img: &image::ImageBuffer<P, C>, usage: Option<wgpu::TextureUsage>)
    where
        P: image::Pixel + 'static,
        P::Subpixel: num::Unsigned + num::Integer + 'static,
        C: std::ops::Deref<Target = [P::Subpixel]>,
    {
        let (width, height) = img.dimensions();

        let (format, type_size) = match P::COLOR_TYPE {
            image::ColorType::L8 => (wgpu::TextureFormat::R8Unorm, std::mem::size_of::<u8>()),
            image::ColorType::La8 => (wgpu::TextureFormat::R8Unorm, std::mem::size_of::<u8>()),
            image::ColorType::Rgb8 => (wgpu::TextureFormat::R8Unorm, std::mem::size_of::<u8>()),
            image::ColorType::Rgba8 => (wgpu::TextureFormat::R8Unorm, std::mem::size_of::<u8>()),
            image::ColorType::L16 => (wgpu::TextureFormat::R8Unorm, std::mem::size_of::<u8>()),
            image::ColorType::La16 => (wgpu::TextureFormat::R8Unorm, std::mem::size_of::<u8>()),
            image::ColorType::Rgb16 => (wgpu::TextureFormat::R8Unorm, std::mem::size_of::<u8>()),
            image::ColorType::Rgba16 => (wgpu::TextureFormat::R8Unorm, std::mem::size_of::<u8>()),
            image::ColorType::Bgr8 => (wgpu::TextureFormat::R8Unorm, std::mem::size_of::<u8>()),
            image::ColorType::Bgra8 => (wgpu::TextureFormat::R8Unorm, std::mem::size_of::<u8>()),
            image::ColorType::__NonExhaustive(_) => {
                panic!("This channel order and channel data type combo is not implemented!")
            }
        };

        let usage = if let Some(t) = usage {
            t
        } else {
            wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::COPY_DST
                | wgpu::TextureUsage::SAMPLED
        };

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth: 1,
        };

        let texture_desc = wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            label: None,
        };

        let texture = self.device.create_texture(&texture_desc);

        self.queue.write_texture(
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &vec![0u8; 100],
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: (type_size as u32) * width,
                rows_per_image: height,
            },
            texture_size,
        );

        todo!()
    }
}
