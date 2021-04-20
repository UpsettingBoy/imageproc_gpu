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

use image::ImageBuffer;
use log::info;
use wgpu::{include_spirv, BindGroupLayoutDescriptor, Device, Queue, TextureUsage};

pub mod contrast;

pub struct Gpu {
    dev: Device,
    que: Queue,
}

impl Gpu {
    pub fn new(
        backends: Option<wgpu::BackendBit>,
        power_preference: wgpu::PowerPreference,
    ) -> Self {
        futures::executor::block_on(Self::new_async(backends, power_preference))
    }

    pub async fn new_async(
        backends: Option<wgpu::BackendBit>,
        power_preference: wgpu::PowerPreference,
    ) -> Self {
        let back = backends.unwrap_or(wgpu::BackendBit::PRIMARY);

        let instance = wgpu::Instance::new(back);
        info!("Using {:?} as backend(s)", &back);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface: None,
            })
            .await
            .unwrap();

        let (dev, que) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    shader_validation: true,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        Self { dev, que }
    }

    pub fn alloc_img<P, C>(
        &self,
        img: &image::ImageBuffer<P, C>,
        usage: Option<wgpu::TextureUsage>,
    ) -> (wgpu::Texture, wgpu::TextureFormat)
    where
        P: image::Pixel<Subpixel = u8> + 'static,
        C: std::ops::Deref<Target = [u8]>,
    {
        let dimension = wgpu::TextureDimension::D2;

        let img_dims = img.dimensions();
        let size = wgpu::Extent3d {
            width: img_dims.0,
            height: img_dims.1,
            depth: 1,
        };

        let format = match P::COLOR_TYPE {
            image::ColorType::L8 => wgpu::TextureFormat::R8Uint,
            image::ColorType::La8 => wgpu::TextureFormat::Rg8Uint,
            image::ColorType::Rgb8 => wgpu::TextureFormat::Rgba8Uint,
            image::ColorType::Rgba8 => wgpu::TextureFormat::Rgba8Uint,
            image::ColorType::Bgr8 => wgpu::TextureFormat::Bgra8Unorm,
            image::ColorType::Bgra8 => wgpu::TextureFormat::Bgra8Unorm,
            // image::ColorType::L16 => wgpu::TextureFormat::R16Uint,
            // image::ColorType::La16 => wgpu::TextureFormat::Rg16Uint,
            // image::ColorType::Rgb16 => wgpu::TextureFormat::Rgba16Uint,
            // image::ColorType::Rgba16 => wgpu::TextureFormat::Rgba16Uint,
            _ => panic!("Format is not supported! Only u8 containers!"),
        };

        let usage = usage.unwrap_or_else(wgpu::TextureUsage::all);

        let desc = wgpu::TextureDescriptor {
            label: None,
            sample_count: 1,
            mip_level_count: 1,
            dimension,
            format,
            size,
            usage,
        };

        let tex = self.dev.create_texture(&desc);

        self.que.write_texture(
            wgpu::TextureCopyView {
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                texture: &tex,
            },
            img,
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: 4 * img_dims.0,
                rows_per_image: img_dims.1,
            },
            size,
        );

        (tex, format)
    }
}
