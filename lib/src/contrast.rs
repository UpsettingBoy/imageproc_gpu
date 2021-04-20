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
use wgpu::{include_spirv, BindingResource};

use crate::Gpu;

pub struct ContrastExe<'a> {
    gpu: &'a Gpu,
    module: wgpu::ShaderModule,
}

impl<'a> ContrastExe<'a> {
    pub fn threshold(&self, img: &GrayImage, th: u8) -> GrayImage {
        let (input, format) = self.gpu.alloc_img(img, Some(wgpu::TextureUsage::COPY_SRC));
        let (output, _) = self.gpu.alloc_img(img, Some(wgpu::TextureUsage::COPY_DST));

        let input_view = input.create_view(&Default::default());
        let output_view = output.create_view(&Default::default());

        let bind_group_layout =
            self.gpu
                .dev
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("contrast::bind_group"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            count: None,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                dimension: wgpu::TextureViewDimension::D2,
                                format,
                                readonly: true,
                            },
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            count: None,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                dimension: wgpu::TextureViewDimension::D2,
                                format,
                                readonly: false,
                            },
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            count: None,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                dimension: wgpu::TextureViewDimension::D2,
                                format,
                                readonly: true,
                            },
                        },
                    ],
                });

        let pipeline_layout =
            self.gpu
                .dev
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("contrast::pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            self.gpu
                .dev
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("contrast::compute_threshold"),
                    layout: Some(&self.layout),
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &self.module,
                        entry_point: "threshold",
                    },
                });

        let group = self.gpu.dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&input_view),
            }],
        });
    }
}

impl Gpu {
    pub fn contrast_executor(&self) -> ContrastExe {
        let module = self
            .dev
            .create_shader_module(include_spirv!(env!("contrast.spv")));

        ContrastExe {
            gpu: &self,
            module,
            bind: bind_group_layout,
            layout: pipeline_layout,
        }
    }
}
