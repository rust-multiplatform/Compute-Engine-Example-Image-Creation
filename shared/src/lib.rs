#![allow(clippy::all)]

use compute_engine::{BaseEngine, ComputeEngine};
use image::{ImageBuffer, Rgba};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, ClearColorImageInfo, CopyImageToBufferInfo},
    format::{ClearColorValue, Format},
    image::{ImageDimensions, StorageImage},
};

#[cfg(test)]
mod tests;

pub fn entrypoint() {
    // Prepare Engine
    let compute_engine = ComputeEngine::new();

    // Print information
    ComputeEngine::print_api_information(compute_engine.get_instance(), log::Level::Info);

    // Prepare Image
    let image = StorageImage::new(
        compute_engine.get_logical_device().get_device(),
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(compute_engine.get_logical_device().get_queue_family_index()),
    )
    .expect("failed to create image");

    // Prepare output buffer
    let output_buffer = CpuAccessibleBuffer::from_iter(
        compute_engine.get_logical_device().get_device(),
        BufferUsage {
            transfer_dst: true,
            ..Default::default()
        },
        false,
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    // Submit Command Buffer for Computation
    compute_engine.compute(&|engine: &ComputeEngine| {
        let mut builder = AutoCommandBufferBuilder::primary(
            engine.get_logical_device().get_device(),
            engine.get_logical_device().get_queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .clear_color_image(ClearColorImageInfo {
                clear_value: ClearColorValue::Float([0.0, 0.0, 1.0, 1.0]),
                ..ClearColorImageInfo::image(image.clone())
            })
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                image.clone(),
                output_buffer.clone(),
            ))
            .unwrap();

        builder.build().unwrap()
    });

    // Assert results
    let buffer_content = output_buffer.read().unwrap();

    log::debug!("Convert Texel to Image");
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();

    log::debug!("Save Image as PNG");
    image.save("image.png").unwrap();
    log::debug!("Successfully saved image");
}
