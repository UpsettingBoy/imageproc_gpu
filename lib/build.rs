use spirv_builder::SpirvBuilder;
use std::error::Error;

fn build_shader(path_to_crate: &str) -> Result<(), Box<dyn Error>> {
    SpirvBuilder::new(path_to_crate)
        .spirv_version(1, 0)
        .print_metadata(true)
        .build()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    build_shader("../shaders/contrast")?;
    build_shader("../shaders/geometric_trans")?;
    Ok(())
}
