use std::{
    fs::{File, FileType},
    io::{Read, Write},
};

fn main() {
    let mut compiler = shaderc::Compiler::new().unwrap();
    let args = shaderc::CompileOptions::new().unwrap();

    for entry in glob::glob("./programs/**/*.comp").unwrap() {
        if let Ok(path) = entry {
            let mut src = String::new();
            let mut file = File::open(&path).unwrap();

            let msg = format!("Problem with {}", &path.display());
            assert!(file.read_to_string(&mut src).unwrap() > 0, msg);

            let file_name = path.file_stem().unwrap().to_str().unwrap().to_string();

            let artifact = compiler
                .compile_into_spirv_assembly(
                    &src,
                    shaderc::ShaderKind::Compute,
                    &file_name,
                    "main",
                    Some(&args),
                )
                .unwrap();

            let binary = artifact.as_text();

            let parent = format!(
                "./compiled/{}/",
                path.parent()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string()
            );
            let out_path = format!("{}{}.spirv", &parent, file_name);

            std::fs::create_dir_all(&parent).unwrap();

            // println!("File name: {}\nFinal path: {}", &file_name, &out_path);

            let mut out_file = File::create(&out_path)
                .unwrap_or_else(|_| panic!("Cannot create path: {}", &out_path));
            out_file.write_all(binary.as_bytes()).unwrap();
        }
    }
}
