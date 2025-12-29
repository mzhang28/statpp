#[macro_use]
extern crate serde;
#[macro_use]
extern crate serde_json;

mod convert;
mod entity;

use anyhow::Result;
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use clap::Parser;

type Backend = Wgpu;

#[derive(Parser, Debug)]
struct Opt {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Parser, Debug)]
enum Command {
    Convert,
}

#[tokio::main]
async fn main() -> Result<()> {
    let opt = Opt::parse();

    match opt.command {
        Command::Convert => {
            convert::run().await?;
        }
    }

    Ok(())
}
