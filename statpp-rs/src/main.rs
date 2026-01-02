#[macro_use]
extern crate serde;
#[macro_use]
extern crate serde_json;

mod convert;
mod entity;
mod train;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
struct Opt {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Parser, Debug)]
enum Command {
    Convert,
    Train,
}

#[tokio::main]
async fn main() -> Result<()> {
    let opt = Opt::parse();

    match opt.command {
        Command::Convert => {
            convert::run().await?;
        }
        Command::Train => {
            train::run().await?;
        }
    }

    Ok(())
}