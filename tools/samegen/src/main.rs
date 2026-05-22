use std::env;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

const SAME_BITRATE: f64 = 520.83;
const SAME_FREQ_ZERO: f64 = 1562.5;
const SAME_FREQ_ONE: f64 = 2083.3;
const SAME_PREAMBLE_BYTE: u8 = 0xAB;
const SAME_PREAMBLE_LEN: usize = 16;

#[derive(Debug)]
struct EncodeArgs {
    message: String,
    out: PathBuf,
    sample_rate: u32,
    amplitude: f64,
    bursts: usize,
    pause_seconds: f64,
}

fn usage() -> &'static str {
    "usage:\n  samegen encode --message <SAME> --out <file.wav> [--sample-rate 48000] [--amplitude 0.35] [--bursts 3] [--pause-seconds 1.0]\n  samegen eom --out <file.wav> [--sample-rate 48000] [--amplitude 0.35] [--bursts 3] [--pause-seconds 1.0]"
}

fn parse_encode_args(mut args: impl Iterator<Item = String>, eom: bool) -> Result<EncodeArgs, String> {
    let mut message: Option<String> = if eom { Some("NNNN".to_string()) } else { None };
    let mut out: Option<PathBuf> = None;
    let mut sample_rate: u32 = 48_000;
    let mut amplitude: f64 = 0.35;
    let mut bursts: usize = 3;
    let mut pause_seconds: f64 = 1.0;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--message" => {
                message = Some(args.next().ok_or("--message requires a value")?);
            }
            "--out" => {
                out = Some(PathBuf::from(args.next().ok_or("--out requires a value")?));
            }
            "--sample-rate" => {
                let value = args.next().ok_or("--sample-rate requires a value")?;
                sample_rate = value.parse().map_err(|_| "--sample-rate must be an integer")?;
            }
            "--amplitude" => {
                let value = args.next().ok_or("--amplitude requires a value")?;
                amplitude = value.parse().map_err(|_| "--amplitude must be a number")?;
            }
            "--bursts" => {
                let value = args.next().ok_or("--bursts requires a value")?;
                bursts = value.parse().map_err(|_| "--bursts must be an integer")?;
            }
            "--pause-seconds" => {
                let value = args.next().ok_or("--pause-seconds requires a value")?;
                pause_seconds = value.parse().map_err(|_| "--pause-seconds must be a number")?;
            }
            "-h" | "--help" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let message = message.ok_or("--message is required")?;
    if message.trim().is_empty() {
        return Err("--message must not be empty".to_string());
    }
    let out = out.ok_or("--out is required")?;
    if sample_rate == 0 {
        return Err("--sample-rate must be > 0".to_string());
    }
    if bursts == 0 {
        return Err("--bursts must be > 0".to_string());
    }
    if !amplitude.is_finite() || amplitude < 0.0 {
        return Err("--amplitude must be a finite non-negative number".to_string());
    }
    if !pause_seconds.is_finite() || pause_seconds < 0.0 {
        return Err("--pause-seconds must be a finite non-negative number".to_string());
    }

    Ok(EncodeArgs {
        message,
        out,
        sample_rate,
        amplitude: amplitude.clamp(0.0, 1.0),
        bursts,
        pause_seconds,
    })
}

fn iter_payload_bits_lsb_first(message: &str) -> Vec<u8> {
    let mut payload = Vec::with_capacity(SAME_PREAMBLE_LEN + message.len());
    payload.extend(std::iter::repeat(SAME_PREAMBLE_BYTE).take(SAME_PREAMBLE_LEN));
    payload.extend(message.bytes().map(|b| b & 0x7f));

    let mut bits = Vec::with_capacity(payload.len() * 8);
    for byte in payload {
        for i in 0..8 {
            bits.push((byte >> i) & 0x01);
        }
    }
    bits
}

fn render_bits_to_pcm(bits: &[u8], sample_rate: u32, amplitude: f64) -> Vec<i16> {
    let sr = sample_rate as f64;
    let peak = (32767.0 * amplitude.clamp(0.0, 1.0)) as f64;
    let samples_per_bit = sr / SAME_BITRATE;
    let mut accumulator = 0.0_f64;
    let mut phase = 0.0_f64;
    let mut samples = Vec::new();

    for bit in bits {
        let freq = if *bit == 0 { SAME_FREQ_ZERO } else { SAME_FREQ_ONE };
        accumulator += samples_per_bit;
        let n = accumulator.floor() as usize;
        accumulator -= n as f64;
        if n == 0 {
            continue;
        }
        let step = 2.0 * PI * freq / sr;
        for _ in 0..n {
            phase += step;
            if phase > 1.0e9 {
                phase %= 2.0 * PI;
            }
            samples.push((phase.sin() * peak) as i16);
        }
    }

    samples
}

fn render_same_bursts(args: &EncodeArgs) -> Vec<i16> {
    let bits = iter_payload_bits_lsb_first(args.message.trim());
    let burst = render_bits_to_pcm(&bits, args.sample_rate, args.amplitude);
    let pause_frames = (args.pause_seconds * args.sample_rate as f64).max(0.0).floor() as usize;

    let total_frames = (burst.len() * args.bursts) + pause_frames.saturating_mul(args.bursts.saturating_sub(1));
    let mut out = Vec::with_capacity(total_frames);
    for i in 0..args.bursts {
        out.extend_from_slice(&burst);
        if i + 1 != args.bursts {
            out.extend(std::iter::repeat(0_i16).take(pause_frames));
        }
    }
    out
}

fn write_le_u16(mut w: impl Write, value: u16) -> io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn write_le_u32(mut w: impl Write, value: u32) -> io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn write_wav_stereo_16(path: &PathBuf, mono: &[i16], sample_rate: u32) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut file = File::create(path)?;
    let channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let bytes_per_sample = (bits_per_sample / 8) as u32;
    let data_bytes = ((mono.len() + 1) as u32) * channels as u32 * bytes_per_sample;
    let byte_rate = sample_rate * channels as u32 * bytes_per_sample;
    let block_align = channels * (bits_per_sample / 8);

    file.write_all(b"RIFF")?;
    write_le_u32(&mut file, 36 + data_bytes)?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    write_le_u32(&mut file, 16)?;
    write_le_u16(&mut file, 1)?; // PCM
    write_le_u16(&mut file, channels)?;
    write_le_u32(&mut file, sample_rate)?;
    write_le_u32(&mut file, byte_rate)?;
    write_le_u16(&mut file, block_align)?;
    write_le_u16(&mut file, bits_per_sample)?;
    file.write_all(b"data")?;
    write_le_u32(&mut file, data_bytes)?;

    // Tiny stereo zero pad, matching the Python encoder's picky-decoder padding.
    write_le_u16(&mut file, 0)?;
    write_le_u16(&mut file, 0)?;
    for sample in mono {
        file.write_all(&sample.to_le_bytes())?;
        file.write_all(&sample.to_le_bytes())?;
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = env::args();
    let _program = args.next();
    let Some(cmd) = args.next() else {
        return Err(usage().to_string());
    };

    let encode_args = match cmd.as_str() {
        "encode" => parse_encode_args(args, false)?,
        "eom" => parse_encode_args(args, true)?,
        "-h" | "--help" => return Err(usage().to_string()),
        other => return Err(format!("unknown command: {other}\n{}", usage())),
    };

    let pcm = render_same_bursts(&encode_args);
    write_wav_stereo_16(&encode_args.out, &pcm, encode_args.sample_rate)
        .map_err(|err| format!("failed to write WAV: {err}"))?;
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_bits_are_lsb_first_with_preamble() {
        let bits = iter_payload_bits_lsb_first("A");
        assert_eq!(&bits[..8], &[1, 1, 0, 1, 0, 1, 0, 1]);
        let a_start = SAME_PREAMBLE_LEN * 8;
        assert_eq!(&bits[a_start..a_start + 8], &[1, 0, 0, 0, 0, 0, 1, 0]);
    }

    #[test]
    fn render_produces_samples() {
        let args = EncodeArgs {
            message: "NNNN".to_string(),
            out: PathBuf::from("/tmp/ignored.wav"),
            sample_rate: 48_000,
            amplitude: 0.35,
            bursts: 1,
            pause_seconds: 0.0,
        };
        let pcm = render_same_bursts(&args);
        assert!(!pcm.is_empty());
    }
}
