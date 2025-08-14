use crate::worker::{Config, Model};
use candle::Device;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

pub struct MoshiASRModel {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub timestamps: bool,
    pub vad: bool,
    pub config: Config,
    pub dev: Device,
}

impl MoshiASRModel {
    pub fn debug_buffers(weights: &[u8], tokenizer: &[u8], mimi: &[u8], config: &[u8]) {
        console_log!("=== Buffer Debug Info ===");

        console_log!(
            "Weights buffer - size: {}, first 10 bytes: {:?}",
            weights.len(),
            &weights[..std::cmp::min(10, weights.len())]
        );

        console_log!(
            "Tokenizer buffer - size: {}, first 10 bytes: {:?}",
            tokenizer.len(),
            &tokenizer[..std::cmp::min(10, tokenizer.len())]
        );

        console_log!(
            "Mimi buffer - size: {}, first 10 bytes: {:?}",
            mimi.len(),
            &mimi[..std::cmp::min(10, mimi.len())]
        );

        console_log!(
            "Config buffer - size: {}, first 10 bytes: {:?}",
            config.len(),
            &config[..std::cmp::min(10, config.len())]
        );

        console_log!("=== End Buffer Debug ===");
    }
}

#[wasm_bindgen]
pub struct MoshiASRDecoder {
    // For now, just store basic info for debugging
}

#[wasm_bindgen]
impl MoshiASRDecoder {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8], tokenizer: &[u8], mimi: &[u8], config: &[u8]) -> Self {
        // Debug the buffers that were passed in
        console_log!("=== Decoder Constructor Called ===");
        console_log!(
            "Weights buffer - size: {}, first 10 bytes: {:?}",
            weights.len(),
            &weights[..std::cmp::min(10, weights.len())]
        );

        console_log!(
            "Tokenizer buffer - size: {}, first 10 bytes: {:?}",
            tokenizer.len(),
            &tokenizer[..std::cmp::min(10, tokenizer.len())]
        );

        console_log!(
            "Mimi buffer - size: {}, first 10 bytes: {:?}",
            mimi.len(),
            &mimi[..std::cmp::min(10, mimi.len())]
        );

        console_log!(
            "Config buffer - size: {}, first 10 bytes: {:?}",
            config.len(),
            &config[..std::cmp::min(10, config.len())]
        );

        console_log!("=== End Decoder Constructor ===");

        Self {}
    }

    pub fn decode(&self, audio: &[u8]) -> String {
        console_log!("=== Decode Method Called ===");
        console_log!("Audio buffer size: {}", audio.len());
        if audio.len() > 0 {
            console_log!(
                "First 10 audio bytes: {:?}",
                &audio[..std::cmp::min(10, audio.len())]
            );
        }
        console_log!("=== End Decode Method ===");

        // Return a placeholder result for now
        r#"[{
            "start": 0.0,
            "duration": 1.0,
            "dr": {
                "tokens": [1, 2, 3],
                "text": "Debug: received buffers successfully in Rust!",
                "avg_logprob": -0.5,
                "no_speech_prob": 0.1,
                "temperature": 0.0,
                "compression_ratio": 1.0
            }
        }]"#
        .to_string()
    }
}
