fn main() {
    wasm_logger::init(wasm_logger::Config::new(log::Level::Trace));
    yew::Renderer::<candle_wasm_moshi_asr::App>::new().render();
}
