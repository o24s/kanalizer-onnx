use criterion::{Criterion, criterion_group, criterion_main};
use kanalizer::{ConvertOptions, Kanalizer};
use ndarray::{Array2, Array3};
use ort::{session::Session, value::TensorRef};
use std::{collections::HashMap, hint::black_box};

const DIM: usize = 256;
const MAX_LENGTH: usize = 16;

#[allow(dead_code, non_upper_case_globals)]
mod constants {
    include!("../../infer/crates/kanalizer-rs/src/constants.rs");
}

pub struct KanalizerOrt {
    encoder: Session,
    decoder: Session,
    ascii_map: HashMap<String, i64>,
    kana_map: HashMap<usize, String>,
}

impl KanalizerOrt {
    pub fn new(enc_path: &str, dec_path: &str) -> ort::Result<Self> {
        let encoder = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(enc_path)?;

        let decoder = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(dec_path)?;

        let ascii_map = constants::ASCII_ENTRIES
            .iter()
            .enumerate()
            .map(|(i, &s)| (s.to_string(), i as i64))
            .collect();

        let kana_map = constants::KANAS
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s.to_string()))
            .collect();

        Ok(Self {
            encoder,
            decoder,
            ascii_map,
            kana_map,
        })
    }

    pub fn infer(&mut self, input: &str) -> Result<String, String> {
        if input.is_empty() {
            return Err("Empty input".to_string());
        }
        let mut source = Vec::with_capacity(input.len() + 2);
        source.push(constants::SOS_IDX as i64);
        for c in input.chars() {
            let s = c.to_string();
            if let Some(&idx) = self.ascii_map.get(&s) {
                source.push(idx);
            } else {
                return Err(format!("Invalid character: {}", c));
            }
        }
        source.push(constants::EOS_IDX as i64);

        let src_len = source.len();
        let src_array = Array2::from_shape_vec((1, src_len), source).unwrap();

        let enc_inputs = ort::inputs![
            "src" => TensorRef::from_array_view(src_array.view()).unwrap()
        ];
        let enc_outputs = self.encoder.run(enc_inputs).map_err(|e| e.to_string())?;

        let enc_tuple = enc_outputs["enc_out"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let seq_len = enc_tuple.1.len() / (1 * DIM);

        let enc_out = ndarray::ArrayView3::from_shape((1, seq_len, DIM), enc_tuple.1).unwrap();

        let mut result = vec![constants::SOS_IDX as i64];

        let mut h1 = Array3::<f32>::zeros((1, 1, DIM));
        let mut h2 = Array3::<f32>::zeros((1, 1, DIM));
        let mut dec_input = Array2::<i64>::zeros((1, 1));

        for i in 0..MAX_LENGTH {
            let dec_input_val = *result.last().unwrap();
            dec_input[[0, 0]] = dec_input_val;

            let dec_inputs = ort::inputs![
                "dec_input" => TensorRef::from_array_view(dec_input.view()).unwrap(),
                "enc_out" => TensorRef::from_array_view(enc_out.view()).unwrap(),
                "h1" => TensorRef::from_array_view(h1.view()).unwrap(),
                "h2" => TensorRef::from_array_view(h2.view()).unwrap()
            ];

            let dec_outputs = self.decoder.run(dec_inputs).map_err(|e| e.to_string())?;

            let h1_tuple = dec_outputs["h1_new"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            h1.assign(&ndarray::ArrayView3::from_shape((1, 1, DIM), h1_tuple.1).unwrap());

            let h2_tuple = dec_outputs["h2_new"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            h2.assign(&ndarray::ArrayView3::from_shape((1, 1, DIM), h2_tuple.1).unwrap());

            let logits_tuple = dec_outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| e.to_string())?;
            let logits_slice = logits_tuple.1;

            let mut next_token = 0;
            let mut max_val = f32::MIN;

            for (idx, &val) in logits_slice.iter().enumerate() {
                let idx_i64 = idx as i64;
                // 1ステップ目はEOSを出力させない
                let masked_val = if i == 0 && idx_i64 == constants::EOS_IDX as i64 {
                    f32::MIN
                } else {
                    val
                };

                if masked_val > max_val {
                    max_val = masked_val;
                    next_token = idx_i64;
                }
            }

            result.push(next_token);
            if next_token == constants::EOS_IDX as i64 {
                break;
            }
        }

        let mut output_str = String::new();
        for &token in result.iter().skip(1) {
            if token == constants::EOS_IDX as i64 {
                break;
            }
            if let Some(kana) = self.kana_map.get(&(token as usize)) {
                output_str.push_str(kana);
            }
        }

        Ok(output_str)
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let kanalizer_rs = Kanalizer::new();
    let options = ConvertOptions::default();
    let mut kanalizer_ort = KanalizerOrt::new(
        "../train/kanalizer_encoder.onnx",
        "../train/kanalizer_decoder_step.onnx",
    )
    .unwrap();

    let test_inputs = [
        ("short", "hi"),
        ("medium", "hello"),
        ("long", "international"),
    ];

    let mut group = c.benchmark_group("kanalizer");
    group.measurement_time(std::time::Duration::from_secs(10));

    for (label, input) in test_inputs {
        group.bench_with_input(format!("ndarray/{label}"), input, |b, i| {
            b.iter(|| {
                kanalizer_rs
                    .convert(black_box(i), black_box(&options))
                    .unwrap()
            })
        });
        group.bench_with_input(format!("ort/{label}"), input, |b, i| {
            b.iter(|| kanalizer_ort.infer(black_box(i)).unwrap())
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
