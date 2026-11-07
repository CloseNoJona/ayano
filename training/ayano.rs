/*
Red de Ayano: (768x8hm -> HIDDEN)x2 -> 8, entrenada con datos de Leela en formato binpack.

Las cubetas de entrada dependen de la casilla del rey propio, con el tablero reflejado para
que el rey quede siempre en el flanco de dama, y la cubeta de salida del numero de piezas.
La disposicion debe coincidir con la del motor (src/nnue/network.h).

Variables de entorno: AYANO_NET, AYANO_SB, AYANO_WDL_START, AYANO_WDL_END, AYANO_LR, AYANO_DATA.

Parte del ejemplo examples/progression/3_input_buckets.rs de bullet, de Jamie Whiting, con
licencia MIT: el texto esta en LICENSE.bullet. Como usarlo: README.md de esta carpeta.
*/
use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::sfbinpack::{MoveType, PieceType, SfBinpackLoader, TrainingDataEntry},
    },
};

const HIDDEN: usize = 1024;
const OUTPUT_BUCKETS: usize = 8;
const QA: i16 = 255;
const QB: i16 = 64;
const SCALE: f32 = 400.0;
const LOADER_BUFFER_MB: usize = 2048;
const LOADER_THREADS: usize = 3;

#[rustfmt::skip]
const KING_BUCKETS: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    6, 6, 6, 6,
    7, 7, 7, 7,
    7, 7, 7, 7,
    7, 7, 7, 7,
    7, 7, 7, 7,
];
const INPUT_BUCKETS: usize = get_num_buckets(&KING_BUCKETS);

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

fn keep(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

fn main() {
    let net_id: String = env_or("AYANO_NET", "ayano".to_string());
    let superbatches: usize = env_or("AYANO_SB", 200);
    let wdl_start: f32 = env_or("AYANO_WDL_START", 0.2);
    let wdl_end: f32 = env_or("AYANO_WDL_END", 0.4);
    let initial_lr: f32 = env_or("AYANO_LR", 0.001);
    let final_lr = initial_lr * 0.3f32.powi(5);
    let data: String = env_or("AYANO_DATA", "data/t91".to_string());

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(KING_BUCKETS))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(INPUT_BUCKETS);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(HIDDEN, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(INPUT_BUCKETS);

            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, HIDDEN);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", 2 * HIDDEN, OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer).select(output_buckets)
        });

    // Con el factorizador cada peso final suma dos: asi la suma no se sale de lo cuantizable.
    let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::LinearWDL { start: wdl_start, end: wdl_end },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: 10,
    };

    let settings =
        LocalSettings { threads: 2, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let mut paths: Vec<String> = std::fs::read_dir(&data)
        .expect("no se pudo leer la carpeta de datos")
        .map(|entry| entry.unwrap().path().to_str().unwrap().to_string())
        .filter(|path| path.ends_with(".binpack"))
        .collect();
    paths.sort();
    println!("Datos: {paths:?}");
    let paths: Vec<&str> = paths.iter().map(String::as_str).collect();
    let loader = SfBinpackLoader::new_concat_multiple(&paths, LOADER_BUFFER_MB, LOADER_THREADS, keep);

    trainer.run(&schedule, &settings, &loader);
}
