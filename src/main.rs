use axum::http::header;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use image::Rgba;
use std::fs;
use std::{net::SocketAddr, path::PathBuf};
use tokio::net::TcpListener;
use tower_http::cors::{Cors, CorsLayer};

use axum::{
    extract::{DefaultBodyLimit, Multipart},
    http::StatusCode,
    routing::post,
    Form, Json, Router,
};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam::{self, Sam};
use tower_http::services::ServeDir;

const MODEL_PATH: &str = "model/mobile_sam-tiny-vitt.safetensors";
const IMAGE_TEMP_PATH: &str = "image/temp.png";
const OUTPUT_PATH: &str = "image/output.png";
const MODEL_PATH: &str = "sample.obj";
const THRESHOLD: f32 = 0.;
const PORT: u16 = 3000;

async fn upload(Json(data): Json<Vec<u8>>) -> StatusCode {
    fs::write(IMAGE_TEMP_PATH, &data).unwrap();

    StatusCode::OK
}

#[axum::debug_handler]
async fn segment(Json(data): Json<Vec<(f64, f64, bool)>>) -> Vec<u8> {
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[PathBuf::from(MODEL_PATH)], DType::F32, &device)
            .unwrap()
    };

    let sam = Sam::new_tiny(vb).unwrap();

    let (image, initial_height, initial_width) =
        candle_examples::load_image(IMAGE_TEMP_PATH, Some(sam::IMAGE_SIZE)).unwrap();

    let points = &data;

    let (mask, iou_predictions) = sam.forward(&image, points, false).unwrap();

    let mask = (mask.ge(THRESHOLD).unwrap() * 255.).unwrap();
    let (_, h, w) = mask.dims3().unwrap();
    let mask = mask.expand((3, h, w)).unwrap();

    let mut img =
        image::ImageBuffer::<Rgba<u8>, _>::new(initial_width as u32, initial_height as u32);
    let original_img = image::io::Reader::open(IMAGE_TEMP_PATH)
        .unwrap()
        .decode()
        .unwrap();

    let mask_pixels = mask
        .permute((1, 2, 0))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(w as u32, h as u32, mask_pixels).unwrap();
    let mask_img = image::DynamicImage::from(mask_img).resize_to_fill(
        img.width(),
        img.height(),
        image::imageops::FilterType::CatmullRom,
    );
    for x in 0..img.width() {
        for y in 0..img.height() {
            let mask_p = imageproc::drawing::Canvas::get_pixel(&mask_img, x, y);
            let original_p = imageproc::drawing::Canvas::get_pixel(&original_img, x, y);
            if mask_p[0] > 100 {
                imageproc::drawing::Canvas::draw_pixel(&mut img, x, y, original_p);
            } else {
                imageproc::drawing::Canvas::draw_pixel(&mut img, x, y, image::Rgba([0, 0, 0, 0]));
            }
        }
    }
    // for (x, y, b) in points {
    //     let x = (x * img.width() as f64) as i32;
    //     let y = (y * img.height() as f64) as i32;
    //     let color = if *b {
    //         image::Rgba([255, 0, 0, 200])
    //     } else {
    //         image::Rgba([0, 255, 0, 200])
    //     };
    //     imageproc::drawing::draw_filled_circle_mut(&mut img, (x, y), 3, color);
    // }
    img.save(OUTPUT_PATH).unwrap();

    fs::read(OUTPUT_PATH).unwrap()
}

async fn generate() -> impl IntoResponse {
    fs::read(MODEL_PATH).unwrap()
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .nest_service("/", ServeDir::new("../Frontend"))
        .route("/upload", post(upload))
        .route("/segment", post(segment))
        .route("/generate", get(generate))
        .layer(CorsLayer::permissive())
        .layer(DefaultBodyLimit::max(1024 * 1024 * 1024));
    let listener = TcpListener::bind(SocketAddr::new([0, 0, 0, 0].into(), PORT))
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
