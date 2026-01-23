#![windows_subsystem = "windows"]

mod wfc;

use egui::{Color32, ColorImage, Context, TextureHandle, TextureOptions};
use crate::wfc::Tile;

fn main() -> eframe::Result {
    eframe::run_native(
        "Wavefunction Collapse",
        eframe::NativeOptions::default(),
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::<App>::default())
        }),
    )
}

struct App {
    texture: Option<TextureHandle>,
    width: usize,
    height: usize,
}

impl Default for App {
    fn default() -> Self {
        Self {
            texture: None,
            width: 4,
            height: 4,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(egui::Slider::new(&mut self.width, 1..=16).text("Width"));
            ui.add(egui::Slider::new(&mut self.height, 1..=16).text("Height"));

            if ui.button("Generate").clicked() {
                let tiles = wfc::generate((self.width, self.height)).unwrap();
                let image = ColorImage::new(
                    [self.width, self.height],
                    tiles
                        .iter()
                        .map(|tile| match tile {
                            Tile::Water => Color32::from_rgb(57, 83, 164),
                            Tile::Beach => Color32::from_rgb(246, 235, 20),
                            Tile::Grass => Color32::from_rgb(105, 189, 69),
                            Tile::Forest => Color32::from_rgb(12, 128, 64),
                        })
                        .collect(),
                );
                self.texture = Some(ctx.load_texture("Map", image, TextureOptions::NEAREST));
            }

            if let Some(texture) = &self.texture {
                ui.add(egui::Image::new(texture).fit_to_original_size(16.0));
            }
        });
    }
}
