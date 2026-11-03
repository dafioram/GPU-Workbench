//! egui front-end.

use crate::bench::{self, Config, Msg, Status, Test, TestResult};
use crate::gpu::{self, AdapterSummary, fmt_bytes};
use crate::sensors::{self, History, Sample, Sensor, StaticInfo};
use eframe::egui::{self, Color32, RichText};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The UI itself is drawn with wgpu when the GPU we'll test has Vulkan/DX12,
/// and with plain OpenGL (glow) otherwise, so very old cards can still show
/// the window. Override with WORKBENCH_RENDERER=glow|wgpu.
fn pick_renderer() -> eframe::Renderer {
    match std::env::var("WORKBENCH_RENDERER").as_deref() {
        Ok("glow") | Ok("gl") => return eframe::Renderer::Glow,
        Ok("wgpu") => return eframe::Renderer::Wgpu,
        _ => {}
    }
    let adapters = gpu::enumerate();
    let modern = adapters
        .get(crate::cli::default_adapter(&adapters))
        .is_some_and(|a| !a.is_gl() && a.info.device_type != wgpu::DeviceType::Cpu);
    if modern {
        eframe::Renderer::Wgpu
    } else {
        eframe::Renderer::Glow
    }
}

pub fn run() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: pick_renderer(),
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1320.0, 860.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("GPU Workbench"),
        ..Default::default()
    };
    eframe::run_native(
        "GPU Workbench",
        options,
        Box::new(|cc| {
            // Follows the OS theme; WORKBENCH_THEME=light|dark overrides it.
            match std::env::var("WORKBENCH_THEME").as_deref() {
                Ok("light") => cc.egui_ctx.set_theme(egui::ThemePreference::Light),
                Ok("dark") => cc.egui_ctx.set_theme(egui::ThemePreference::Dark),
                _ => {}
            }
            Ok(Box::new(App::new()))
        }),
    )
}

/// UI log that mirrors every line to the session log file.
#[derive(Default)]
struct LogBuf {
    lines: Vec<(String, String)>, // (HH:MM:SS, text)
}

impl LogBuf {
    fn push(&mut self, s: impl Into<String>) {
        let s = s.into();
        crate::logfile::line(&s);
        let t = chrono::Local::now().format("%H:%M:%S").to_string();
        self.lines.push((t, s));
        const MAX: usize = 5000; // the file keeps everything
        if self.lines.len() > MAX {
            self.lines.drain(..self.lines.len() - MAX);
        }
    }

    fn text(&self) -> String {
        self.lines
            .iter()
            .map(|(t, l)| format!("[{t}] {l}\n"))
            .collect()
    }
}

struct Running {
    rx: Receiver<Msg>,
    stop: Arc<AtomicBool>,
    current: Option<Test>,
    progress: f32,
    /// Index of the run's tab in `App::runs`.
    tab: usize,
}

/// One run's results, shown as a tab. Each run keeps its own, so results from
/// different adapters never mix.
struct RunTab {
    /// Adapter name, e.g. "AMD Radeon RX 6750 XT".
    adapter: String,
    /// Backend, with " · software" for CPU adapters such as WARP.
    backend: String,
    /// Wall-clock start, HH:MM:SS.
    started: String,
    /// Heading for "Copy as Markdown": the adapter, backend and driver.
    gpu_line: String,
    results: Vec<TestResult>,
}

impl RunTab {
    fn title(&self) -> String {
        format!("{} · {}", self.backend, self.started)
    }
}

struct Telemetry {
    history: History,
    stop: Arc<AtomicBool>,
    info: StaticInfo,
    vram_free: Option<u64>,
    diagnostics: Vec<String>,
}

impl Telemetry {
    fn start(adapter: &AdapterSummary) -> Telemetry {
        let sensor = Sensor::for_adapter(&adapter.info);
        let info = sensor.static_info();
        let vram_free = sensor.vram_free();
        let diagnostics = sensor.diagnose();
        let history: History = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        if sensor.available() {
            sensors::spawn_sampler(
                sensor,
                history.clone(),
                Duration::from_millis(500),
                stop.clone(),
            );
        }
        Telemetry {
            history,
            stop,
            info,
            vram_free,
            diagnostics,
        }
    }
}

impl Drop for Telemetry {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

struct App {
    adapters: Vec<AdapterSummary>,
    selected: usize,
    telemetry: Option<Telemetry>,
    cfg: Config,
    enabled: [bool; Test::ALL.len()],
    running: Option<Running>,
    /// One tab per run this session, oldest first.
    runs: Vec<RunTab>,
    /// Index into `runs` of the tab on show.
    active_run: usize,
    stress: Vec<[f64; 2]>,
    log: LogBuf,
    /// Last run-stopping error, shown above the results until dismissed.
    error: Option<String>,
}

impl App {
    fn new() -> App {
        let adapters = gpu::enumerate();
        let selected = crate::cli::default_adapter(&adapters);
        let mut app = App {
            adapters,
            selected,
            telemetry: None,
            cfg: Config::default(),
            enabled: Test::ALL.map(|t| Test::QUICK.contains(&t)),
            running: None,
            runs: Vec::new(),
            active_run: 0,
            stress: Vec::new(),
            log: LogBuf::default(),
            error: None,
        };
        if app.adapters.is_empty() {
            app.log.push("No GPU adapters found. Check GPU drivers.");
        }
        app.select(selected);
        app
    }

    fn select(&mut self, idx: usize) {
        self.selected = idx;
        self.telemetry = self.adapters.get(idx).map(Telemetry::start);
        if let (Some(a), Some(t)) = (self.adapters.get(idx), &self.telemetry) {
            self.log.push(format!(
                "Selected {} — sensors: {}",
                a.label(),
                t.info.source
            ));
            for d in &t.diagnostics {
                self.log.push(format!("    {d}"));
            }
        }
        if let Some(free) = self.telemetry.as_ref().and_then(|t| t.vram_free) {
            self.cfg.vram_mb = (free as f64 * 0.9 / (1024.0 * 1024.0)) as u64;
        }
    }

    fn start(&mut self, tests: Vec<Test>) {
        if self.running.is_some() || tests.is_empty() || self.adapters.is_empty() {
            return;
        }
        self.error = None;
        if let Some(a) = self.adapters.get(self.selected) {
            self.log.push(format!(
                "=== Run on {} | driver: {} | tests: {}",
                a.label(),
                gpu::driver_text(&a.info),
                tests.iter().map(|t| t.key()).collect::<Vec<_>>().join(",")
            ));
            self.log.push(format!("    config: {:?}", self.cfg));
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        if tests.contains(&Test::Stress) {
            self.stress.clear();
        }
        bench::spawn(self.selected, tests, self.cfg.clone(), tx, stop.clone());
        let tab = self.new_run_tab();
        self.running = Some(Running {
            rx,
            stop,
            current: None,
            progress: 0.0,
            tab,
        });
    }

    /// Adds an empty tab for a run on the selected adapter, and shows it.
    fn new_run_tab(&mut self) -> usize {
        let idx = self.runs.len();
        if let Some(a) = self.adapters.get(self.selected) {
            let mut backend = gpu::backend_str(a.info.backend).to_string();
            if a.effective_type() == wgpu::DeviceType::Cpu {
                backend.push_str(" · software");
            }
            self.runs.push(RunTab {
                adapter: a.info.name.clone(),
                backend,
                started: chrono::Local::now().format("%H:%M:%S").to_string(),
                gpu_line: format!(
                    "GPU: {} ({:?}, driver {})",
                    a.info.name,
                    a.info.backend,
                    gpu::driver_text(&a.info)
                ),
                results: Vec::new(),
            });
        }
        self.active_run = idx;
        idx
    }

    fn pump(&mut self) {
        let Some(run) = &mut self.running else { return };
        let tab = run.tab;
        let mut done = false;
        let mut stopped_by_user = false;
        while let Ok(msg) = run.rx.try_recv() {
            match msg {
                Msg::Log(s) => self.log.push(s),
                Msg::Error(s) => {
                    self.log.push(format!("ERROR: {s}"));
                    self.error = Some(s);
                }
                Msg::Started(t) => {
                    run.current = Some(t);
                    run.progress = 0.0;
                    self.log.push(format!("Running: {}", t.name()));
                }
                Msg::Progress(p) => run.progress = p,
                Msg::Result(r) => {
                    self.log.push(if r.unit.is_empty() {
                        format!("  {}: {}", r.status.label(), r.detail)
                    } else {
                        format!(
                            "  {}: {} {} — {}",
                            r.status.label(),
                            bench::num(r.value),
                            r.unit,
                            r.detail
                        )
                    });
                    let snap = self
                        .telemetry
                        .as_ref()
                        .and_then(|t| t.history.lock().ok().and_then(|h| h.last().cloned()));
                    if let Some(s) = snap {
                        let rows = live_rows(&s)
                            .into_iter()
                            .map(|(k, v)| format!("{k} {v}"))
                            .collect::<Vec<_>>();
                        if !rows.is_empty() {
                            self.log.push(format!("    sensors: {}", rows.join(", ")));
                        }
                    }
                    if let Some(runtab) = self.runs.get_mut(tab) {
                        runtab.results.retain(|x| x.test != r.test);
                        runtab.results.push(r);
                        runtab
                            .results
                            .sort_by_key(|x| Test::ALL.iter().position(|t| *t == x.test));
                    }
                }
                Msg::Stress { t, gflops } => self.stress.push([t, gflops]),
                Msg::Done => {
                    done = true;
                    stopped_by_user = run.stop.load(Ordering::Relaxed);
                }
            }
        }
        if done {
            self.running = None;
            self.log.push(if stopped_by_user {
                "Run stopped."
            } else {
                "Run finished."
            });
        }
    }

    /// The results on show as a Markdown table, under the adapter they came from.
    fn results_text(&self) -> String {
        let Some(run) = self.runs.get(self.active_run) else {
            return String::new();
        };
        let mut s = format!("{}\n\n", run.gpu_line);
        s += "| Test | Result | Status | Detail |\n|---|---|---|---|\n";
        for r in &run.results {
            s += &format!(
                "| {} | {} {} | {:?} | {} |\n",
                r.test.name(),
                bench::num(r.value),
                r.unit,
                r.status,
                r.detail
            );
        }
        s
    }

    // ---------------- panels ----------------

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("GPU Workbench");
            ui.separator();
            let busy = self.running.is_some();
            let current = self
                .adapters
                .get(self.selected)
                .map(|a| a.label())
                .unwrap_or_else(|| "no adapter".into());
            let mut pick = None;
            ui.add_enabled_ui(!busy, |ui| {
                egui::ComboBox::from_id_salt("adapter")
                    .selected_text(current)
                    .width(420.0)
                    .show_ui(ui, |ui| {
                        for a in &self.adapters {
                            if ui
                                .selectable_label(a.index == self.selected, a.label())
                                .clicked()
                            {
                                pick = Some(a.index);
                            }
                        }
                    });
            });
            if let Some(i) = pick
                && i != self.selected
            {
                self.select(i);
            }
            ui.separator();
            if busy {
                if icon_button(ui, Icon::Stop, "Stop").clicked()
                    && let Some(r) = &self.running
                {
                    r.stop.store(true, Ordering::Relaxed);
                }
            } else {
                if icon_button(ui, Icon::Play, "Run selected").clicked() {
                    let tests = Test::ALL
                        .into_iter()
                        .zip(self.enabled)
                        .filter(|(_, e)| *e)
                        .map(|(t, _)| t)
                        .collect();
                    self.start(tests);
                }
                if ui.button("Quick suite").clicked() {
                    self.start(Test::QUICK.to_vec());
                }
            }
            if let Some(r) = &self.running {
                let name = r.current.map(|t| t.name()).unwrap_or("starting…");
                progress_bar(ui, r.progress, 240.0);
                ui.label(format!("{:>3.0}%", r.progress * 100.0));
                ui.spinner();
                ui.label(RichText::new(name).strong());
            }
        });
    }

    /// Always-visible running log along the bottom of the window. Follows new
    /// lines automatically; scrolling up pauses that until you scroll back down.
    fn log_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("Log");
            if ui.small_button("Copy").clicked() {
                ui.ctx().copy_text(self.log.text());
            }
            if ui.small_button("Clear").clicked() {
                self.log.lines.clear();
            }
            match crate::logfile::path() {
                Some(p) => {
                    if ui.small_button("Open folder").clicked() {
                        crate::logfile::open_folder();
                    }
                    ui.label(
                        RichText::new(format!("saved to {}", p.display()))
                            .weak()
                            .small(),
                    );
                }
                None => {
                    ui.label(RichText::new("log file could not be created").weak());
                }
            }
        });
        ui.separator();

        let font = egui::FontId::monospace(14.0);
        let row_h = ui.fonts_mut(|f| f.row_height(&font));
        let dark = ui.visuals().dark_mode;
        let (red, green, accent) = if dark {
            (
                Color32::from_rgb(255, 110, 100),
                Color32::from_rgb(110, 210, 130),
                Color32::from_rgb(120, 180, 255),
            )
        } else {
            (
                Color32::from_rgb(190, 30, 30),
                Color32::from_rgb(20, 130, 50),
                Color32::from_rgb(20, 90, 200),
            )
        };
        let warn = if dark {
            Color32::from_rgb(240, 180, 60)
        } else {
            Color32::from_rgb(170, 100, 0)
        };
        let weak = ui.visuals().weak_text_color();
        let normal = ui.visuals().text_color();
        let lines = &self.log.lines;
        egui::ScrollArea::both()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show_rows(ui, row_h, lines.len(), |ui, range| {
                for (t, l) in &lines[range] {
                    let lower = l.to_ascii_lowercase();
                    let color = if lower.starts_with("error")
                        || lower.contains("device lost")
                        || lower.contains("panic")
                        || lower.trim_start().starts_with("fail")
                    {
                        red
                    } else if lower.trim_start().starts_with("warn") {
                        warn
                    } else if lower.trim_start().starts_with("pass") {
                        green
                    } else if l.starts_with("Running:") || l.starts_with("===") {
                        accent
                    } else {
                        normal
                    };
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(t).font(font.clone()).color(weak));
                        ui.add(
                            egui::Label::new(RichText::new(l).font(font.clone()).color(color))
                                .extend(),
                        );
                    });
                }
            });
    }

    fn info_panel(&self, ui: &mut egui::Ui) {
        let Some(a) = self.adapters.get(self.selected) else {
            ui.label("No adapter.");
            return;
        };
        let i = &a.info;
        let l = &a.limits;
        ui.heading("Adapter");
        egui::Grid::new("info")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                let mut row = |k: &str, v: String| {
                    ui.label(RichText::new(k).weak());
                    ui.label(v);
                    ui.end_row();
                };
                row("Name", i.name.clone());
                row("Vendor", format!("{} ({:#06x})", a.vendor_name(), i.vendor));
                row("Device ID", format!("{:#06x}", i.device));
                row("Type", gpu::device_type_str(a.effective_type()).into());
                row("Backend", gpu::backend_str(i.backend).into());
                row(
                    "Compute shaders",
                    if a.has_compute() {
                        format!("yes ({:?})", a.downlevel.shader_model)
                    } else {
                        "NO — OpenGL needs 4.3 for compute".into()
                    },
                );
                row("Driver", gpu::driver_text(i));
                if !i.device_pci_bus_id.is_empty() {
                    row("PCI bus", i.device_pci_bus_id.clone());
                }
                row(
                    "Subgroup size",
                    format!("{}–{}", i.subgroup_min_size, i.subgroup_max_size),
                );
                row("Max buffer", fmt_bytes(l.max_buffer_size));
                row(
                    "Max storage bind",
                    fmt_bytes(l.max_storage_buffer_binding_size),
                );
                row(
                    "Workgroup",
                    format!(
                        "{} threads, {} shared",
                        l.max_compute_invocations_per_workgroup,
                        fmt_bytes(l.max_compute_workgroup_storage_size as u64)
                    ),
                );
                row(
                    "FP16 / timestamps",
                    format!(
                        "{} / {}",
                        yes_no(a.features.contains(wgpu::Features::SHADER_F16)),
                        yes_no(a.features.contains(wgpu::Features::TIMESTAMP_QUERY))
                    ),
                );
            });

        ui.add_space(10.0);
        let Some(t) = &self.telemetry else { return };
        ui.heading("Telemetry");
        ui.label(RichText::new(format!("source: {}", t.info.source)).weak());
        let last = t.history.lock().ok().and_then(|h| h.last().cloned());
        egui::Grid::new("telemetry")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                for (k, v) in &t.info.extra {
                    ui.label(RichText::new(k).weak());
                    ui.label(v);
                    ui.end_row();
                }
                if let Some(s) = &last {
                    for (k, v) in live_rows(s) {
                        ui.label(RichText::new(k).weak());
                        ui.label(RichText::new(v).strong());
                        ui.end_row();
                    }
                }
            });

        ui.add_space(10.0);
        egui::CollapsingHeader::new("All limits").show(ui, |ui| {
            ui.label(RichText::new(format!("{l:#?}")).monospace().small());
        });
        egui::CollapsingHeader::new("All features").show(ui, |ui| {
            for (name, _) in a.features.iter_names() {
                ui.label(RichText::new(name).monospace().small());
            }
        });
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        let busy = self.running.is_some();
        let free_now = self.telemetry.as_ref().and_then(|t| t.vram_free);
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (i, t) in Test::ALL.iter().enumerate() {
                    ui.checkbox(&mut self.enabled[i], t.name())
                        .on_hover_text(t.help());
                }
            });
            ui.add_space(4.0);
            egui::Grid::new("cfg")
                .num_columns(4)
                .spacing([16.0, 6.0])
                .show(ui, |ui| {
                    ui.label("Transfer size").on_hover_text(TRANSFER_HELP);
                    ui.add(
                        egui::DragValue::new(&mut self.cfg.transfer_mb)
                            .range(bench::limits::TRANSFER_MB)
                            .suffix(" MiB"),
                    )
                    .on_hover_text(TRANSFER_HELP);
                    ui.label("Matmul N").on_hover_text(MATMUL_N_HELP);
                    n_picker(ui, "mm_n", &mut self.cfg.matmul_n).on_hover_text(MATMUL_N_HELP);
                    ui.end_row();

                    ui.label("VRAM to test").on_hover_text(VRAM_HELP);
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.cfg.vram_auto, "Auto")
                            .on_hover_text(VRAM_AUTO_HELP);
                        if self.cfg.vram_auto {
                            let hint = match free_now {
                                Some(free) => format!(
                                    "90% of free at test start (now ~{})",
                                    fmt_bytes((free as f64 * 0.9) as u64)
                                ),
                                None => "90% of free at test start".into(),
                            };
                            ui.label(RichText::new(hint).weak())
                                .on_hover_text(VRAM_AUTO_HELP);
                        } else {
                            ui.add(
                                egui::DragValue::new(&mut self.cfg.vram_mb)
                                    .range(bench::limits::VRAM_MB)
                                    .speed(16)
                                    .suffix(" MiB"),
                            )
                            .on_hover_text(VRAM_HELP);
                            let hint = match free_now {
                                Some(free) => format!("{} free", fmt_bytes(free)),
                                None => "free VRAM unknown".into(),
                            };
                            ui.label(RichText::new(hint).weak())
                                .on_hover_text(VRAM_HELP);
                        }
                    });
                    ui.label("VRAM patterns").on_hover_text(PATTERNS_HELP);
                    ui.add(
                        egui::DragValue::new(&mut self.cfg.vram_passes)
                            .range(bench::limits::VRAM_PASSES),
                    )
                    .on_hover_text(PATTERNS_HELP);
                    ui.end_row();

                    ui.label("Stress duration").on_hover_text(STRESS_SECS_HELP);
                    ui.add(
                        egui::DragValue::new(&mut self.cfg.stress_secs)
                            .range(bench::limits::STRESS_SECS)
                            .suffix(" s"),
                    )
                    .on_hover_text(STRESS_SECS_HELP);
                    ui.label("Stress N").on_hover_text(STRESS_N_HELP);
                    n_picker(ui, "st_n", &mut self.cfg.stress_n).on_hover_text(STRESS_N_HELP);
                    ui.end_row();
                });
        });
    }

    /// Results heading, one tab per run (newest first), and the run on show.
    fn results_section(&mut self, ui: &mut egui::Ui) {
        let busy = self.running.is_some();
        let has_results = self
            .runs
            .get(self.active_run)
            .is_some_and(|r| !r.results.is_empty());
        ui.horizontal(|ui| {
            ui.heading("Results");
            if has_results && ui.small_button("Copy as Markdown").clicked() {
                ui.ctx().copy_text(self.results_text());
            }
            let can_clear = !busy && !self.runs.is_empty();
            ui.add_enabled_ui(can_clear, |ui| {
                if ui
                    .small_button("Clear")
                    .on_hover_text("Remove all run tabs")
                    .clicked()
                {
                    self.runs.clear();
                    self.active_run = 0;
                }
            });
        });
        if self.runs.is_empty() {
            self.results_table(ui);
            return;
        }
        let mut picked = None;
        ui.horizontal_wrapped(|ui| {
            for i in (0..self.runs.len()).rev() {
                let on = i == self.active_run;
                if ui.selectable_label(on, self.runs[i].title()).clicked() {
                    picked = Some(i);
                }
            }
        });
        if let Some(i) = picked {
            self.active_run = i;
        }
        ui.add_space(4.0);
        self.results_table(ui);
    }

    fn results_table(&self, ui: &mut egui::Ui) {
        let Some(run) = self.runs.get(self.active_run) else {
            ui.label(RichText::new("No results yet — pick tests and press Run.").weak());
            return;
        };
        ui.label(
            RichText::new(format!(
                "{} · {} · run {}",
                run.adapter, run.backend, run.started
            ))
            .weak(),
        );
        if run.results.is_empty() {
            ui.label(RichText::new("No results in this run yet.").weak());
            return;
        }
        let counts = tally(&run.results);
        if !counts.is_empty() {
            ui.label(RichText::new(counts).weak());
        }
        egui::Grid::new("results")
            .num_columns(4)
            .striped(true)
            .spacing([18.0, 6.0])
            .show(ui, |ui| {
                for h in ["Test", "Result", "Status", "Detail"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for r in &run.results {
                    ui.label(r.test.name()).on_hover_text(r.test.help());
                    if r.unit.is_empty() {
                        ui.label("—");
                    } else {
                        ui.label(
                            RichText::new(format!("{} {}", bench::num(r.value), r.unit))
                                .monospace()
                                .strong(),
                        );
                    }
                    let (txt, col) = match r.status {
                        Status::Pass => ("PASS", Color32::from_rgb(60, 170, 90)),
                        Status::Warn => ("WARN", Color32::from_rgb(220, 150, 20)),
                        Status::Fail => ("FAIL", Color32::from_rgb(220, 70, 60)),
                        Status::Skipped => ("SKIP", Color32::GRAY),
                        Status::Stopped => ("STOP", Color32::from_rgb(150, 150, 160)),
                    };
                    ui.label(RichText::new(txt).color(col).strong());
                    ui.label(&r.detail);
                    ui.end_row();
                }
            });
    }

    fn plots(&self, ui: &mut egui::Ui) {
        let hist: Vec<Sample> = self
            .telemetry
            .as_ref()
            .and_then(|t| t.history.lock().ok().map(|h| h.clone()))
            .unwrap_or_default();
        let link = egui::Id::unique("time_axis");
        let height = 150.0;
        let series = |f: &dyn Fn(&Sample) -> Option<f64>| -> Vec<[f64; 2]> {
            hist.iter().filter_map(|s| f(s).map(|v| [s.t, v])).collect()
        };

        type Chart<'a> = (&'a str, Vec<(&'a str, Vec<[f64; 2]>)>);
        let charts: Vec<Chart> = vec![
            ("Temperature (°C)", vec![("GPU", series(&|s| s.temp_c))]),
            (
                "Power (W)",
                vec![
                    ("draw", series(&|s| s.power_w)),
                    ("limit", series(&|s| s.power_limit_w)),
                ],
            ),
            (
                "Clocks (MHz)",
                vec![
                    ("core", series(&|s| s.core_mhz)),
                    ("memory", series(&|s| s.mem_mhz)),
                ],
            ),
            (
                "Utilization (%)",
                vec![
                    ("GPU", series(&|s| s.util_pct)),
                    ("memory", series(&|s| s.mem_util_pct)),
                    ("fan", series(&|s| s.fan_pct)),
                ],
            ),
            (
                "VRAM used (GiB)",
                vec![(
                    "used",
                    series(&|s| s.vram_used.map(|b| b as f64 / 1073741824.0)),
                )],
            ),
            (
                "Stress throughput (GFLOPS)",
                vec![("matmul", self.stress.clone())],
            ),
        ];

        // Only draw charts that have something in them.
        let charts: Vec<Chart> = charts
            .into_iter()
            .filter(|(_, lines)| lines.iter().any(|(_, d)| !d.is_empty()))
            .collect();
        if charts.is_empty() {
            ui.label(
                RichText::new(
                    "Charts appear here once telemetry or stress data arrives. \
                     Live sensors need NVML (NVIDIA driver) or amdgpu sysfs (Linux AMD).",
                )
                .weak(),
            );
            return;
        }

        let width = (ui.available_width() - 12.0) / 2.0;
        egui::Grid::new("plots")
            .num_columns(2)
            .spacing([12.0, 12.0])
            .show(ui, |ui| {
                for (i, (title, lines)) in charts.into_iter().enumerate() {
                    ui.vertical(|ui| {
                        ui.set_width(width);
                        ui.label(RichText::new(title).strong());
                        Plot::new(title)
                            .height(height)
                            .width(width)
                            .legend(Legend::default())
                            .link_axis(link, [true, false])
                            .include_y(0.0)
                            .allow_scroll(false)
                            .show(ui, |p| {
                                for (name, data) in lines {
                                    if !data.is_empty() {
                                        p.line(Line::new(name, PlotPoints::from(data)));
                                    }
                                }
                            });
                    });
                    if i % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.pump();

        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(4.0);
        });

        egui::Panel::bottom("log")
            .resizable(true)
            .default_size(230.0)
            .min_size(90.0)
            .show(ui, |ui| self.log_panel(ui));

        egui::Panel::left("info")
            .resizable(true)
            .default_size(340.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.info_panel(ui));
            });

        egui::CentralPanel::default_margins().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::CollapsingHeader::new(RichText::new("Tests & settings").heading())
                    .default_open(true)
                    .show(ui, |ui| self.settings(ui));
                ui.add_space(8.0);
                let mut dismiss = false;
                if let Some(err) = &self.error {
                    egui::Frame::group(ui.style())
                        .fill(Color32::from_rgba_unmultiplied(220, 70, 60, 40))
                        .show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                ui.label(RichText::new("Couldn't run:").strong());
                                ui.label(err);
                                dismiss = ui.small_button("✕").clicked();
                            });
                        });
                    ui.add_space(6.0);
                }
                if dismiss {
                    self.error = None;
                }
                self.results_section(ui);
                ui.add_space(12.0);
                egui::CollapsingHeader::new(RichText::new("Live").heading())
                    .default_open(true)
                    .show(ui, |ui| self.plots(ui));
            });
        });

        // Keep plots and progress moving.
        let every = if self.running.is_some() { 100 } else { 500 };
        ui.ctx().request_repaint_after(Duration::from_millis(every));
    }
}

/// "2 pass, 1 skip" for the statuses present in a run; empty when there are none.
fn tally(results: &[TestResult]) -> String {
    [
        ("PASS", "pass"),
        ("WARN", "warn"),
        ("FAIL", "fail"),
        ("SKIP", "skip"),
        ("STOP", "stopped"),
    ]
    .iter()
    .filter_map(|(label, word)| {
        let n = results
            .iter()
            .filter(|r| r.status.label() == *label)
            .count();
        (n > 0).then(|| format!("{n} {word}"))
    })
    .collect::<Vec<_>>()
    .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(status: Status) -> TestResult {
        TestResult {
            test: Test::Upload,
            value: 1.0,
            unit: "GB/s",
            detail: String::new(),
            status,
        }
    }

    #[test]
    fn tally_counts_each_status_present() {
        let r = [
            result(Status::Pass),
            result(Status::Pass),
            result(Status::Skipped),
        ];
        assert_eq!(tally(&r), "2 pass, 1 skip");
        assert_eq!(tally(&[]), "");
    }
}

fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

fn n_picker(ui: &mut egui::Ui, id: &str, n: &mut u32) -> egui::Response {
    egui::ComboBox::from_id_salt(id)
        .selected_text(n.to_string())
        .show_ui(ui, |ui| {
            for v in [512, 1024, 2048, 3072, 4096, 6144, 8192] {
                ui.selectable_value(n, v, v.to_string());
            }
        })
        .response
}

fn live_rows(s: &Sample) -> Vec<(&'static str, String)> {
    let mut v = Vec::new();
    if let Some(t) = s.temp_c {
        v.push(("Temperature", format!("{t:.0} °C")));
    }
    if let Some(p) = s.power_w {
        let lim = s
            .power_limit_w
            .map(|l| format!(" / {l:.0}"))
            .unwrap_or_default();
        v.push(("Power", format!("{p:.0}{lim} W")));
    }
    if let Some(u) = s.util_pct {
        v.push(("GPU util", format!("{u:.0} %")));
    }
    if let (Some(u), Some(t)) = (s.vram_used, s.vram_total) {
        v.push(("VRAM used", format!("{} / {}", fmt_bytes(u), fmt_bytes(t))));
    }
    if let Some(c) = s.core_mhz {
        let m = s.mem_mhz.map(|m| format!(" / {m:.0}")).unwrap_or_default();
        v.push(("Clocks", format!("{c:.0}{m} MHz")));
    }
    if let Some(f) = s.fan_pct {
        v.push(("Fan", format!("{f:.0} %")));
    }
    v
}

const TRANSFER_HELP: &str = "Buffer size for the Upload, Readback, Copy and Compute bandwidth tests. Bigger sizes give steadier numbers because fixed per-copy overhead matters less; 256 MiB or more is a good default.\n\nCapped by the GPU's maximum buffer size (Compute bandwidth also by 'Max storage bind').";

const MATMUL_N_HELP: &str = "Matrix size for the Matmul tests: two N x N matrices are multiplied (2·N³ operations). 2048 is a good default; larger N keeps fast GPUs busier and gives steadier GFLOPS.\n\nIf a size would run long enough to trip the Windows 2-second GPU watchdog, it is reduced automatically and the result says so.";

const VRAM_HELP: &str = "Total video memory the VRAM fill + verify test allocates, in many chunks. Set it close to the card's FREE VRAM, roughly 85-90% of total (about 1700 MiB on a 2 GB card). It has nothing to do with 'Max buffer', which only limits a single chunk.\n\nAsking for more than is free usually doesn't fail: the driver puts the excess in system RAM, and the result is marked WARN with how much really fit in VRAM.";

const VRAM_AUTO_HELP: &str = "Auto measures free VRAM the moment the VRAM test starts and uses 90% of it.\n\nOn Windows it uses the VRAM budget Windows grants this app (DXGI), the most accurate limit. Elsewhere it uses free VRAM from the sensors (NVIDIA NVML, AMD sysfs), and falls back to 1 GiB if neither is available.\n\nUntick to enter a size yourself.";

const PATTERNS_HELP: &str = "How many times the VRAM test fills and verifies all memory, each time with a different pattern (1-6). More passes catch more intermittent or pattern-sensitive faults but take proportionally longer.\n\n3 is a good default; use 6 when hunting a suspected bad card.";

const STRESS_SECS_HELP: &str = "How long the Sustained stress test runs matmul continuously. Throttling usually shows within 1-5 minutes as the card heats up, so use 300 s or more to check cooling, and watch the throughput chart under Live.";

const STRESS_N_HELP: &str = "Matrix size used by the stress test. Larger N means heavier GPU work per batch; 2048 keeps most cards fully loaded. Reduced automatically if a batch would risk the GPU watchdog on a slow card.";

#[derive(Clone, Copy)]
enum Icon {
    Play,
    Stop,
}

/// A button with a painted icon, so it sits centred regardless of how the
/// font draws ▶ / ⏹ (those glyphs ride high in egui's default font).
fn icon_button(ui: &mut egui::Ui, icon: Icon, label: &str) -> egui::Response {
    let pad = ui.spacing().button_padding;
    let galley = egui::WidgetText::from(label).into_galley(
        ui,
        Some(egui::TextWrapMode::Extend),
        f32::INFINITY,
        egui::TextStyle::Button,
    );
    let icon_size = galley.size().y * 0.62;
    let gap = 6.0;
    let size = egui::vec2(
        pad.x * 2.0 + icon_size + gap + galley.size().x,
        (galley.size().y + pad.y * 2.0).max(ui.spacing().interact_size.y),
    );
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, ui.is_enabled(), label));
    if ui.is_rect_visible(rect) {
        let v = ui.style().interact(&resp);
        let p = ui.painter();
        p.rect_filled(rect, v.corner_radius, v.weak_bg_fill);
        p.rect_stroke(rect, v.corner_radius, v.bg_stroke, egui::StrokeKind::Inside);
        let color = v.fg_stroke.color;
        let c = egui::pos2(rect.left() + pad.x + icon_size / 2.0, rect.center().y);
        let h = icon_size / 2.0;
        match icon {
            Icon::Play => {
                // Nudge right so the triangle looks optically centred.
                let pts = vec![
                    egui::pos2(c.x - h * 0.75, c.y - h),
                    egui::pos2(c.x + h, c.y),
                    egui::pos2(c.x - h * 0.75, c.y + h),
                ];
                p.add(egui::Shape::convex_polygon(pts, color, egui::Stroke::NONE));
            }
            Icon::Stop => {
                let r = egui::Rect::from_center_size(c, egui::vec2(h * 1.75, h * 1.75));
                p.rect_filled(r, 1.5, color);
            }
        }
        let text_pos = egui::pos2(
            rect.left() + pad.x + icon_size + gap,
            rect.center().y - galley.size().y / 2.0,
        );
        p.galley(text_pos, galley, color);
    }
    resp
}

/// Progress bar with a clearly outlined track and a saturated fill, readable
/// in both light and dark themes.
fn progress_bar(ui: &mut egui::Ui, frac: f32, width: f32) {
    let h = ui.spacing().interact_size.y * 0.7;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, h), egui::Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let dark = ui.visuals().dark_mode;
    let track = if dark {
        Color32::from_gray(28)
    } else {
        Color32::from_gray(225)
    };
    let border = if dark {
        Color32::from_gray(110)
    } else {
        Color32::from_gray(120)
    };
    let fill = if dark {
        Color32::from_rgb(64, 156, 255)
    } else {
        Color32::from_rgb(20, 100, 220)
    };
    let radius = h / 2.0;
    let p = ui.painter();
    p.rect_filled(rect, radius, track);
    let frac = frac.clamp(0.0, 1.0);
    if frac > 0.0 {
        let mut f = rect;
        f.set_width((rect.width() * frac).max(h)); // keep the rounded end visible
        p.rect_filled(f, radius, fill);
    }
    p.rect_stroke(
        rect,
        radius,
        egui::Stroke::new(1.0, border),
        egui::StrokeKind::Inside,
    );
}
