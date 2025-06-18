import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font as tkfont
from PIL import Image, ImageTk
import json
import subprocess
import threading
import queue
import os
from datetime import datetime
import math
from idlelib.tooltip import Hovertip
import sys

APP_TITLE = "AI Construction Risk Evaluator v2.1"
ICON_PATH = "resources/icon.png"

COLORS = {
    "primary": "#2E5A88",
    "secondary": "#5A8FB8",
    "background": "#F5F5F5",
    "text": "#333333",
    "success": "#4CAF50",
    "warning": "#FFC107",
    "danger": "#D32F2F",
    "border": "#BDBDBD",
    "highlight": "#E1EBF7"
}

TOOLTIPS = {
    "seismic_zone_rating": "<html>Rating of seismic activity risk in the project area<br>1-Very low: Minimal precautions<br>2-Low: Standard precautions<br>3-Moderate: Enhanced structural design<br>4-High: Specialized seismic design<br>5-Very high: Extensive seismic engineering</html>",
    "flood_risk_probability_percent": "Annual probability of flooding in the project area",
    "bim_adoption_level_percent": "Percentage of project using Building Information Modeling",
    "lead_time_rating": "<html>Rating of critical materials lead time:<br>1-Very short (<15 days)<br>2-Short (15-30 days)<br>3-Moderate (30-60 days)<br>4-Long (60-90 days)<br>5-Very long (>90 days)</html>",
    "tariff_rate_percent": "Average tariff rate applied to imported construction materials",
    "supply_chain_disruption_risk_percent": "Probability of supply chain disruptions during the project",
    "socio_political_risk_score": "<html>Assessment of external political and community opposition risks<br>1-Very low: Stable politics<br>2-Low: Mostly stable politics<br>3-Moderate: Some political uncertainty<br>4-High: Political instability<br>5-Very high: Highly unstable</html>",
    "permit_approval_rating": "<html>Rating of permit approval time:<br>1-Very fast (<15 days)<br>2-Fast (15-30 days)<br>3-Moderate (30-60 days)<br>4-Slow (60-90 days)<br>5-Very slow (>90 days)</html>",
    "lien_claims_rating": "<html>Rating of past lien claims:<br>1-None<br>2-1 claim<br>3-2-3 claims<br>4-4-5 claims<br>5-More than 5 claims</html>",
    "contingency_budget_percent": "Percentage of total budget allocated for contingencies",
    "weather_delay_rating": "<html>Rating of weather delay risk:<br>1-Very low (<5 days)<br>2-Low (5-10 days)<br>3-Moderate (10-20 days)<br>4-High (20-30 days)<br>5-Very high (>30 days)</html>",
    "worker_exp_rating": "<html>Rating of worker experience:<br>1-Very experienced (>15 years)<br>2-Experienced (10-15 years)<br>3-Moderate (5-10 years)<br>4-Inexperienced (2-5 years)<br>5-Very inexperienced (<2 years)</html>",
    "turnover_rating": "<html>Rating of worker turnover:<br>1-Very low (<5%)<br>2-Low (5-10%)<br>3-Moderate (10-20%)<br>4-High (20-30%)<br>5-Very high (>30%)</html>",
    "cybersecurity_risk_assessment": "<html>Assessment of cybersecurity vulnerabilities<br>1-Very low: Robust measures<br>2-Low: Good measures<br>3-Moderate: Adequate measures<br>4-High: Minimal measures<br>5-Very high: Poor measures</html>",
    "renewable_energy_contribution_percent": "Percentage of project energy from renewable sources",
    "energy_efficiency_compliance_percent": "Percentage compliance with energy efficiency standards",
    "structural_complexity_rating": "<html>Rating of structural complexity (1=Simple → 5=Extreme)</html>",
    "temporary_structures_rating": "<html>Rating of temporary structures needed (1=Few → 5=Many)</html>",
    "rainfall_flood_risk_percent": "Estimated flood risk % due to rainfall (0-100%)",
    "safety_risk_rating": "<html>Rating of safety incidents (1=None → 5=High)</html>"
}

VALIDATION_RULES = {
    "seismic_zone_rating": [1, 5, True],
    "flood_risk_probability_percent": [0.0, 100.0, False],
    "bim_adoption_level_percent": [0.0, 100.0, False],
    "lead_time_rating": [1, 5, True],
    "tariff_rate_percent": [0.0, 100.0, False],
    "supply_chain_disruption_risk_percent": [0.0, 100.0, False],
    "socio_political_risk_score": [1, 5, True],
    "permit_approval_rating": [1, 5, True],
    "lien_claims_rating": [1, 5, True],
    "contingency_budget_percent": [0.0, 30.0, False],
    "weather_delay_rating": [1, 5, True],
    "worker_exp_rating": [1, 5, True],
    "turnover_rating": [1, 5, True],
    "cybersecurity_risk_assessment": [1, 5, True],
    "renewable_energy_contribution_percent": [0.0, 100.0, False],
    "energy_efficiency_compliance_percent": [0.0, 100.0, False],
    "structural_complexity_rating": [1, 5, True],
    "temporary_structures_rating": [1, 5, True],
    "rainfall_flood_risk_percent": [0.0, 100.0, False],
    "safety_risk_rating": [1, 5, True]
}

FORM_SECTIONS_LAYOUT = [
    {"title": "Environmental & Location Factors", "params": [
        "seismic_zone_rating:Seismic Zone Rating:slider:1:5",
        "flood_risk_probability_percent:Flood Risk (%):slider:0:100",
        "rainfall_flood_risk_percent:Rainfall Flood Risk (%):slider:0:100"]},
    {"title": "Engineering & Technical Challenges", "params": [
        "structural_complexity_rating:Structural Complexity:slider:1:5",
        "temporary_structures_rating:Temporary Structures:slider:1:5",
        "bim_adoption_level_percent:BIM Adoption (%):slider:0:100"]},
    {"title": "Resource & Supply Chain", "params": [
        "lead_time_rating:Lead Time Rating:slider:1:5",
        "tariff_rate_percent:Tariff Rate (%):slider:0:100",
        "supply_chain_disruption_risk_percent:Supply Chain Risk (%):slider:0:100"]},
    {"title": "Administrative & Regulatory", "params": [
        "socio_political_risk_score:Socio-political Risk Score:slider:1:5",
        "permit_approval_rating:Permit Approval Rating:slider:1:5",
        "lien_claims_rating:Lien Claims Rating:slider:1:5"]},
    {"title": "Financial & Schedule", "params": [
        "contingency_budget_percent:Contingency Budget (%):slider:0:30",
        "weather_delay_rating:Weather Delay Rating:slider:1:5"]},
    {"title": "Safety & Workforce", "params": [
        "safety_risk_rating:Safety Risk:slider:1:5",
        "worker_exp_rating:Worker Experience Rating:slider:1:5",
        "turnover_rating:Turnover Rating:slider:1:5"]},
    {"title": "Technology & Sustainability", "params": [
        "cybersecurity_risk_assessment:Cybersecurity Risk:slider:1:5",
        "renewable_energy_contribution_percent:Renewable Energy (%):slider:0:100",
        "energy_efficiency_compliance_percent:Energy Efficiency (%):slider:0:100"]}
]

class ModernSlider(ttk.Frame):
    def __init__(self, master, min_val, max_val, default_val, is_int=True, **kwargs):
        super().__init__(master, **kwargs)
        self.is_int = is_int
        self._variable = tk.IntVar() if is_int else tk.DoubleVar()
        self.value_label = ttk.Label(
            self,
            width=5,
            anchor="e",
            font=('Segoe UI', 10),
            foreground=COLORS["text"]
        )
        self.slider = ttk.Scale(
            self,
            from_=min_val,
            to=max_val,
            variable=self._variable,
            command=self._update_label,
            orient=tk.HORIZONTAL,
            style="Modern.Horizontal.TScale"
        )
        self._variable.set(default_val)
        initial_text = f"{int(default_val)}" if is_int else f"{float(default_val):.1f}"
        self.value_label.config(text=initial_text)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.value_label.pack(side=tk.LEFT)

    def _update_label(self, event=None):
        value = self._variable.get()
        self.value_label.config(
            text=f"{int(value)}" if self.is_int else f"{value:.1f}"
        )

    def get(self):
        return self._variable.get()

    def set(self, value):
        self._variable.set(int(value) if self.is_int else float(value))

class RiskIndicator(tk.Canvas):
    def __init__(self, master, width, height, **kwargs):
        super().__init__(master, width=width, height=height,
                        bg="white", highlightthickness=0, **kwargs)
        self.risk_level = 0
        self.probabilities = [0.7, 0.2, 0.1]
        self.bind("<Configure>", self._draw_gauge)

    def set_risk_data(self, risk_level, probabilities):
        self.risk_level = risk_level
        self.probabilities = probabilities
        self._draw_gauge()

    def _draw_gauge(self, event=None):
        self.delete("all")
        width = self.winfo_width()
        height = self.winfo_height()
        if width < 10 or height < 10:
            return
        bar_height = 20
        bar_y = height * 0.4
        padding = 20
        gauge_width = width - 2 * padding
        for i in range(gauge_width):
            ratio = i / gauge_width
            if ratio < 0.33:
                color = self._blend_colors("#4CAF50", "#FFEB3B", ratio*3)
            elif ratio < 0.66:
                color = self._blend_colors("#FFEB3B", "#FF9800", (ratio-0.33)*3)
            else:
                color = self._blend_colors("#FF9800", "#D32F2F", (ratio-0.66)*3)
            self.create_line(padding + i, bar_y, padding + i, bar_y + bar_height, fill=color)
        for i in range(0, 101, 5):
            x = padding + int(gauge_width * i / 100.0)
            tick_height = 10 if i % 10 == 0 else 5
            self.create_line(x, bar_y - 5, x, bar_y + bar_height + 5, fill="gray")
            if i % 10 == 0:
                self.create_text(x, bar_y + bar_height + 15,
                                text=str(i),
                                fill="black",
                                font=('Segoe UI', 8))
        raw_score = (0.1 * self.probabilities[0] +
                    0.4 * self.probabilities[1] +
                    0.5 * self.probabilities[2]) * 100
        if self.risk_level == 2:
            raw_score = min(100, raw_score * 1.3)
        elif self.risk_level == 1:
            raw_score = min(100, raw_score * 1.1)
        arrow_x = padding + int(gauge_width * raw_score / 100.0)
        arrow_x = max(padding + 5, min(width - padding - 5, arrow_x))
        arrow_points = [
            arrow_x - 7, bar_y - 12,
            arrow_x + 7, bar_y - 12,
            arrow_x, bar_y - 2
        ]
        self.create_polygon(arrow_points, fill=COLORS["primary"], outline="white")

    def _blend_colors(self, color1, color2, ratio):
        r1, g1, b1 = [int(color1[i:i+2], 16) for i in (1, 3, 5)]
        r2, g2, b2 = [int(color2[i:i+2], 16) for i in (1, 3, 5)]
        r = int(r1 * (1 - ratio) + r2 * ratio)
        g = int(g1 * (1 - ratio) + g2 * ratio)
        b = int(b1 * (1 - ratio) + b2 * ratio)
        return f"#{r:02x}{g:02x}{b:02x}"

class LoadingDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self._setup_ui()
        self._center_window()

    def _setup_ui(self):
        self.title("Processing...")
        self.geometry("300x120")
        self.resizable(False, False)
        self.configure(background=COLORS["background"])
        self.transient(self.master)
        self.grab_set()
        ttk.Label(
            self,
            text="Analyzing risks...",
            font=('Segoe UI', 12),
            background=COLORS["background"],
            foreground=COLORS["text"]
        ).pack(pady=(20, 10))
        self.progress = ttk.Progressbar(
            self,
            orient=tk.HORIZONTAL,
            length=200,
            mode='indeterminate',
            style="Modern.Horizontal.TProgressbar"
        )
        self.progress.pack()
        self.progress.start(10)

    def _center_window(self):
        self.update_idletasks()
        parent_x = self.master.winfo_x()
        parent_y = self.master.winfo_y()
        parent_width = self.master.winfo_width()
        parent_height = self.master.winfo_height()
        x = parent_x + (parent_width - 300) // 2
        y = parent_y + (parent_height - 120) // 2
        self.geometry(f"+{x}+{y}")

class RiskAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.loading_dialog = None
        self.input_components = {}
        self.risk_infos_from_py = []
        self.probabilities_from_py = [0.2, 0.6, 0.2]
        self.prediction_from_py = 0
        self.counts_from_py = {"low": 0, "medium": 0, "high": 0}
        self.effective_score_from_py = 0.0
        self.project_info = {
            "project_name": "",
            "city": "",
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        self._configure_styles()
        self._setup_main_window()
        self._create_input_panel()
        self._setup_results_dialog()
        self._perform_initial_analysis()
        try:
            if os.path.exists(ICON_PATH):
                self.iconphoto(True, ImageTk.PhotoImage(file=ICON_PATH))
        except Exception as e:
            print(f"Could not load application icon: {e}")

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=11)
        tkfont.nametofont("TkTextFont").configure(family="Segoe UI", size=11)
        tkfont.nametofont("TkFixedFont").configure(family="Consolas", size=11)
        style.configure(".", 
                       background=COLORS["background"],
                       foreground=COLORS["text"],
                       font=default_font)
        style.configure("Accent.TButton",
                       background=COLORS["primary"],
                       foreground="white",
                       bordercolor=COLORS["primary"],
                       focusthickness=0,
                       font=('Segoe UI', 12, 'bold'),
                       padding=8)
        style.map("Accent.TButton",
                 background=[('active', COLORS["secondary"]), 
                            ('disabled', '#B0BEC5')])
        style.configure("Section.TLabelframe",
                       background=COLORS["background"],
                       bordercolor=COLORS["primary"],
                       borderwidth=2,
                       relief="solid",
                       labelmargins=10)
        style.configure("Section.TLabelframe.Label",
                       font=('Segoe UI', 12, 'bold'),
                       foreground=COLORS["primary"])
        style.configure("Modern.Horizontal.TScale",
                       troughcolor="#E0E0E0",
                       bordercolor=COLORS["primary"],
                       darkcolor=COLORS["secondary"],
                       lightcolor="#FFFFFF",
                       sliderthickness=14,
                       sliderrelief="flat")
        style.configure("Modern.Horizontal.TProgressbar",
                       troughcolor=COLORS["background"],
                       background=COLORS["primary"],
                       bordercolor=COLORS["primary"],
                       lightcolor=COLORS["secondary"],
                       darkcolor=COLORS["primary"])
        style.configure("StatusBar.TFrame",
                       background=COLORS["primary"])
        style.configure("StatusBar.TLabel",
                       background=COLORS["primary"],
                       foreground="white",
                       font=('Segoe UI', 10))
        style.map("Invalid.TEntry",
                 fieldbackground=[("invalid", "#FFEBEE"), ("!invalid", "white")])

    def _setup_main_window(self):
        self.geometry("975x900")
        self.minsize(975, 900)
        self.configure(background=COLORS["background"])
        
        welcome_frame = ttk.Frame(self, padding=(10, 15))
        welcome_frame.pack(fill=tk.X)
        welcome_label = ttk.Label(
            welcome_frame,
            text="Welcome to BuildSafe",
            font=('Segoe UI', 18, 'bold'),
            foreground=COLORS["primary"]
        )
        welcome_label.pack()
        
        project_info_frame = ttk.Frame(self, padding=(20, 10))
        project_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(project_info_frame, text="Project Name:").grid(row=0, column=0, padx=5, sticky="e")
        self.project_name_entry = ttk.Entry(project_info_frame)
        self.project_name_entry.grid(row=0, column=1, padx=5, sticky="ew")
        
        ttk.Label(project_info_frame, text="City:").grid(row=0, column=2, padx=5, sticky="e")
        self.city_entry = ttk.Entry(project_info_frame)
        self.city_entry.grid(row=0, column=3, padx=5, sticky="ew")
        
        ttk.Label(project_info_frame, text="Date:").grid(row=0, column=4, padx=5, sticky="e")
        self.date_entry = ttk.Entry(project_info_frame)
        self.date_entry.insert(0, self.project_info["date"])
        self.date_entry.grid(row=0, column=5, padx=5, sticky="ew")
        
        project_info_frame.columnconfigure(1, weight=1)
        project_info_frame.columnconfigure(3, weight=1)
        project_info_frame.columnconfigure(5, weight=1)
        
        self.main_canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.main_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all")))
        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self.status_bar = ttk.Frame(self, style="StatusBar.TFrame")
        self.status_label = ttk.Label(self.status_bar, text="Ready", style="StatusBar.TLabel")
        self.status_label.pack(side="left", padx=15)
        self.progress_bar = ttk.Progressbar(
            self.status_bar,
            orient=tk.HORIZONTAL,
            length=150,
            mode='indeterminate',
            style="Modern.Horizontal.TProgressbar"
        )
        self.status_bar.pack(side="bottom", fill="x")

    def _create_input_panel(self):
        input_outer_frame = ttk.Frame(self.scrollable_frame, padding=30)
        input_outer_frame.pack(fill=tk.X, expand=True)
        columns_frame = ttk.Frame(input_outer_frame)
        columns_frame.pack(fill=tk.X, expand=True)
        left_column = ttk.Frame(columns_frame, padding=10)
        right_column = ttk.Frame(columns_frame, padding=10)
        columns_frame.grid_columnconfigure(0, weight=1)
        columns_frame.grid_columnconfigure(1, weight=1)
        left_column.grid(row=0, column=0, sticky="nsew")
        right_column.grid(row=0, column=1, sticky="nsew")
        mid_point = len(FORM_SECTIONS_LAYOUT) // 2 + (len(FORM_SECTIONS_LAYOUT) % 2)
        current_column = left_column
        for i, section_data in enumerate(FORM_SECTIONS_LAYOUT):
            if i >= mid_point:
                current_column = right_column
            section_frame = ttk.LabelFrame(
                current_column,
                text=section_data["title"],
                padding=10,
                style="Section.TLabelframe"
            )
            section_frame.pack(fill=tk.X, pady=5, anchor='n')
            for item_str in section_data["params"]:
                parts = item_str.split(':')
                param_name, label_text, input_type = parts[0], parts[1], parts[2]
                row_frame = ttk.Frame(section_frame)
                row_frame.pack(fill=tk.X, pady=2)
                lbl = ttk.Label(row_frame, text=label_text + ":", width=25, anchor="e")
                lbl.pack(side=tk.LEFT)
                component = None
                if input_type == "slider" and len(parts) >= 5:
                    min_val, max_val = int(parts[3]), int(parts[4])
                    default_val = (min_val + max_val) // 2
                    is_int_slider = VALIDATION_RULES.get(param_name, [0, 0, True])[2]
                    component = ModernSlider(
                        row_frame,
                        min_val,
                        max_val,
                        default_val,
                        is_int=is_int_slider
                    )
                    component.pack(side=tk.LEFT, fill=tk.X, expand=True)
                else:
                    component = ttk.Entry(row_frame, width=10)
                    component.insert(0, "0")
                    component.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.input_components[param_name] = component
                tooltip_text = TOOLTIPS.get(param_name, "Project parameter input")
                tooltip_text_plain = tooltip_text.replace("<br>", "\n").replace("<html>", "").replace("</html>", "")
                Hovertip(component, tooltip_text_plain, hover_delay=500)
                rule = VALIDATION_RULES.get(param_name)
                if rule and isinstance(component, ttk.Entry):
                    component.bind("<KeyRelease>",
                                 lambda e, p=param_name, r=rule: self._validate_entry_field(p, r))
        button_frame = ttk.Frame(input_outer_frame, padding=10)
        button_frame.pack(fill=tk.X, pady=10)
        analysis_button = ttk.Button(
            button_frame,
            text="Run Analysis",
            command=self._start_analysis,
            style="Accent.TButton",
            padding=(20, 8)
        )
        analysis_button.pack(side=tk.TOP, anchor="center")

    def _validate_entry_field(self, param_name, rule):
        component = self.input_components[param_name]
        if not isinstance(component, ttk.Entry):
            return True
        min_val, max_val, is_int = rule
        try:
            text_val = component.get()
            if not text_val:
                component.state(['!invalid'])
                self.status_label.config(text="Ready")
                return True
            value = float(text_val)
            if is_int:
                if not value.is_integer():
                    raise ValueError("Not an integer")
                value = int(value)
            if not (min_val <= value <= max_val):
                component.state(['invalid'])
                self.status_label.config(text=f"Invalid {param_name}: range {min_val}-{max_val}")
                return False
            else:
                component.state(['!invalid'])
                self.status_label.config(text="Ready")
                return True
        except ValueError:
            component.state(['invalid'])
            self.status_label.config(text=f"Invalid number for {param_name}")
            return False

    def _validate_all_inputs(self):
        errors = []
        for param_name, component in self.input_components.items():
            rule = VALIDATION_RULES.get(param_name)
            if rule:
                if isinstance(component, ttk.Entry):
                    if not self._validate_entry_field(param_name, rule):
                        errors.append(f"Invalid value for {param_name.replace('_', ' ')}")
        return errors

    def _collect_input_data(self):
        data = {}
        for param_name, component in self.input_components.items():
            if isinstance(component, ModernSlider):
                data[param_name] = component.get()
            elif isinstance(component, ttk.Entry):
                try:
                    data[param_name] = float(component.get())
                except ValueError:
                    data[param_name] = 0.0
            else:
                data[param_name] = 0.0
        return data

    def _show_loading_dialog(self):
        if self.loading_dialog is not None:
            try:
                if self.loading_dialog.winfo_exists():
                    self.loading_dialog.destroy()
            except:
                pass
        self.loading_dialog = LoadingDialog(self)

    def _hide_loading_dialog(self):
        if self.loading_dialog is not None:
            try:
                if self.loading_dialog.winfo_exists():
                    self.loading_dialog.grab_release()
                    self.loading_dialog.destroy()
            except:
                pass
            finally:
                self.loading_dialog = None

    def _start_analysis(self):
        self.project_info["project_name"] = self.project_name_entry.get()
        self.project_info["city"] = self.city_entry.get()
        self.project_info["date"] = self.date_entry.get()
        
        self.project_name_display.config(
            text=f"Project: {self.project_info['project_name']}"
        )
        self.city_date_display.config(
            text=f"Location: {self.project_info['city']} | Date: {self.project_info['date']}"
        )
        
        errors = self._validate_all_inputs()
        if errors:
            messagebox.showerror("Input Errors", "\n".join(errors), parent=self)
            return
        self._show_loading_dialog()
        self.progress_bar.pack(side="right", padx=15)
        self.progress_bar.start(10)
        self.status_label.config(text="Analyzing risks...")
        self.analysis_queue = queue.Queue()
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()
        self.after(100, self._check_analysis_queue)

    def _run_analysis_thread(self):
        try:
            input_data = self._collect_input_data()
            json_input = json.dumps(input_data)
            python_exe = sys.executable
            process = subprocess.Popen(
                [python_exe, "predict_risk.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            stdout, stderr = process.communicate(json_input)
            if process.returncode != 0:
                self.analysis_queue.put({"error": f"Python script error: {stderr}"})
                return
            result = json.loads(stdout)
            self.analysis_queue.put(result)
        except Exception as e:
            self.analysis_queue.put({"error": str(e)})

    def _check_analysis_queue(self):
        try:
            result = self.analysis_queue.get_nowait()
            self._hide_loading_dialog()
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            if "error" in result:
                self.status_label.config(text="Analysis Error")
                messagebox.showerror("Analysis Error", result["error"], parent=self)
            else:
                self.status_label.config(text="Analysis complete")
                self.prediction_from_py = result.get("prediction", 0)
                self.probabilities_from_py = result.get("probabilities", [0.7, 0.2, 0.1])
                self.risk_infos_from_py = result.get("risk_infos", [])
                self.counts_from_py = {
                    "low": result.get("low_count", 0),
                    "medium": result.get("medium_count", 0),
                    "high": result.get("high_count", 0),
                }
                self.effective_score_from_py = result.get("effective_score", 0.0)
                self.analysis_dialog.deiconify()
                self.analysis_dialog.lift()
                self.analysis_dialog.focus_force()
                self._update_ui_with_results()
        except queue.Empty:
            self.after(100, self._check_analysis_queue)

    def _perform_initial_analysis(self):
        self.risk_indicator.set_risk_data(0, [0.7, 0.2, 0.1])
        self._update_risk_distribution_chart()
        self._update_risks_and_recommendations_display()
        self._update_final_recommendation()
        self._update_methodology_content()

    def _update_ui_with_results(self):
        self.risk_indicator.set_risk_data(self.prediction_from_py, self.probabilities_from_py)
        self._update_risk_distribution_chart()
        self._update_risks_and_recommendations_display()
        self._update_final_recommendation()
        self._update_methodology_content()

    def _setup_results_dialog(self):
        self.analysis_dialog = tk.Toplevel(self)
        self.analysis_dialog.title("Analysis Results")
        self.analysis_dialog.geometry("1090x900")
        self.analysis_dialog.protocol("WM_DELETE_WINDOW", self.analysis_dialog.withdraw)
        self.analysis_dialog.withdraw()
        self.analysis_dialog.configure(background=COLORS["background"])
        
        project_display_frame = ttk.Frame(self.analysis_dialog, padding=(20, 10))
        project_display_frame.pack(fill=tk.X)
        
        self.project_name_display = ttk.Label(
            project_display_frame,
            text="Project: ",
            font=('Segoe UI', 12, 'bold')
        )
        self.project_name_display.pack(anchor="w")
        
        self.city_date_display = ttk.Label(
            project_display_frame,
            text="Location: ",
            font=('Segoe UI', 10)
        )
        self.city_date_display.pack(anchor="w")
        
        results_canvas = tk.Canvas(self.analysis_dialog, highlightthickness=0)
        results_scrollbar = ttk.Scrollbar(
            self.analysis_dialog,
            orient="vertical",
            command=results_canvas.yview
        )
        self.results_scrollable_frame = ttk.Frame(results_canvas)
        self.results_scrollable_frame.bind(
            "<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all")))
        results_canvas.create_window((0, 0), window=self.results_scrollable_frame, anchor="nw")
        results_canvas.configure(yscrollcommand=results_scrollbar.set)
        results_canvas.pack(side="left", fill="both", expand=True)
        results_scrollbar.pack(side="right", fill="y")
        
        vis_frame = ttk.LabelFrame(
            self.results_scrollable_frame,
            text="Risk Factor Distribution by Category",
            padding=15,
            style="Section.TLabelframe"
        )
        vis_frame.pack(fill=tk.X, padx=20, pady=10)
        self.risk_dist_canvas = tk.Canvas(
            vis_frame,
            height=250,
            bg='white',
            highlightthickness=0
        )
        self.risk_dist_canvas.pack(pady=5, fill=tk.X, expand=True)
        
        assess_frame = ttk.LabelFrame(
            self.results_scrollable_frame,
            text="Aggregate Risk Assessment",
            padding=15,
            style="Section.TLabelframe"
        )
        assess_frame.pack(fill=tk.X, padx=20, pady=10)
        self.risk_indicator = RiskIndicator(assess_frame, width=0, height=88)
        self.risk_indicator.pack(pady=10, fill=tk.X, expand=True)
        
        self.risks_recs_frame_outer = ttk.LabelFrame(
            self.results_scrollable_frame,
            text="Risks and Recommendations",
            padding=15,
            style="Section.TLabelframe"
        )
        self.risks_recs_frame_outer.pack(fill=tk.X, padx=20, pady=10)
        self.risks_recs_frame_inner = ttk.Frame(self.risks_recs_frame_outer)
        self.risks_recs_frame_inner.pack(fill=tk.BOTH, expand=True)
        
        final_rec_frame = ttk.LabelFrame(
            self.results_scrollable_frame,
            text="Final Recommendations",
            padding=15,
            style="Section.TLabelframe"
        )
        final_rec_frame.pack(fill=tk.X, padx=20, pady=10)
        self.final_recommendation_label = ttk.Label(
            final_rec_frame,
            text="",
            wraplength=900,
            justify=tk.LEFT,
            font=('Segoe UI', 12)
        )
        self.final_recommendation_label.pack(pady=10, fill=tk.X)
        
        methodology_frame = ttk.LabelFrame(
            self.results_scrollable_frame,
            text="Methodology & Calculations",
            padding=15,
            style="Section.TLabelframe"
        )
        methodology_frame.pack(fill=tk.BOTH, padx=20, pady=10, expand=True)
        self.methodology_frame = methodology_frame
        
        export_button_frame = ttk.Frame(self.results_scrollable_frame, padding=10)
        export_button_frame.pack(fill=tk.X, pady=10)
        export_button = ttk.Button(
            export_button_frame,
            text="Export Results",
            command=self._export_results,
            style="Accent.TButton"
        )
        export_button.pack(side=tk.TOP, anchor="center")

    def _update_methodology_content(self):
        if hasattr(self, 'methodology_image_frame'):
            for widget in self.methodology_image_frame.winfo_children():
                widget.destroy()
        else:
            self.methodology_image_frame = ttk.Frame(self.methodology_frame)
            self.methodology_image_frame.pack(fill=tk.BOTH, expand=True)
        try:
            image_path = os.path.join("resources", "methodology.png")
            if os.path.exists(image_path):
                img = Image.open(image_path)
                self.methodology_img_original = img
                canvas = tk.Canvas(self.methodology_image_frame,
                                 bg=COLORS["background"],
                                 highlightthickness=0)
                scroll_y = ttk.Scrollbar(self.methodology_image_frame,
                                       orient="vertical",
                                       command=canvas.yview)
                canvas.configure(yscrollcommand=scroll_y.set)
                scroll_y.pack(side="right", fill="y")
                canvas.pack(side="left", fill="both", expand=True)
                def update_image(event):
                    canvas_width = event.width
                    if canvas_width < 10:
                        return
                    original_width, original_height = self.methodology_img_original.size
                    new_height = int((canvas_width / original_width) * original_height)
                    resized_img = self.methodology_img_original.resize(
                        (canvas_width, new_height),
                        Image.Resampling.LANCZOS
                    )
                    photo = ImageTk.PhotoImage(resized_img)
                    canvas.delete("all")
                    canvas.create_image(0, 0, anchor="nw", image=photo)
                    canvas.config(scrollregion=canvas.bbox("all"))
                    canvas.image = photo
                canvas.bind("<Configure>", update_image)
                canvas.after(100, lambda: canvas.event_generate("<Configure>"))
        except Exception as e:
            ttk.Label(self.methodology_image_frame,
                     text=f"Error loading methodology image: {str(e)}",
                     foreground="red").pack(pady=20)

    def _update_risk_distribution_chart(self):
        canvas = self.risk_dist_canvas
        canvas.delete("all")
        low_count = self.counts_from_py["low"]
        medium_count = self.counts_from_py["medium"]
        high_count = self.counts_from_py["high"]
        categories_data = [
            {"label": f"Low: {low_count}", "count": low_count, "color": "#66BB6A"},
            {"label": f"Medium: {medium_count}", "count": medium_count, "color": "#FFEE58"},
            {"label": f"High: {high_count}", "count": high_count, "color": "#EF5350"}
        ]
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            canvas.after(100, self._update_risk_distribution_chart)
            return
        tower_base_width = 80
        block_depth = 20
        block_unit_height = 8
        max_tower_visual_height = canvas_height * 0.8
        max_drawable_blocks_total = int(max_tower_visual_height / block_unit_height) if block_unit_height > 0 else 0
        x_center_tower = canvas_width / 2
        x_start_pos = x_center_tower - tower_base_width / 2
        current_base_y = canvas_height - 30
        total_blocks_drawn_so_far = 0
        at_least_one_section_drawn = False
        for cat_data in categories_data:
            num_blocks_in_section = cat_data["count"]
            if num_blocks_in_section == 0:
                continue
            at_least_one_section_drawn = True
            remaining_drawable_slots = max_drawable_blocks_total - total_blocks_drawn_so_far
            blocks_to_draw_in_section = min(num_blocks_in_section, remaining_drawable_slots)
            if blocks_to_draw_in_section <= 0:
                continue
            section_height_visual = blocks_to_draw_in_section * block_unit_height
            label_y_pos = current_base_y - (section_height_visual / 2) - (total_blocks_drawn_so_far * block_unit_height)
            canvas.create_text(
                x_start_pos - 10, label_y_pos,
                text=cat_data["label"], 
                anchor="e", 
                fill="black",
                font=('Segoe UI', 9)
            )
            for j in range(blocks_to_draw_in_section):
                block_top_y = current_base_y - ((total_blocks_drawn_so_far + j + 1) * block_unit_height)
                block_base_y = current_base_y - ((total_blocks_drawn_so_far + j) * block_unit_height)
                canvas.create_rectangle(
                    x_start_pos, block_top_y,
                    x_start_pos + tower_base_width, block_base_y,
                    fill=cat_data["color"], 
                    outline="black"
                )
                pts_top = [
                    x_start_pos, block_top_y,
                    x_start_pos + block_depth / 2, block_top_y - block_depth / 2,
                    x_start_pos + tower_base_width + block_depth / 2, block_top_y - block_depth / 2,
                    x_start_pos + tower_base_width, block_top_y
                ]
                r_val, g_val, b_val = canvas.winfo_rgb(cat_data["color"])
                brighter_r = min(255, (r_val // 256) + 60)
                brighter_g = min(255, (g_val // 256) + 60)
                brighter_b = min(255, (b_val // 256) + 60)
                brighter_color = f"#{brighter_r:02x}{brighter_g:02x}{brighter_b:02x}"
                canvas.create_polygon(pts_top, fill=brighter_color, outline="black")
                pts_right = [
                    x_start_pos + tower_base_width, block_top_y,
                    x_start_pos + tower_base_width + block_depth / 2, block_top_y - block_depth / 2,
                    x_start_pos + tower_base_width + block_depth / 2, block_base_y - block_depth / 2,
                    x_start_pos + tower_base_width, block_base_y
                ]
                darker_r = max(0, (r_val // 256) - 40)
                darker_g = max(0, (g_val // 256) - 40)
                darker_b = max(0, (b_val // 256) - 40)
                darker_color = f"#{darker_r:02x}{darker_g:02x}{darker_b:02x}"
                canvas.create_polygon(pts_right, fill=darker_color, outline="black")
            total_blocks_drawn_so_far += blocks_to_draw_in_section
            if total_blocks_drawn_so_far >= max_drawable_blocks_total:
                break
        if not at_least_one_section_drawn:
            canvas.create_text(
                canvas_width / 2, canvas_height / 2,
                text="No risk items to display.",
                anchor="center",
                fill="gray",
                font=('Segoe UI', 10)
            )

    def _update_risks_and_recommendations_display(self):
        for widget in self.risks_recs_frame_inner.winfo_children():
            widget.destroy()
        if not self.risk_infos_from_py:
            ttk.Label(
                self.risks_recs_frame_inner,
                text="No risk factors identified or analysis not run."
            ).pack(pady=10)
            return
        categorized_risks = {"high": [], "medium": [], "low": []}
        for r_info in self.risk_infos_from_py:
            cat = r_info.get("category", "low")
            if cat in categorized_risks:
                categorized_risks[cat].append(r_info)
        category_order = [
            ("high", "High Risk Factors:"), 
            ("medium", "Medium Risk Factors:"), 
            ("low", "Low Risk Factors:")
        ]
        any_risks_displayed_overall = False
        for cat_key, cat_display_title in category_order:
            risks_in_category = categorized_risks[cat_key]
            risks_in_category.sort(key=lambda r: r.get("recommendation", {}).get("priority", 0), reverse=True)
            if not risks_in_category:
                continue
            any_risks_displayed_overall = True
            category_row_frame = ttk.Frame(self.risks_recs_frame_inner)
            category_row_frame.pack(fill=tk.X, anchor="w", pady=(10, 0))
            category_row_frame.columnconfigure(0, weight=0, minsize=180)
            category_row_frame.columnconfigure(1, weight=1)
            title_label = ttk.Label(
                category_row_frame,
                text=cat_display_title,
                font=('Segoe UI', 11, "bold"),
                anchor="nw"
            )
            title_label.grid(row=0, column=0, sticky="nw", padx=(0, 10), pady=(0,5))
            risk_items_stack_frame = ttk.Frame(category_row_frame)
            risk_items_stack_frame.grid(row=0, column=1, sticky="new")
            for item_index, risk_info in enumerate(risks_in_category):
                desc = risk_info.get("description", "N/A")
                rec_data = risk_info.get("recommendation", {})
                suggestion = rec_data.get("suggestion", "No specific suggestion.")
                priority = rec_data.get("priority", 0)
                risk_label_text = f"• {desc}"
                if priority > 0:
                    risk_label_text += f" (Priority: {priority})"
                ttk.Label(
                    risk_items_stack_frame,
                    text=risk_label_text,
                    font=('Segoe UI', 10)
                ).pack(anchor="w", fill=tk.X)
                suggestion_wraplength = 900
                rec_label = ttk.Label(
                    risk_items_stack_frame,
                    text=f"  → {suggestion}",
                    foreground="darkgreen",
                    wraplength=suggestion_wraplength,
                    font=('Segoe UI', 9)
                )
                rec_label.pack(
                    anchor="w",
                    fill=tk.X,
                    padx=(10,0),
                    pady=(0,8 if item_index < len(risks_in_category) -1 else 0)
                )
        if not any_risks_displayed_overall:
            ttk.Label(
                self.risks_recs_frame_inner,
                text="No applicable risk factors identified based on current inputs."
            ).pack(pady=10)

    def _update_final_recommendation(self):
        risk_levels_text = ["LOW", "MODERATE", "HIGH"]
        final_risk_status_text = risk_levels_text[self.prediction_from_py]
        aggregate_score_display = (0.1 * self.probabilities_from_py[0] +
                                0.4 * self.probabilities_from_py[1] +
                                0.5 * self.probabilities_from_py[2]) * 100
        if self.prediction_from_py == 2:
            aggregate_score_display = min(100, aggregate_score_display * 1.3)
        elif self.prediction_from_py == 1:
            aggregate_score_display = min(100, aggregate_score_display * 1.1)
        high_count = self.counts_from_py["high"]
        medium_count = self.counts_from_py["medium"]
        low_count = self.counts_from_py["low"]
        worth_taking = ""
        if self.prediction_from_py == 2:
            worth_taking = "NOT worth taking without significant risk mitigation measures"
        elif self.prediction_from_py == 1:
            worth_taking = "potentially worth taking IF key risk mitigation measures are implemented"
        else:
            worth_taking = "worth taking with standard risk management practices"
        final_rec_text = (
            f"Formula Notice: (0.1×Low + 0.4×Medium + 0.5×High) × 100 used for gauge (with risk level adjustments)\n"
            f"This project is assessed as {final_risk_status_text}\n"
            f"Aggregate Risk Score (gauge): {aggregate_score_display:.1f}%\n"
            f"Effective Score (matrix-based): {self.effective_score_from_py:.2f}\n"
            f"Risk Factors by count: {high_count} High, {medium_count} Medium, {low_count} Low\n"
            f"VERDICT: The project is {worth_taking}."
        )
        self.final_recommendation_label.config(text=final_rec_text)

    def _export_results(self):
        if self.prediction_from_py == 0 and not self.risk_infos_from_py:
            messagebox.showinfo("Export", "Please run an analysis first.", parent=self)
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Analysis Results",
            parent=self
        )
        if not filepath:
            return
        try:
            with open(filepath, "w", encoding='utf-8') as f:
                f.write("CONSTRUCTION PROJECT RISK ANALYSIS (Python Version)\n")
                f.write("========================================\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Project Name: {self.project_info['project_name']}\n")
                f.write(f"Location: {self.project_info['city']}\n")
                f.write(f"Date: {self.project_info['date']}\n\n")
                f.write("INPUT PARAMETERS:\n")
                f.write("==================\n")
                input_data = self._collect_input_data()
                param_labels = {}
                for section in FORM_SECTIONS_LAYOUT:
                    for item_str in section["params"]:
                        parts = item_str.split(':')
                        param_labels[parts[0]] = parts[1]
                for section_data in FORM_SECTIONS_LAYOUT:
                    f.write(f"{section_data['title']}:\n")
                    for item_str in section_data["params"]:
                        param_name = item_str.split(':')[0]
                        label = param_labels.get(param_name, param_name)
                        value = input_data.get(param_name, 0)
                        unit = "%" if "percent" in param_name else ""
                        is_int_param = VALIDATION_RULES.get(param_name, [0,0,True])[2]
                        value_str = f"{int(value)}" if is_int_param else f"{value:.1f}"
                        f.write(f"- {label}: {value_str}{unit}\n")
                    f.write("\n")
                f.write("RISK ASSESSMENT RESULTS:\n")
                f.write("========================\n")
                levels = ["Low", "Medium", "High"]
                f.write(f"Aggregate Risk Level (Overall): {levels[self.prediction_from_py]}\n")
                agg_score_gauge = (0.1 * self.probabilities_from_py[0] +
                                  0.4 * self.probabilities_from_py[1] +
                                  0.5 * self.probabilities_from_py[2]) * 100
                if self.prediction_from_py == 2:
                    agg_score_gauge = min(100, agg_score_gauge * 1.3)
                elif self.prediction_from_py == 1:
                    agg_score_gauge = min(100, agg_score_gauge * 1.1)
                f.write(f"Aggregate Risk Score (Gauge): {agg_score_gauge:.1f}%\n")
                f.write(f"Effective Score (Matrix-based): {self.effective_score_from_py:.2f}\n")
                f.write("RISKS AND RECOMMENDATIONS:\n")
                f.write("==========================\n")
                sorted_risks = sorted(self.risk_infos_from_py, 
                                    key=lambda r: r.get("recommendation", {}).get("priority", 0), 
                                    reverse=True)
                for cat_key, cat_title in [("high", "High Risk Factors"), 
                                         ("medium", "Medium Risk Factors"), 
                                         ("low", "Low Risk Factors")]:
                    f.write(f"{cat_title}:\n")
                    cat_risks = [r for r in sorted_risks if r.get("category") == cat_key]
                    if not cat_risks:
                        f.write("- None identified\n")
                    else:
                        for risk in cat_risks:
                            f.write(f"- {risk.get('description', 'N/A')}\n")
                            rec = risk.get("recommendation", {})
                            sug = rec.get("suggestion", "N/A")
                            pri = rec.get("priority", 0)
                            f.write(f"  Recommendation: {sug} (Priority: {pri})\n")
                    f.write("\n")
                f.write("FINAL RECOMMENDATION:\n")
                f.write("=====================\n")
                final_rec_text_export = self.final_recommendation_label.cget("text")
                notice_to_remove = "Formula Notice: (0.1×Low + 0.4×Medium + 0.5×High) × 100 used for gauge (with risk level adjustments)\n"
                if final_rec_text_export.startswith(notice_to_remove):
                    final_rec_text_export = final_rec_text_export[len(notice_to_remove):]
                f.write(final_rec_text_export + "\n")
                f.write("\nMETHODOLOGY NOTE:\n")
                f.write("=================\n")
                f.write("This analysis uses a hybrid approach combining a machine learning model prediction with rule-based risk factor analysis and an enhanced matrix-based effective score.\n")
                f.write("The Aggregate Risk Score (Gauge) is derived from model probabilities: (0.1×LowProb + 0.4×MediumProb + 0.5×HighProb) × 100 with risk level adjustments.\n")
                f.write("The overall risk level and verdict consider the model's prediction, counts of high/medium/low rule-based factors, the gauge score, and the effective score.\n")
            messagebox.showinfo("Export Successful", f"Analysis results saved to:\n{filepath}", parent=self)
        except IOError as e:
            messagebox.showerror("Error", f"Error saving file: {e}", parent=self)

if __name__ == '__main__':
    if not os.path.exists("resources"):
        os.makedirs("resources")
    if not os.path.exists(ICON_PATH):
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (32, 32), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            d.text((10,10), "R", fill=(255,255,0))
            img.save(ICON_PATH)
            print(f"Created dummy icon at {ICON_PATH}")
        except Exception as e:
            print(f"Could not create dummy icon: {e}")
    app = RiskAnalyzerApp()
    app.mainloop()
