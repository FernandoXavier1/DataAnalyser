# Se precisar: !pip install kagglehub pandas numpy matplotlib scipy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import kagglehub
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ----------------------
# SciPy (fallback leve)
# ----------------------
try:
    from scipy import stats as st
except Exception:
    import math
    def _norm_cdf(x, loc=0.0, scale=1.0):
        if scale <= 0 or math.isnan(x) or math.isnan(loc) or math.isnan(scale):
            return float("nan")
        z = (x - loc) / (scale * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))
    class _Norm:
        @staticmethod
        def cdf(x, loc=0.0, scale=1.0):
            return _norm_cdf(x, loc, scale)
    class _Stats:
        norm = _Norm()
    st = _Stats()

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================================================
# 1) LOADERS
# ==========================================================
def load_dataset_from_input(user_input: str) -> pd.DataFrame:
    dataset_path = None

    # 1) Arquivo .csv?
    if os.path.isfile(user_input) and user_input.endswith(".csv"):
        dataset_path = user_input

    # 2) Diretório? procura primeiro .csv
    elif os.path.isdir(user_input):
        for root_dir, _, files in os.walk(user_input):
            for file in files:
                if file.endswith(".csv"):
                    dataset_path = os.path.join(root_dir, file)
                    break
            if dataset_path:
                break

    # 3) Tenta baixar do Kaggle Hub
    else:
        download_dir = kagglehub.dataset_download(user_input)
        for root_dir, _, files in os.walk(download_dir):
            for file in files:
                if file.endswith(".csv"):
                    dataset_path = os.path.join(root_dir, file)
                    break
            if dataset_path:
                break

    if not dataset_path:
        raise FileNotFoundError(f"Nenhum arquivo .csv válido para: {user_input}")

    return pd.read_csv(dataset_path)

def get_dataset_path_from_user() -> str | None:
    result_holder = [None]
    root = tk.Tk()
    root.title("Selecionar Fonte do Dataset")
    root.geometry("500x180")

    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill="both", expand=True)

    label = ttk.Label(
        main_frame,
        text="Insira o ID do Kaggle Hub (ex: user/dataset)\nou procure um arquivo/pasta local:"
    )
    label.pack(pady=(0, 10), anchor="w")

    entry = ttk.Entry(main_frame, width=60)
    entry.pack(fill="x", expand=True)

    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill="x", pady=10)

    def browse_file():
        filepath = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Todos", "*.*")])
        if filepath:
            entry.delete(0, tk.END)
            entry.insert(0, filepath)

    def browse_directory():
        dirpath = filedialog.askdirectory()
        if dirpath:
            entry.delete(0, tk.END)
            entry.insert(0, dirpath)

    ttk.Button(button_frame, text="Procurar Arquivo", command=browse_file).pack(side="left", fill="x", expand=True, padx=5)
    ttk.Button(button_frame, text="Procurar Pasta", command=browse_directory).pack(side="left", fill="x", expand=True, padx=5)

    def submit():
        user_input = entry.get().strip()
        if not user_input:
            messagebox.showwarning("Entrada Inválida", "Forneça um caminho ou ID.")
            return
        result_holder[0] = user_input
        root.destroy()

    ttk.Button(main_frame, text="Carregar Dataset", command=submit).pack(pady=10)
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    root.mainloop()
    return result_holder[0]

# ==========================================================
# 2) HELP / NOTES
# ==========================================================
HELP_FIELDS = [
    ("--- ESTATÍSTICA ---", "MÉTRICAS GERAIS DE ANÁLISE"),
    ("Correlação ($r$ - Pearson)", "Mede a força e a direção da relação linear (-1 a +1)."),
    ("r próximo a 1 / -1", "Relação forte; próximo de 0 é fraca."),
    ("Média ($\mu$)", "Valor central dos dados."),
    ("Desvio Padrão ($\sigma$)", "Dispersão em relação à média."),
    ("Mediana (Q2 / 50%)", "Divide o conjunto de dados ao meio."),
    ("Quartis (Q1, Q3)", "25% e 75% dos dados."),
    ("Assimetria (Skewness)", "Inclinação da distribuição; ~0 é simétrica."),
    ("Histograma", "Frequência dos valores."),
    ("Distribuição Normal", "Curva de sino; base para probs."),
    ("P(X < x) / P(X > x)", "Probabilidades sob Normal usando μ e σ."),
]

def show_help_window():
    top = tk.Toplevel(root)
    top.title("Ajuda — Dicionário de Dados (PT-BR)")
    top.geometry("860x520")

    cols = ("col", "desc")
    tree = ttk.Treeview(top, columns=cols, show="headings")
    tree.heading("col", text="Nome da Coluna / Sigla")
    tree.heading("desc", text="Descrição (PT-BR)")
    tree.column("col", width=260, anchor="w")
    tree.column("desc", width=560, anchor="w")

    vsb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(top, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
    vsb.grid(row=0, column=1, sticky="ns", pady=8)
    hsb.grid(row=1, column=0, sticky="ew", padx=8)

    for col_name, desc_pt in HELP_FIELDS:
        tree.insert("", "end", values=(col_name, desc_pt))

    top.rowconfigure(0, weight=1)
    top.columnconfigure(0, weight=1)

class NotesWindow:
    def __init__(self, master):
        self.top = tk.Toplevel(master)
        self.top.title("Bloco de notas")
        self.top.geometry("720x520")
        self.path = None
        self.dirty = False

        bar = ttk.Frame(self.top); bar.pack(fill="x", padx=8, pady=6)
        tk.Button(bar, text="Salvar .txt", command=self.save_notes).pack(side="left")
        tk.Button(bar, text="Salvar como...", command=self.save_notes_as).pack(side="left", padx=(8,0))

        self.text = tk.Text(self.top, wrap="word")
        self.text.pack(fill="both", expand=True, padx=8, pady=(0,8))
        self.text.bind("<<Modified>>", self._on_modified)
        self.top.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_modified(self, _evt=None):
        self.dirty = True
        self.text.edit_modified(0)

    def _save_to(self, path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text.get("1.0", "end-1c"))
            self.path = path
            self.dirty = False
            messagebox.showinfo("Salvo", f"Notas salvas em:\n{path}")
        except Exception as e:
            messagebox.showerror("Erro ao salvar", str(e))

    def save_notes(self):
        if self.path:
            self._save_to(self.path)
        else:
            self.save_notes_as()

    def save_notes_as(self):
        path = filedialog.asksaveasfilename(
            title="Salvar notas como",
            defaultextension=".txt",
            filetypes=[("Text files","*.txt"), ("All files","*.*")]
        )
        if path:
            self._save_to(path)

    def _on_close(self):
        if self.dirty:
            ans = messagebox.askyesnocancel("Notas não salvas", "Deseja salvar antes de fechar?")
            if ans is None:
                return
            if ans:
                self.save_notes()
        self.top.destroy()

notes_win = None
def open_notes():
    global notes_win
    if notes_win is None or not notes_win.top.winfo_exists():
        notes_win = NotesWindow(root)
    else:
        notes_win.top.lift()

# ==========================================================
# 3) HELPERS
# ==========================================================
def numeric_df(dataframe):
    return dataframe.select_dtypes(include=[np.number])

def stats_text_numeric(series: pd.Series) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "Sem dados numéricos válidos."
    d = s.describe(percentiles=[0.25, 0.5, 0.75])
    skew = s.skew()
    if pd.isna(skew):
        skew_label = "N/A"
    elif abs(skew) < 0.1:
        skew_label = "simétrica"
    elif skew > 0:
        skew_label = "assimétrica à direita (positiva)"
    else:
        skew_label = "assimétrica à esquerda (negativa)"
    lines = [
        f"count: {int(d['count'])}",
        f"missing: {int(series.isna().sum())}",
        f"mean (μ): {d['mean']:.6g}",
        f"std (σ): {d['std']:.6g}",
        f"min: {d['min']:.6g}",
        f"Q1 (25%): {d['25%']:.6g}",
        f"median (50%): {d['50%']:.6g}",
        f"Q3 (75%): {d['75%']:.6g}",
        f"max: {d['max']:.6g}",
        f"assimetria (skew): {skew:.6g} ({skew_label})",
    ]
    return "\n".join(lines)

def stats_text_categorical(series: pd.Series) -> str:
    s = series.astype("string")
    cnt = s.notna().sum()
    miss = s.isna().sum()
    uniq = s.nunique(dropna=True)
    vc = s.value_counts(dropna=True)
    if not vc.empty:
        top_val = str(vc.index[0]); top_freq = int(vc.iloc[0])
    else:
        top_val, top_freq = "-", 0
    return "\n".join([f"count: {cnt}", f"missing: {miss}", f"unique: {uniq}", f"top: {top_val}", f"freq: {top_freq}"])

def clear_frame(frame):
    for w in frame.winfo_children():
        w.destroy()

# ==========================================================
# 4) MAIN UI — construída DEPOIS do df
# ==========================================================
def build_ui(df: pd.DataFrame):
    global root, cbx_x, cbx_y, cbx_desc, notebook
    global frm_desc_plot, txt_stats
    global tab_corr, tab_list, tab_xy, tab_desc
    # NOTA: NÃO declarar _last_xy_figure/_last_xy_title como global aqui!

    root = tk.Tk()
    root.title("Análise — Estatísticas e Correlações")

    # Janela proporcional e compacta
    _ORIG_W, _ORIG_H = 1000, 780
    _ORIG_RATIO = _ORIG_W / _ORIG_H
    target_h = 700
    target_w = int(_ORIG_RATIO * target_h)
    root.geometry(f"{target_w}x{target_h}")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=(6, 8)) 

    button_container = ttk.Frame(root)
    button_container.place(relx=1.0, x=-8, y=6, anchor="ne")
    tk.Button(button_container, text="?", width=2, command=show_help_window).pack(side="right", padx=(6,0))
    tk.Button(button_container, text="✎", width=2, command=open_notes).pack(side="right")

    # Abas
    tab_corr   = ttk.Frame(notebook)    # 1) Matriz de correlação
    tab_list   = ttk.Frame(notebook)    # 2) Maiores correlações
    tab_xy     = ttk.Frame(notebook)    # 3) Correlação X · Y
    tab_desc   = ttk.Frame(notebook)    # 4) Estatística e Probabilidade

    notebook.add(tab_corr, text="Matriz de correlação")
    notebook.add(tab_list, text="Maiores correlações")
    notebook.add(tab_xy,   text="Correlação X · Y")
    notebook.add(tab_desc, text="Estatística e Probabilidade")

    # =========================
    # 1) Matriz de Correlação
    # =========================
    frm_corr_top = ttk.Frame(tab_corr); frm_corr_top.pack(fill="x", pady=(8,2))
    ttk.Label(frm_corr_top, text="Zoom:").pack(side="left", padx=(6,4))
    corr_scale = tk.DoubleVar(value=1.0)

    def set_corr_scale(val):
        try:
            v = float(val)
        except:
            return
        corr_scale.set(max(0.5, min(3.0, v)))
        draw_corr_matrix_on_scroll()

    tk.Button(frm_corr_top, text="−", width=3, command=lambda: set_corr_scale(corr_scale.get() - 0.25)).pack(side="left", padx=2)
    tk.Button(frm_corr_top, text="+", width=3, command=lambda: set_corr_scale(corr_scale.get() + 0.25)).pack(side="left", padx=2)
    tk.Button(frm_corr_top, text="Reset", command=lambda: set_corr_scale(1.0)).pack(side="left", padx=6)

    frm_corr_scroll = ttk.Frame(tab_corr); frm_corr_scroll.pack(fill="both", expand=True, padx=8, pady=8)
    corr_canvas = tk.Canvas(frm_corr_scroll)
    hbar = ttk.Scrollbar(frm_corr_scroll, orient="horizontal", command=corr_canvas.xview)
    vbar = ttk.Scrollbar(frm_corr_scroll, orient="vertical", command=corr_canvas.yview)
    corr_canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    hbar.pack(side="bottom", fill="x")
    vbar.pack(side="right", fill="y")
    corr_canvas.pack(side="left", fill="both", expand=True)

    corr_inner = ttk.Frame(corr_canvas)
    _ = corr_canvas.create_window((0,0), window=corr_inner, anchor="nw")
    def on_corr_configure(_evt=None):
        corr_canvas.configure(scrollregion=corr_canvas.bbox("all"))
    corr_inner.bind("<Configure>", on_corr_configure)

    def draw_corr_matrix_on_scroll():
        clear_frame(corr_inner)
        num = numeric_df(df)
        fig_w = 7.5 * corr_scale.get()
        fig_h = 5.5 * corr_scale.get()
        fig = Figure(figsize=(fig_w, fig_h), dpi=100)
        ax = fig.add_subplot(111)
        if num.shape[1] < 2:
            ax.text(0.5, 0.5, "Precisa de pelo menos 2 colunas numéricas.", ha="center", va="center")
        else:
            corr = num.corr(numeric_only=True)
            im = ax.imshow(corr.values, vmin=-1, vmax=1)
            ax.set_title("Matriz de Correlação (Pearson) — Geral")
            ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            fig.colorbar(im, ax=ax, shrink=0.8)
        canvas = FigureCanvasTkAgg(fig, master=corr_inner)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # =========================
    # 2) Maiores correlações
    # =========================
    frm_list_top = ttk.Frame(tab_list); frm_list_top.pack(fill="x", pady=(8,0))
    ttk.Label(frm_list_top, text="Limiar |r|:").pack(side="left", padx=(8,6))
    thr_var = tk.DoubleVar(value=0.50)
    tk.Spinbox(frm_list_top, from_=0.00, to=0.99, increment=0.01, textvariable=thr_var, width=6).pack(side="left", padx=(0,8))

    btn_list = tk.Button(frm_list_top, text="Gerar lista")
    btn_list.pack(side="left", padx=8)

    frm_list_table = ttk.Frame(tab_list); frm_list_table.pack(fill="both", expand=True, padx=8, pady=8)
    tree_corr = None

    def generate_strong_list():
        nonlocal tree_corr
        clear_frame(frm_list_table)
        try:
            thr = float(thr_var.get())
        except ValueError:
            messagebox.showerror("Erro", "Informe um limiar numérico válido.")
            return
        if thr < 0 or thr >= 1:
            messagebox.showerror("Erro", "Use um limiar entre 0.00 e 0.99.")
            return

        num = numeric_df(df)
        if num.shape[1] < 2:
            ttk.Label(frm_list_table, text="Precisa de pelo menos 2 colunas numéricas.").pack(pady=8)
            return

        corr = num.corr(numeric_only=True)
        cols = corr.columns.tolist()

        records = []
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                r = corr.iloc[i, j]
                if pd.notna(r) and abs(r) >= thr:
                    records.append((cols[i], cols[j], float(r), abs(float(r))))
        records.sort(key=lambda t: t[3], reverse=True)

        cols_tree = ("var1","var2","r","abs_r")
        tree = ttk.Treeview(frm_list_table, columns=cols_tree, show="headings")
        for cid, label, w, anc in [
            ("var1","Variável 1",300,"w"),
            ("var2","Variável 2",300,"w"),
            ("r","r",100,"center"),
            ("abs_r","|r|",100,"center")
        ]:
            tree.heading(cid, text=label); tree.column(cid, width=w, anchor=anc)

        vsb = ttk.Scrollbar(frm_list_table, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frm_list_table, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        frm_list_table.rowconfigure(0, weight=1)
        frm_list_table.columnconfigure(0, weight=1)

        if not records:
            tree.insert("", "end", values=("—", "—", "—", "—"))
        else:
            for v1, v2, r, ar in records:
                tree.insert("", "end", values=(v1, v2, f"{r:.3f}", f"{ar:.3f}"))

        tree_corr = tree

        def on_double_click(_evt=None):
            sel = tree.focus()
            if not sel:
                return
            vals = tree.item(sel, "values")
            if len(vals) < 2:
                return
            v1, v2 = vals[0], vals[1]
            try:
                cbx_x.set(v1)
                cbx_y.set(v2)
            except Exception:
                return
            notebook.select(tab_xy)
            update_xy_plot()

        tree.bind("<Double-1>", on_double_click)

    btn_list.configure(command=generate_strong_list)

    # =========================
    # 3) Correlação X·Y
    # =========================
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    frm_xy_top = ttk.Frame(tab_xy); frm_xy_top.pack(fill="x", pady=(8,0))
    ttk.Label(frm_xy_top, text="X:").pack(side="left", padx=(8,6))
    cbx_x = ttk.Combobox(frm_xy_top, values=num_cols, state="readonly", width=30); cbx_x.pack(side="left", padx=(0,10))
    if num_cols: cbx_x.current(0)

    ttk.Label(frm_xy_top, text="Y:").pack(side="left", padx=(4,6))
    cbx_y = ttk.Combobox(frm_xy_top, values=num_cols, state="readonly", width=30); cbx_y.pack(side="left", padx=(0,10))
    if len(num_cols) > 1: cbx_y.current(1)
    elif num_cols: cbx_y.current(0)

    # >>> Agora estas variáveis são do escopo de build_ui (enclosing scope)
    _last_xy_figure = None
    _last_xy_title = None

    def save_xy_plot():
        nonlocal _last_xy_figure, _last_xy_title
        if _last_xy_figure is None:
            messagebox.showinfo("Salvar gráfico", "Gere um gráfico primeiro.")
            return
        default_name = (_last_xy_title or "xy_plot").replace("  |  ", "_").replace(" ", "_") + ".png"
        path = filedialog.asksaveasfilename(
            title="Salvar gráfico como",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG","*.png"), ("SVG","*.svg"), ("PDF","*.pdf"), ("Todos","*.*")]
        )
        if not path:
            return
        try:
            _last_xy_figure.savefig(path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Sucesso", f"Gráfico salvo em:\n{path}")
        except Exception as e:
            messagebox.showerror("Erro ao salvar", str(e))

    tk.Button(frm_xy_top, text="Salvar gráfico (PNG)", command=save_xy_plot).pack(side="right", padx=8)

    frm_xy_plot = ttk.Frame(tab_xy); frm_xy_plot.pack(fill="both", expand=True, padx=8, pady=8)

    def draw_scatter_on(frame, xcol, ycol):
        nonlocal _last_xy_figure, _last_xy_title
        clear_frame(frame)
        fig = Figure(figsize=(7.0, 4.8), dpi=100)
        ax = fig.add_subplot(111)
        title = ""
        if xcol not in df.columns or ycol not in df.columns:
            ax.text(0.5, 0.5, "Colunas inválidas.", ha="center", va="center")
        elif not pd.api.types.is_numeric_dtype(df[xcol]) or not pd.api.types.is_numeric_dtype(df[ycol]):
            ax.text(0.5, 0.5, "Ambas as colunas devem ser numéricas.", ha="center", va="center")
        else:
            sub = df[[xcol, ycol]].dropna()
            if sub.empty:
                ax.text(0.5, 0.5, "Sem dados após remover ausentes.", ha="center", va="center")
            else:
                xv = sub[xcol].to_numpy(); yv = sub[ycol].to_numpy()
                r = sub[xcol].corr(sub[ycol])
                m, b = np.polyfit(xv, yv, 1)
                ax.scatter(xv, yv, alpha=0.7)
                x_line = np.linspace(xv.min(), xv.max(), 100)
                ax.plot(x_line, m*x_line + b)
                title = f"{xcol} vs {ycol}  |  r = {r:.3f}"
                ax.set_title(title)
                ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        _last_xy_figure = fig
        _last_xy_title = title or f"{xcol}_vs_{ycol}"

    def update_xy_plot(_evt=None):
        x = cbx_x.get(); y = cbx_y.get()
        if not x or not y:
            return
        draw_scatter_on(frm_xy_plot, x, y)

    cbx_x.bind("<<ComboboxSelected>>", update_xy_plot)
    cbx_y.bind("<<ComboboxSelected>>", update_xy_plot)

    # =========================
    # 4) Estatística e Probabilidade
    # =========================
    def stats_text(series: pd.Series) -> str:
        return stats_text_numeric(series) if pd.api.types.is_numeric_dtype(series) else stats_text_categorical(series)

    frm_desc_top = ttk.Frame(tab_desc); frm_desc_top.pack(fill="x", pady=(8,2))
    ttk.Label(frm_desc_top, text="Selecione a variável:").pack(side="left", padx=(6,6))
    all_cols = df.columns.tolist()
    cbx_desc = ttk.Combobox(frm_desc_top, values=all_cols, state="readonly", width=42); cbx_desc.pack(side="left", padx=(0,12))
    if all_cols: cbx_desc.current(0)

    pane_desc = ttk.PanedWindow(tab_desc, orient=tk.VERTICAL); pane_desc.pack(fill="both", expand=True, padx=8, pady=8)
    frm_desc_text = ttk.Frame(pane_desc, height=150); pane_desc.add(frm_desc_text, weight=1)
    txt_stats = tk.Text(frm_desc_text, height=11, wrap="word"); txt_stats.pack(fill="both", expand=True); txt_stats.config(state="disabled")
    frm_desc_plot = ttk.Frame(pane_desc, height=250); pane_desc.add(frm_desc_plot, weight=3)
    frm_desc_prob = ttk.LabelFrame(pane_desc, text="Calculadora de Probabilidade (Dist. Normal)"); pane_desc.add(frm_desc_prob, weight=1)

    current_series_mean = None
    current_series_std = None

    frm_calc_grid = ttk.Frame(frm_desc_prob); frm_calc_grid.pack(padx=10, pady=10, fill="x")
    entry_less_than = ttk.Entry(frm_calc_grid, width=10)
    entry_greater_than = ttk.Entry(frm_calc_grid, width=10)
    entry_between_A = ttk.Entry(frm_calc_grid, width=10)
    entry_between_B = ttk.Entry(frm_calc_grid, width=10)

    ttk.Label(frm_calc_grid, text="P(X < ").grid(row=0, column=0, sticky="e", pady=2)
    entry_less_than.grid(row=0, column=1, sticky="w", pady=2)
    ttk.Label(frm_calc_grid, text=" ) = ").grid(row=0, column=2, sticky="w", pady=2)

    ttk.Label(frm_calc_grid, text="P(X > ").grid(row=1, column=0, sticky="e", pady=2)
    entry_greater_than.grid(row=1, column=1, sticky="w", pady=2)
    ttk.Label(frm_calc_grid, text=" ) = ").grid(row=1, column=2, sticky="w", pady=2)

    ttk.Label(frm_calc_grid, text="P( ").grid(row=2, column=0, sticky="e", pady=2)
    entry_between_A.grid(row=2, column=1, sticky="w", pady=2)
    ttk.Label(frm_calc_grid, text=" < X < ").grid(row=2, column=2, sticky="w", pady=2)
    entry_between_B.grid(row=2, column=3, sticky="w", pady=2)
    ttk.Label(frm_calc_grid, text=" ) = ").grid(row=2, column=4, sticky="w", pady=2)

    lbl_result_prob = ttk.Label(frm_calc_grid, text="---", font=("Segoe UI", 9, "bold"), width=35)
    lbl_result_prob.grid(row=0, column=6, rowspan=3, sticky="nsew", padx=10)
    btn_calculate_prob = tk.Button(frm_calc_grid, text="Calcular"); btn_calculate_prob.grid(row=0, column=5, rowspan=3, sticky="ns", padx=(20, 10))

    calculator_widgets = [entry_less_than, entry_greater_than, entry_between_A, entry_between_B, btn_calculate_prob]

    def set_prob_calc_state(enabled: bool):
        state = "normal" if enabled else "disabled"
        for widget in calculator_widgets:
            widget.config(state=state)
        if not enabled:
            lbl_result_prob.config(text="--- (Variável não-numérica) ---")
        else:
            lbl_result_prob.config(text="---")

    def calculate_normal_prob():
        nonlocal current_series_mean, current_series_std
        if current_series_mean is None or current_series_std is None or current_series_std == 0:
            lbl_result_prob.config(text="Erro: Média/Desvio Padrão inválidos.")
            return
        mean = current_series_mean
        std = current_series_std
        val_lt = entry_less_than.get()
        val_gt = entry_greater_than.get()
        val_a = entry_between_A.get()
        val_b = entry_between_B.get()
        try:
            if val_lt:
                a = float(val_lt)
                prob = st.norm.cdf(a, loc=mean, scale=std)
                result_text = f"P(X < {a}) = {prob:.4f} ({prob*100:.2f}%)"
            elif val_gt:
                b = float(val_gt)
                prob = 1 - st.norm.cdf(b, loc=mean, scale=std)
                result_text = f"P(X > {b}) = {prob:.4f} ({prob*100:.2f}%)"
            elif val_a and val_b:
                a = float(val_a); b = float(val_b)
                if a >= b:
                    raise ValueError("O primeiro valor deve ser menor que o segundo.")
                prob = st.norm.cdf(b, loc=mean, scale=std) - st.norm.cdf(a, loc=mean, scale=std)
                result_text = f"P({a} < X < {b}) = {prob:.4f} ({prob*100:.2f}%)"
            else:
                result_text = "Preencha um dos campos."
            lbl_result_prob.config(text=result_text)
        except ValueError as e:
            lbl_result_prob.config(text=f"Erro: {e}")
        except Exception as e:
            lbl_result_prob.config(text=f"Erro de cálculo: {e}")

    btn_calculate_prob.config(command=calculate_normal_prob)

    def draw_hist_on(frame, series: pd.Series, title: str):
        clear_frame(frame)
        data = pd.to_numeric(series, errors="coerce").dropna()
        fig = Figure(figsize=(6.5, 3.2), dpi=100)
        ax = fig.add_subplot(111)
        if data.empty:
            ax.text(0.5, 0.5, "Sem dados numéricos válidos.", ha="center", va="center")
        else:
            ax.hist(data, bins=30, edgecolor="black")
            ax.set_title(f"Histograma — {title}")
            ax.set_xlabel(title); ax.set_ylabel("Frequência")
            mean = data.mean(); median = data.median()
            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Média ({mean:.2f})')
            ax.axvline(median, color='orange', linestyle=':', linewidth=2, label=f'Mediana ({median:.2f})')
            ax.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_desc_tab(_evt=None):
        nonlocal current_series_mean, current_series_std
        col = cbx_desc.get()
        if not col:
            return
        text = stats_text(df[col])
        txt_stats.config(state="normal"); txt_stats.delete("1.0", "end"); txt_stats.insert("1.0", f"=== {col} ===\n{text}"); txt_stats.config(state="disabled")
        if pd.api.types.is_numeric_dtype(df[col]):
            draw_hist_on(frm_desc_plot, df[col], col)
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                current_series_mean = s.mean()
                current_series_std = s.std()
                set_prob_calc_state(True)
            else:
                current_series_mean = None; current_series_std = None; set_prob_calc_state(False)
        else:
            clear_frame(frm_desc_plot)
            ttk.Label(frm_desc_plot, text="(Sem histograma: variável não numérica)").pack(pady=12)
            current_series_mean = None; current_series_std = None; set_prob_calc_state(False)

    cbx_desc.bind("<<ComboboxSelected>>", update_desc_tab)

    # =========================
    # Troca de aba
    # =========================
    def on_tab_changed(_evt=None):
        idx = notebook.index(notebook.select())
        tab_id = notebook.tabs()[idx]
        if tab_id == str(tab_corr):
            draw_corr_matrix_on_scroll()
        elif tab_id == str(tab_xy):
            update_xy_plot()
        elif tab_id == str(tab_desc):
            update_desc_tab()

    notebook.bind("<<NotebookTabChanged>>", on_tab_changed)

    # Render inicial
    draw_corr_matrix_on_scroll()
    update_xy_plot()
    update_desc_tab()

    root.mainloop()

# ==========================================================
# 5) ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    try:
        user_src = get_dataset_path_from_user()
        if not user_src:
            raise SystemExit("Encerrado pelo usuário.")
        df = load_dataset_from_input(user_src)
        if df.empty:
            messagebox.showerror("Erro", "O dataset carregado está vazio.")
            raise SystemExit(1)
        build_ui(df)
    except Exception as e:
        # Evita travar no caso de exceção antes da criação do root principal
        try:
            messagebox.showerror("Erro", str(e))
        except Exception:
            print("Erro:", e)
