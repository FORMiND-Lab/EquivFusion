import os
import sys
import tkinter as tk
from tkinter import filedialog, scrolledtext
import subprocess
import threading
from pathlib import Path

from util.parser_util import (
    DEFAULT_CASES_DIR,
    DEFAULT_TOOL_DIR,
    DEFAULT_LLVM_DIR,
    DEFAULT_CIRCT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_THREADS,
    DEFAULT_TIMEOUT
)

class TestRunnerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Integration Tests")
        self.root.geometry("1400x900")

        # Configure global font
        self.default_font = ('Microsoft YaHei', 10)
        self.root.option_add("*Font", self.default_font)
        self.padx, self.pady = 10, 8

        self.input_configs = [
            {
                "key": "tool_dir",
                "Label": "TOOL Directory",
                "default": DEFAULT_TOOL_DIR,
                "has_browse": True,
                "command": self.execute_ui_browse_tool_dir
            },
            {
                "key": "llvm_dir",
                "Label": "LLVM Directory",
                "default": DEFAULT_LLVM_DIR,
                "has_browse": True,
                "command": self.execute_ui_browse_llvm_dir
            },
            {
                "key": "circt_dir",
                "Label": "CIRCT Directory",
                "default": DEFAULT_CIRCT_DIR,
                "has_browse": True,
                "command": self.execute_ui_browse_circt_dir
            },
            {
                "key": "log_dir",
                "Label": "Log Directory",
                "default": DEFAULT_LOG_DIR,
                "has_browse": True,
                "command": self.execute_ui_browse_log_dir
            },
            {
                "key": "cases_dir",
                "Label": "Cases Directory",
                "default": DEFAULT_CASES_DIR,
                "has_browse": True,
                "command": self.execute_ui_browse_cases_dir
            },
            {
                "key": "threads",
                "Label": "Threads",
                "default": DEFAULT_THREADS,
                "has_browse": False,
                "command": None
            },
            {
                "key": "timeout",
                "Label": "Timeout",
                "default": DEFAULT_TIMEOUT,
                "has_browse": False,
                "command": None
            }
        ]
        # Init input_vars use default value
        self.input_vars = {
            cfg["key"] : tk.StringVar(value=cfg["default"]) for cfg in self.input_configs
        }

        # Init select_all_var use default value(False)
        self.select_all_var = tk.BooleanVar(value=False)  # Default to deselect all

        # Store cases data
        self.selected_cases = set()     # current selected cases
        self.all_cases = []             # all cases
        self.filtered_cases = []        # current filtered cases

        # Create widgets
        self.current_row = 0
        self.create_widgets()

        # Load cases from default directory on startup
        self.update_ui_cases_listbox()

    def create_widgets(self):
        """Create all the UI components"""

        self.root.grid_columnconfigure(0, weight=1)

        # Create ui inputs frame
        self.create_ui_inputs_frame()
        self.current_row += 1

        # Create Cases List frame
        self.create_ui_cases_list_frame()
        cases_list_row = self.current_row
        self.current_row += 1

        # Run Button
        self.create_ui_run_button()
        self.current_row += 1

        # Output area
        self.create_ui_output_area()
        output_row = self.current_row

        # Configure grid weights
        # 案例列表Frame所在行自动扩展
        self.root.grid_rowconfigure(cases_list_row, weight=2)
        # 输出区域所在行自动扩展
        self.root.grid_rowconfigure(output_row, weight=2)

    def create_ui_inputs_frame(self):
        """Create each input field for the UI (Cases Dir, Tool Dir, etc.)"""
        inputs_frame = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
        inputs_frame.grid(row=self.current_row, column=0, sticky=tk.EW, padx=self.padx, pady=self.pady)
        inputs_frame.grid_columnconfigure(0, weight=0)  # 标签列：宽度固定
        inputs_frame.grid_columnconfigure(1, weight=1)  # 输入框列：自动扩展（占满剩余空间）
        inputs_frame.grid_columnconfigure(2, weight=0)  # 按钮列：宽度固定

        tk.Label(
            inputs_frame,
            text="Configuration",
            font=('Microsoft YaHei', 11, 'bold')
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=0, pady=(0, self.pady))

        input_row = 1
        for cfg in self.input_configs:
            label = tk.Label(inputs_frame, text=cfg["Label"], anchor=tk.E)
            label.grid(row=input_row, column=0, sticky=tk.EW, padx=self.padx, pady=2)

            # 输入框
            entry = tk.Entry(inputs_frame, textvariable=self.input_vars[cfg["key"]])
            entry.grid(row=input_row, column=1, sticky=tk.EW, padx=self.padx, pady=2)

            # 浏览按钮（如有）
            if cfg["has_browse"]:
                browse_button = tk.Button(inputs_frame, text="browse", command=cfg["command"], width=10)
                browse_button.grid(row=input_row, column=2, padx=self.padx, pady=2)

            input_row += 1

    def create_ui_cases_list_frame(self):
        """Cases List Frame include: Available Cases/Selected Cases"""
        # 主Frame：Cases List
        cases_frame = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
        cases_frame.grid(row=self.current_row, column=0, sticky=tk.EW, padx=self.padx, pady=self.pady)
        # 配置Frame内部的列权重，使案例列表随窗口拉伸
        cases_frame.grid_columnconfigure(0, weight=1)  # 所有案例列表列
        cases_frame.grid_columnconfigure(1, weight=1)  # 已选案例列表列
        # 配置Frame内部的行权重，使案例列表区域随窗口拉伸
        cases_frame.grid_rowconfigure(2, weight=1)  # 案例列表所在行

        cases_row = 0
        # 标题
        tk.Label(
            cases_frame,
            text="Cases List",
            font=('Microsoft YaHei', 11, 'bold')
        ).grid(row=cases_row, column=0, columnspan=4, sticky=tk.W, padx=0, pady=(0, self.pady))
        cases_row += 1

        tk.Label(
            cases_frame,
            text="Available Cases",
            font=('Microsoft YaHei', 10, 'bold')
        ).grid(row=cases_row, column=0, sticky=tk.NW, padx=self.padx)
        tk.Label(
            cases_frame,
            text="Selected Cases",
            font=('Microsoft YaHei', 10, 'bold')
        ).grid(row=cases_row, column=1, sticky=tk.NW, padx=self.padx)
        cases_row += 1

        # 所有案例列表
        self.ui_cases_listbox = tk.Listbox(cases_frame, selectmode=tk.EXTENDED, height=10, bg="#f0f0f0")
        self.ui_cases_listbox.grid(row=cases_row, column=0, sticky=tk.NSEW, padx=self.padx, pady=self.pady)
        # 所有案例滚动条
        case_scrollbar = tk.Scrollbar(cases_frame, orient=tk.VERTICAL, command=self.ui_cases_listbox.yview)
        case_scrollbar.grid(row=cases_row, column=0, sticky=tk.NS + tk.E)
        self.ui_cases_listbox.config(yscrollcommand=case_scrollbar.set)
        self.ui_cases_listbox.bind('<<ListboxSelect>>', self.execute_ui_cases_listbox_select)

        # 已选案例标签和列表
        self.ui_selected_cases_listbox = tk.Listbox(cases_frame, height=10, bg="#f0f0f0")
        self.ui_selected_cases_listbox.grid(row=cases_row, column=1, sticky=tk.NSEW, padx=self.padx, pady=self.pady)
        # 已选案例滚动条
        selected_scrollbar = tk.Scrollbar(cases_frame, orient=tk.VERTICAL, command=self.ui_selected_cases_listbox.yview)
        selected_scrollbar.grid(row=cases_row, column=1, sticky=tk.NS + tk.E)
        self.ui_selected_cases_listbox.config(yscrollcommand=selected_scrollbar.set)
        self.ui_selected_cases_listbox.bind('<Double-1>', self.execute_ui_selected_cases_listbox_click)
        cases_row += 1

        # 控制区域（查找 + 全选，同一行，与下方列表对齐）
        control_frame = tk.Frame(cases_frame, bd=1)
        control_frame.grid(row=cases_row, column=0, sticky=tk.EW, padx=self.padx, pady=(2, self.pady))
        # 配置控制区域列权重，确保查找框能扩展
        control_frame.grid_columnconfigure(1, weight=1)  # 输入框列占主要空间

        # 查找标签
        tk.Label(control_frame, text="Find:").grid(row=0, column=0, sticky=tk.W, padx=(0, 2))

        # 查找输入框
        self.ui_find_entry = tk.Entry(control_frame)
        self.ui_find_entry.grid(row=0, column=1, sticky=tk.EW, padx=(0, 10))  # 右侧留空间给全选框
        self.ui_find_entry.bind("<KeyRelease>", self.execute_ui_find)
        self.root.bind("<Escape>", lambda e: self.ui_find_entry.delete(0, tk.END))

        # 全选框（右侧，与案例列表对齐）
        self.ui_select_all_checkbox = tk.Checkbutton(
            control_frame,
            text="Select All",
            variable=self.select_all_var,
            command=self.execute_ui_select_all_checkbox,
            anchor=tk.W
        )
        self.ui_select_all_checkbox.grid(row=0, column=2, sticky=tk.W)

    def create_ui_run_button(self):
        """Create the Run Test button"""
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=self.current_row, column=0, pady=self.pady)

        run_button = tk.Button(
            button_frame,
            text="Run Test",
            command=self.execute_ui_run_button,
            width=12
        )
        run_button.grid(row=0, column=0, ipady=5, ipadx=20, padx=10)

        clear_output_button = tk.Button(
            button_frame,
            text="Clear Output",
            command=self.execute_ui_clear_output_button,
            width=12
        )
        clear_output_button.grid(row=0, column=1, ipady=5, ipadx=20)

    def create_ui_output_area(self):
        """Create the output text area"""
        self.ui_output_text = scrolledtext.ScrolledText(self.root, height=18, wrap=tk.WORD)
        self.ui_output_text.grid(row=self.current_row, column=0, sticky=tk.NSEW, padx=self.padx, pady=self.pady)
        self.ui_output_text.config(state=tk.DISABLED)

    def _browse_directory(self, var_key):
        directory = filedialog.askdirectory()
        if directory:
            self.input_vars[var_key].set(directory)

    def execute_ui_browse_tool_dir(self):
       self._browse_directory("tool_dir")

    def execute_ui_browse_llvm_dir(self):
        self._browse_directory("llvm_dir")

    def execute_ui_browse_circt_dir(self):
        self._browse_directory("circt_dir")

    def execute_ui_browse_log_dir(self):
        self._browse_directory("log_dir")

    def execute_ui_browse_cases_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_vars["cases_dir"].set(directory)
            self.update_ui_cases_listbox()

    def execute_ui_cases_listbox_select(self, event=None):
        current_selected_indices = self.ui_cases_listbox.curselection()
        if not current_selected_indices:
            return

        current_selected_cases = [self.ui_cases_listbox.get(i) for i in current_selected_indices]
        current_selected_count = len(current_selected_cases)

        if current_selected_count > 1:
            for case in current_selected_cases:
                if case not in self.selected_cases:
                    self.selected_cases.add(case)
        else:
            case = current_selected_cases[0]
            if case in self.selected_cases:
                self.selected_cases.remove(case)
            else:
                self.selected_cases.add(case)

        self.update_ui_cases_listbox_selection()
        self.update_ui_selected_cases_listbox()
        self.update_ui_select_all_checkbox()

    def execute_ui_selected_cases_listbox_click(self, event):
        if not self.ui_selected_cases_listbox.curselection():
            return

        current_index = int(self.ui_selected_cases_listbox.curselection()[0])
        current_case = self.ui_selected_cases_listbox.get(current_index)
        self.selected_cases.remove(current_case)
        self.update_ui_cases_listbox_selection()
        self.update_ui_selected_cases_listbox()
        self.update_ui_select_all_checkbox()

    def update_ui_cases_listbox_selection(self):
        self.ui_cases_listbox.selection_clear(0, tk.END)
        for i in range(self.ui_cases_listbox.size()):
            case = self.ui_cases_listbox.get(i)
            if case in self.selected_cases:
                self.ui_cases_listbox.selection_set(i)

    def update_ui_cases_listbox(self):
        cases_dir = self.input_vars["cases_dir"].get()
        if not os.path.isdir(cases_dir):
            return

        self.selected_cases.clear()
        self.update_ui_selected_cases_listbox()
        self.ui_find_entry.delete(0, tk.END)

        self.all_cases.clear()
        for case in os.listdir(cases_dir):
            case_path = os.path.join(cases_dir, case)
            if os.path.isdir(case_path) and not case.startswith("."):
                self.all_cases.append(case)

        self.filtered_cases = self.all_cases.copy()

        self.ui_cases_listbox.delete(0, tk.END)
        self.ui_cases_listbox.insert(tk.END, *self.filtered_cases)
        self.update_ui_cases_listbox_selection()
        self.update_ui_select_all_checkbox()

    def update_ui_selected_cases_listbox(self):
        self.ui_selected_cases_listbox.delete(0, tk.END)
        self.ui_selected_cases_listbox.insert(tk.END, *sorted(self.selected_cases))

    def update_ui_select_all_checkbox(self, event=None):
        total_cases_cnt = len(self.filtered_cases)
        selected_cases_cnt = len([c for c in self.selected_cases if c in self.filtered_cases])

        self.ui_select_all_checkbox.unbind("<Button-1>")
        self.ui_select_all_checkbox.bind("<Button-1>", lambda e: self.execute_ui_select_all_checkbox())
        self.select_all_var.set(total_cases_cnt == selected_cases_cnt)

    def execute_ui_find(self, event=None):
        keyword = self.ui_find_entry.get().strip().lower()
        if not keyword:
            self.filtered_cases = self.all_cases.copy()
        else:
            self.filtered_cases = [case for case in self.all_cases if keyword in case.lower()]

        self.ui_cases_listbox.delete(0, tk.END)
        self.ui_cases_listbox.insert(tk.END, *self.filtered_cases)
        self.update_ui_cases_listbox_selection()
        self.update_ui_select_all_checkbox()

    def execute_ui_select_all_checkbox(self):
        if self.select_all_var.get():
            self.selected_cases.update(self.filtered_cases)
        else:
            self.selected_cases -= set(self.filtered_cases)

        self.update_ui_cases_listbox_selection()
        self.update_ui_selected_cases_listbox()

    def execute_ui_run_button(self):
        test_thread = threading.Thread(target=self.run_tests)
        test_thread.daemon = True
        test_thread.start()

    def execute_ui_clear_output_button(self):
        self.ui_output_text.config(state=tk.NORMAL)
        self.ui_output_text.delete(1.0, tk.END)
        self.ui_output_text.config(state=tk.DISABLED)

    def append_output(self, text):
        self.ui_output_text.config(state=tk.NORMAL)
        self.ui_output_text.insert(tk.END, text)
        self.ui_output_text.see(tk.END)
        self.ui_output_text.config(state=tk.DISABLED)

    def run_tests(self):
        args = [
            "--tool-dir", self.input_vars["tool_dir"].get(),
            "--llvm-dir", self.input_vars["llvm_dir"].get(),
            "--circt-dir", self.input_vars["circt_dir"].get(),
            "--cases-dir", self.input_vars["cases_dir"].get(),
            "--log-dir", self.input_vars['log_dir'].get(),
            "--threads", self.input_vars['threads'].get(),
            "--timeout", self.input_vars['timeout'].get()
        ]
        selected_cases = list(self.selected_cases)
        args.extend(["--cases"] + selected_cases)

        main_py_path = Path(__file__).parent / "main.py"
        cmd = [sys.executable, main_py_path] + args

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in process.stdout:
                self.append_output(line)

            process.wait()

        except Exception as e:
            self.append_output(f"Error during execution: {str(e)}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = TestRunnerGUI(root)
    root.mainloop()
