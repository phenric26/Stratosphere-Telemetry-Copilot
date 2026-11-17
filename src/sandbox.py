"""
Sandbox de Execução Segura de Código Científico (Nó 3)
Executa scripts Python gerados em um ambiente isolado com Pandas, NumPy, SciPy e Plotly.
Implementa análise AST de segurança, timeout, captura de stdout, extração de objetos (fig, metrics)
e formatação detalhada de tracebacks para auto-correção (Self-Healing Loop).
"""

import ast
import io
import math
import sys
import time
import traceback
import threading
from typing import Dict, Any, Optional, Tuple
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy


FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "requests", "urllib",
    "pathlib", "ctypes", "multiprocessing", "pty", "commands", "builtins"
}

FORBIDDEN_CALLS = {
    "eval", "exec", "__import__", "compile", "open", "input",
    "breakpoint", "globals", "locals", "memoryview"
}


class SecurityViolationError(Exception):
    pass


class SafeAstValidator(ast.NodeVisitor):
    def __init__(self):
        self.violations = []

    def visit_Import(self, node):
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod in FORBIDDEN_MODULES:
                self.violations.append(f"Importação proibida: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod in FORBIDDEN_MODULES:
                self.violations.append(f"Importação proibida: 'from {node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                self.violations.append(f"Chamada de função proibida: '{node.func.id}()'")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in ["system", "popen", "spawn", "rmdir", "remove", "unlink"]:
                self.violations.append(f"Chamada de método de risco: '.{node.func.attr}()'")
        self.generic_visit(node)


class ExecutionResult:
    def __init__(
        self,
        success: bool,
        fig: Optional[go.Figure] = None,
        metrics: Optional[Dict[str, Any]] = None,
        stdout: str = "",
        error: Optional[str] = None,
        error_type: Optional[str] = None,
        traceback_str: Optional[str] = None,
        execution_time_ms: float = 0.0,
        raw_code: str = ""
    ):
        self.success = success
        self.fig = fig
        self.metrics = metrics or {}
        self.stdout = stdout
        self.error = error
        self.error_type = error_type
        self.traceback_str = traceback_str
        self.execution_time_ms = execution_time_ms
        self.raw_code = raw_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "has_figure": self.fig is not None,
            "metrics": self.metrics,
            "stdout": self.stdout,
            "error": self.error,
            "error_type": self.error_type,
            "execution_time_ms": self.execution_time_ms
        }


class SafePythonSandbox:
    def __init__(self, timeout_sec: float = 5.0):
        self.timeout_sec = timeout_sec

    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} na linha {e.lineno}"

        validator = SafeAstValidator()
        validator.visit(tree)
        if validator.violations:
            return False, f"Violação de Segurança: {'; '.join(validator.violations)}"

        return True, None

    def execute(self, code: str, df: pd.DataFrame) -> ExecutionResult:
        t_start = time.perf_counter()

        # 1. Validação estática AST
        is_safe, sec_err = self.validate_code(code)
        if not is_safe:
            return ExecutionResult(
                success=False,
                error=sec_err,
                error_type="SecurityOrSyntaxError",
                traceback_str=sec_err,
                execution_time_ms=(time.perf_counter() - t_start) * 1000,
                raw_code=code
            )

        # 2. Ambiente controlado
        # Cria cópia rasa para evitar mutação indesejada da base principal
        local_scope = {
            "df": df.copy(),
            "pd": pd,
            "np": np,
            "scipy": scipy,
            "px": px,
            "go": go,
            "math": math,
            "datetime": datetime,
            "fig": None,
            "metrics": {}
        }

        # Redireciona stdout
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout

        exec_exception = []
        exec_traceback = []

        def target():
            try:
                sys.stdout = stdout_capture
                exec(code, local_scope)
            except Exception as ex:
                exec_exception.append(ex)
                exec_traceback.append(traceback.format_exc())
            finally:
                sys.stdout = old_stdout

        th = threading.Thread(target=target)
        th.start()
        th.join(timeout=self.timeout_sec)

        sys.stdout = old_stdout
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if th.is_alive():
            return ExecutionResult(
                success=False,
                error=f"TimeoutError: O script excedeu o limite máximo de {self.timeout_sec}s.",
                error_type="TimeoutError",
                traceback_str="Execution timed out.",
                execution_time_ms=elapsed_ms,
                raw_code=code
            )

        if exec_exception:
            ex = exec_exception[0]
            tb = exec_traceback[0]
            return ExecutionResult(
                success=False,
                error=f"{type(ex).__name__}: {str(ex)}",
                error_type=type(ex).__name__,
                traceback_str=tb,
                stdout=stdout_capture.getvalue(),
                execution_time_ms=elapsed_ms,
                raw_code=code
            )

        # 3. Extração dos artefatos produzidos
        fig_obj = local_scope.get("fig")
        # Se fig não foi setada, mas existe um gráfico em outra variável
        if not isinstance(fig_obj, (go.Figure,)):
            for v in local_scope.values():
                if isinstance(v, go.Figure):
                    fig_obj = v
                    break

        metrics_dict = local_scope.get("metrics")
        if not isinstance(metrics_dict, dict):
            metrics_dict = {}

        # Coleta quaisquer variáveis numéricas escalares úteis como metrics
        for k, v in local_scope.items():
            if k not in ["df", "pd", "np", "scipy", "px", "go", "math", "datetime", "fig", "metrics"]:
                if isinstance(v, (int, float, str, bool)) and not k.startswith("_"):
                    if isinstance(v, float) and not np.isnan(v):
                        metrics_dict[k] = round(v, 4)
                    else:
                        metrics_dict[k] = v

        return ExecutionResult(
            success=True,
            fig=fig_obj,
            metrics=metrics_dict,
            stdout=stdout_capture.getvalue(),
            execution_time_ms=elapsed_ms,
            raw_code=code
        )
