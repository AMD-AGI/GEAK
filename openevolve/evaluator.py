# Modifications Copyright(C)[2025] Advanced Micro Devices, Inc. All rights reserved.
# https://github.com/algorithmicsuperintelligence/openevolve  - Apache License 2.0

"""
Evaluation system for OpenEvolve
"""

import asyncio
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import traceback

from openevolve.config import EvaluatorConfig
from openevolve.database import ProgramDatabase
from openevolve.evaluation_result import EvaluationResult
from openevolve.database import ProgramDatabase
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.utils.async_utils import TaskPool, run_in_executor
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.utils.other_utils import get_time_spent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU Pool -- ensures no two concurrent evaluations share a GPU
# ---------------------------------------------------------------------------

class GPUPool:
    """
    Thread-safe pool of GPU devices for exclusive access during evaluation.

    When a candidate program needs to be benchmarked, it acquires a GPU from
    the pool.  No other evaluation can use that GPU until it is released.
    This prevents GPU contention that would corrupt latency measurements.

    Usage:
        pool = GPUPool(gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7])
        gpu_id = pool.acquire()       # blocks until a GPU is free
        try:
            ... run evaluation on gpu_id ...
        finally:
            pool.release(gpu_id)
    """

    def __init__(self, gpu_ids: Optional[List[int]] = None):
        """
        Args:
            gpu_ids: List of GPU device IDs to manage.
                     If None or empty, auto-detects available GPUs.
        """
        if gpu_ids:
            self._gpu_ids = list(gpu_ids)
        else:
            self._gpu_ids = self._detect_gpus()

        self._semaphore = threading.Semaphore(len(self._gpu_ids))
        self._lock = threading.Lock()
        self._available = list(self._gpu_ids)
        logger.info(
            f"GPUPool initialized with {len(self._gpu_ids)} GPUs: {self._gpu_ids}"
        )

    @staticmethod
    def _detect_gpus() -> List[int]:
        """Auto-detect available GPUs via rocm-smi or torch."""
        # Try rocm-smi first (works even when HIP_VISIBLE_DEVICES is restricted)
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                import re
                gpu_ids = sorted(set(
                    int(m.group(1))
                    for m in re.finditer(r"GPU\[(\d+)\]", result.stdout)
                ))
                if gpu_ids:
                    return gpu_ids
        except Exception:
            pass

        # Fallback: try torch
        try:
            import torch
            count = torch.cuda.device_count()
            if count > 0:
                return list(range(count))
        except Exception:
            pass

        # Last resort: single GPU 0
        return [0]

    @property
    def num_gpus(self) -> int:
        return len(self._gpu_ids)

    @property
    def gpu_ids(self) -> List[int]:
        return list(self._gpu_ids)

    def acquire(self, timeout: float = 600) -> int:
        """
        Acquire exclusive access to a GPU.  Blocks until one is free.

        Returns:
            GPU device ID
        """
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(
                f"Could not acquire a GPU within {timeout}s "
                f"(all {len(self._gpu_ids)} GPUs are busy)"
            )
        with self._lock:
            gpu_id = self._available.pop(0)
        logger.debug(f"Acquired GPU {gpu_id} (remaining: {self._available})")
        return gpu_id

    def release(self, gpu_id: int) -> None:
        """Release a GPU back to the pool."""
        with self._lock:
            if gpu_id not in self._available:
                self._available.append(gpu_id)
        self._semaphore.release()
        logger.debug(f"Released GPU {gpu_id} (available: {self._available})")


class Evaluator:
    """
    Evaluates programs and assigns scores

    The evaluator is responsible for executing programs, measuring their performance,
    and assigning scores based on the evaluation criteria.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        evaluation_file: str,
        initial_program_name: str,
        llm_ensemble: Optional[LLMEnsemble] = None,
        prompt_sampler: Optional[PromptSampler] = None,
        database: Optional[ProgramDatabase] = None,
        kernel_evaluator_path: Optional[str] = None,
        gpu_ids: Optional[List[int]] = None,
    ):
        self.config = config
        self.eval_dir = config.eval_dir or tempfile.gettempdir()
        self.evaluation_file = evaluation_file
        self.llm_ensemble = llm_ensemble
        self.prompt_sampler = prompt_sampler
        self.database = database
        self.kernel_evaluator_path = kernel_evaluator_path
        self.initial_program_name = initial_program_name
        if not os.path.exists(kernel_evaluator_path): 
            print(f"⚠️ WARNING: Kernel evaluator path {kernel_evaluator_path} does not exist, using default given path.")

        # GPU pool for exclusive GPU access during benchmarking.
        # If gpu_ids is provided, use those; otherwise auto-detect.
        self.gpu_pool = GPUPool(gpu_ids=gpu_ids)

        # Limit parallel evaluations to number of available GPUs so that
        # no two evaluations ever share a GPU for latency measurement.
        effective_parallel = min(
            config.parallel_evaluations, self.gpu_pool.num_gpus
        )
        if effective_parallel != config.parallel_evaluations:
            logger.info(
                f"Capping parallel_evaluations from {config.parallel_evaluations} "
                f"to {effective_parallel} (= number of available GPUs)"
            )
        # Create a task pool for parallel evaluation
        self.task_pool = TaskPool(max_concurrency=effective_parallel)

        # Set up evaluation function if file exists
        self._load_evaluation_function()

        # Pending artifacts storage for programs
        self._pending_artifacts: Dict[str, Dict[str, Union[str, bytes]]] = {}

        logger.info(
            f"Initialized evaluator with {evaluation_file} "
            f"(parallel={effective_parallel}, gpus={self.gpu_pool.gpu_ids})"
        )

    def _load_evaluation_function(self) -> None:
        """Load the evaluation function from the evaluation file"""
        if not os.path.exists(self.evaluation_file):
            raise ValueError(f"Evaluation file {self.evaluation_file} not found")

        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for local imports")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load spec from {self.evaluation_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["evaluation_module"] = module
            spec.loader.exec_module(module)

            if not hasattr(module, "evaluate"):
                raise AttributeError(
                    f"Evaluation file {self.evaluation_file} does not contain an 'evaluate' function"
                )

            self.evaluate_function = module.evaluate
            logger.info(f"Successfully loaded evaluation function from {self.evaluation_file}")
        except Exception as e:
            logger.error(f"Error loading evaluation function: {str(e)}")
            raise

    async def evaluate_program(
        self,
        program_code: str,
        program_id: str = "",
        gpu_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a program and return scores.

        GPU isolation: acquires an exclusive GPU from the pool before
        benchmarking.  No two concurrent evaluations will share a GPU.
        The assigned GPU ID is written to .gpu_id in the temp dir so
        that the evaluation function (e.g., geak_eval_evaluator.py)
        can read it and pass it to the CommandmentEvaluator.

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging
            gpu_id: Hint for GPU ID (ignored -- pool assignment is used)

        Returns:
            Dictionary of metric name to score
        """
        start_time = time.time()
        program_id_str = f" {program_id}" if program_id else ""

        # Acquire exclusive GPU from the pool (blocks if all GPUs busy)
        assigned_gpu = self.gpu_pool.acquire()
        logger.info(
            f"Program{program_id_str}: acquired GPU {assigned_gpu} for evaluation"
        )

        # Check if artifacts are enabled
        artifacts_enabled = os.environ.get("ENABLE_ARTIFACTS", "true").lower() == "true"

        # Retry logic for evaluation
        last_exception = None
        try:
          for attempt in range(self.config.max_retries + 1):
            # Create temp directory - compatible with Python 3.11
            temp_dir = tempfile.mkdtemp(dir=self.eval_dir)
            try:
                # Support multi-file programs: if program_code is a JSON-encoded
                # dict of {filepath: content}, write all files.  Otherwise write
                # a single file as before (backward compatible).
                temp_file_path = os.path.join(temp_dir, self.initial_program_name)
                is_multifile = False
                try:
                    parsed = json.loads(program_code)
                    if isinstance(parsed, dict) and all(
                        isinstance(v, str) for v in parsed.values()
                    ):
                        is_multifile = True
                        for rel_path, content in parsed.items():
                            fpath = os.path.join(temp_dir, rel_path)
                            os.makedirs(os.path.dirname(fpath), exist_ok=True)
                            with open(fpath, "w", encoding="utf-8") as f:
                                f.write(content)
                                f.flush()
                                os.fsync(f.fileno())
                            os.chmod(fpath, 0o777)
                        # Set temp_file_path to the main file (initial_program_name)
                        temp_file_path = os.path.join(temp_dir, self.initial_program_name)
                        if not os.path.exists(temp_file_path):
                            # Fall back to first file if main file not in dict
                            temp_file_path = os.path.join(
                                temp_dir, list(parsed.keys())[0]
                            )
                except (json.JSONDecodeError, TypeError):
                    pass

                if not is_multifile:
                    with open(temp_file_path, "w", encoding="utf-8") as temp_file:
                        temp_file.write(program_code)
                        temp_file.flush()
                        os.fsync(temp_file.fileno())
                    os.chmod(temp_file_path, 0o777)

                os.chmod(temp_dir, 0o777)

                # Write .gpu_id file so the evaluation function knows which
                # GPU to use (thread-safe: each eval has its own temp_dir)
                gpu_id_path = os.path.join(temp_dir, ".gpu_id")
                with open(gpu_id_path, "w") as gf:
                    gf.write(str(assigned_gpu))

            except Exception as e:
                # Clean up on error
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise

            assert os.path.exists(temp_file_path), f"Temporary file {temp_file_path} was not created!"
            try:
                start_time = time.time()
                # Run evaluation
                if self.config.cascade_evaluation:
                    # Run cascade evaluation
                    result = await self._cascade_evaluate(
                                                            program_text = temp_file_path,
                                                            test_suite_path = self.config.test_suite_path,
                                                            ref_wrapper_path= self.config.ref_wrapper_path, 
                                                            wrapper_fn_name= self.config.wrapper_fn_name, 
                                                            unit_tests_path= self.config.unit_tests_path, 
                                                            n_warmup= self.config.n_warmup, 
                                                            n_iters= self.config.n_iters, 
                                                            atol= self.config.atol, 
                                                            rtol= self.config.rtol, 
                                                            verbose= self.config.verbose, 
                                                        )
                else:
                    # Run direct evaluation
                    result = await self._direct_evaluate(
                                                            program_text = temp_file_path,
                                                            test_suite_path = self.config.test_suite_path,
                                                            ref_wrapper_path= self.config.ref_wrapper_path, 
                                                            wrapper_fn_name= self.config.wrapper_fn_name, 
                                                            unit_tests_path= self.config.unit_tests_path, 
                                                            n_warmup= self.config.n_warmup, 
                                                            n_iters= self.config.n_iters, 
                                                            atol= self.config.atol, 
                                                            rtol= self.config.rtol, 
                                                            verbose= self.config.verbose,     
                                                        )

                # Process the result based on type
                eval_result = self._process_evaluation_result(result)
                logger.info(f"Time spent in evaluation: {get_time_spent(start_time)}")
                # Check if this was a timeout and capture artifacts if enabled
                if artifacts_enabled and program_id and eval_result.metrics.get("timeout") is True:
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    self._pending_artifacts[program_id].update(
                        {
                            "timeout": True,
                            "timeout_duration": self.config.timeout,
                            "failure_stage": "evaluation",
                            "error_type": "timeout",
                        }
                    )

                # Add LLM feedback if configured
                llm_eval_result = None
                analysis = None
                if self.config.use_llm_feedback and self.llm_ensemble:
                    start_time = time.time()
                    llm_result = await self._llm_evaluate(program_code, program_id=program_id)
                    logger.info(f"Time spent in LLM evaluation: {get_time_spent(start_time)}")
                    llm_eval_result = self._process_evaluation_result(llm_result)

                    analysis = llm_eval_result.artifacts.get("reasoning", None) if eval_result.metrics.get("success", 0.0) > 0 else eval_result.metrics.get("error", None)
                    # Combine metrics
                    for name, value in llm_result.metrics.items():
                        if isinstance(value, (int, float)):
                            # Scale LLM metrics by the configured weight
                            value = float(value)
                            eval_result.metrics[f"llm_{name}"] = value * self.config.llm_feedback_weight if eval_result.metrics.get("success", 0.0) > 0 else 0
                        else:
                            eval_result.metrics[f"llm_{name}"] = value
                            logger.warning(f"LLM metric {name} is not numeric: {value}")

                # Store artifacts if enabled and present
                if (
                    artifacts_enabled
                    and (
                        eval_result.has_artifacts()
                        or (llm_eval_result and llm_eval_result.has_artifacts())
                    )
                    and program_id
                ):
                    if program_id not in self._pending_artifacts:
                        self._pending_artifacts[program_id] = {}

                    # Merge eval_result artifacts with llm artifacts if they exist
                    if eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned artifacts: "
                            f"{eval_result.artifacts}"
                        )

                    if llm_eval_result and llm_eval_result.has_artifacts():
                        self._pending_artifacts[program_id].update(llm_eval_result.artifacts)
                        logger.debug(
                            f"Program{program_id_str} returned LLM artifacts: "
                            f"{llm_eval_result.artifacts}"
                        )

                elapsed = time.time() - start_time
                logger.info(
                    f"Evaluated program{program_id_str} in {elapsed:.2f}s: "
                    f"{format_metrics_safe(eval_result.metrics)}"
                )

                # Return just metrics for backward compatibility
                return eval_result.metrics, analysis

            except asyncio.TimeoutError:
                # Handle timeout specially - don't retry, just return timeout result
                logger.warning(f"Evaluation timed out after {self.config.timeout}s")

                # Capture timeout artifacts if enabled
                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "timeout": True,
                        "timeout_duration": self.config.timeout,
                        "failure_stage": "evaluation",
                        "error_type": "timeout",
                    }

                return {"error": 0.0, "timeout": True}, None

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Evaluation attempt {attempt + 1}/{self.config.max_retries + 1} failed for program{program_id_str}: {str(e)}"
                )
                traceback.print_exc()

                # Capture failure artifacts if enabled
                if artifacts_enabled and program_id:
                    self._pending_artifacts[program_id] = {
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        "failure_stage": "evaluation",
                        "attempt": attempt + 1,
                    }

                # If this is not the last attempt, wait a bit before retrying
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)  # Wait 1 second before retry

            finally:
                # # Clean up temporary file
                # if os.path.exists(temp_file_path):
                #     os.unlink(temp_file_path)
                #     ## remove the temporary directory
                #     temp_dir = os.path.dirname(temp_file_path)
                #     try:
                #         os.rmdir(temp_dir)
                #     except OSError as e:
                #         logger.warning(f"Failed to remove temporary directory {temp_dir}: {str(e)}")
                pass

          # All retries failed
          logger.error(
              f"All evaluation attempts failed for program{program_id_str}. Last error: {str(last_exception)}"
          )
          return {"error": 0.0}, None
        finally:
            # ALWAYS release the GPU back to the pool, even on failure/timeout
            self.gpu_pool.release(assigned_gpu)
            logger.debug(f"Program{program_id_str}: released GPU {assigned_gpu}")

    def _process_evaluation_result(self, result: Any) -> EvaluationResult:
        """
        Process evaluation result to handle both dict and EvaluationResult returns

        Args:
            result: Raw result from evaluation function

        Returns:
            EvaluationResult instance
        """
        if isinstance(result, dict):
            # Backward compatibility - wrap dict in EvaluationResult
            return EvaluationResult.from_dict(result)
        elif isinstance(result, EvaluationResult):
            # New format - use directly
            return result
        else:
            # Error case - return error metrics
            logger.warning(f"Unexpected evaluation result type: {type(result)}")
            return EvaluationResult(metrics={"error": 0.0})

    def get_pending_artifacts(self, program_id: str) -> Optional[Dict[str, Union[str, bytes]]]:
        """
        Get and clear pending artifacts for a program

        Args:
            program_id: Program ID

        Returns:
            Artifacts dictionary or None if not found
        """
        return self._pending_artifacts.pop(program_id, None)

    async def _direct_evaluate(
        self,  program_text, test_suite_path, ref_wrapper_path, wrapper_fn_name, unit_tests_path, n_warmup, n_iters, atol, rtol, verbose,                       
        ) -> Dict[str, float]:
        """
        Directly evaluate a program using the evaluation function with timeout

        Args:
            program_path: Path to the program file

        Returns:
            Dictionary of metric name to score

        Raises:
            asyncio.TimeoutError: If evaluation exceeds timeout
            Exception: If evaluation function raises an exception
        """

        # Create a coroutine that runs the evaluation function in an executor
        async def run_evaluation():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.evaluate_function,
                                              test_suite_path,
                                              program_text,
                                              ref_wrapper_path,
                                              wrapper_fn_name,
                                              unit_tests_path,
                                              n_warmup,
                                              n_iters,
                                              atol,
                                              rtol,
                                              verbose
                                              )

        # Run the evaluation with timeout - let exceptions bubble up for retry handling
        result = await asyncio.wait_for(run_evaluation(), timeout=self.config.timeout)

        # Validate result
        if not isinstance(result, dict):
            logger.warning(f"Evaluation returned non-dictionary result: {result}")
            return {"error": 0.0}

        return result

    async def _cascade_evaluate(
        self,  program_text, test_suite_path, ref_wrapper_path, wrapper_fn_name, unit_tests_path, n_warmup, n_iters, atol, rtol, verbose,
    ) -> Union[Dict[str, float], EvaluationResult]:
        """
        Run cascade evaluation with increasingly challenging test cases

        Args:
            program_path: Path to the program file

        Returns:
            Dictionary of metrics or EvaluationResult with metrics and artifacts
        """
        # Import the evaluation module to get cascade functions if they exist
        try:
            # Add the evaluation file's directory to Python path so it can import local modules
            eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
                logger.debug(f"Added {eval_dir} to Python path for cascade evaluation")

            spec = importlib.util.spec_from_file_location("evaluation_module", self.evaluation_file)
            if spec is None or spec.loader is None:
                return await self._direct_evaluate(
                    program_text,
                    test_suite_path,
                    ref_wrapper_path,
                    wrapper_fn_name,
                    unit_tests_path,
                    n_warmup,
                    n_iters,
                    atol,
                    rtol,
                    verbose,
                )

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Check if cascade functions exist
            if not hasattr(module, "evaluate_stage1"):
                return await self._direct_evaluate(
                    program_text,
                    test_suite_path,
                    ref_wrapper_path,
                    wrapper_fn_name,
                    unit_tests_path,
                    n_warmup,
                    n_iters,
                    atol,
                    rtol,
                    verbose,
                )

            # Run first stage with timeout
            try:

                async def run_stage1():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage1,
                                                       program_text,
                                                       test_suite_path,
                                                       ref_wrapper_path,
                                                       wrapper_fn_name,
                                                       unit_tests_path,
                                                       n_warmup,
                                                       n_iters,
                                                       atol,
                                                       rtol,
                                                       verbose,
                                                   ) 

                stage1_result = await asyncio.wait_for(run_stage1(), timeout=self.config.timeout)
                stage1_eval_result = self._process_evaluation_result(stage1_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 1 evaluation timed out after {self.config.timeout}s")
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0, "timeout": True},
                    artifacts={
                        "failure_stage": "stage1",
                        "timeout": True,
                    },
                )
            except Exception as e:
                logger.error(f"Error in stage 1 evaluation: {str(e)}")
                # Capture stage 1 failure as artifacts
                return EvaluationResult(
                    metrics={"stage1_passed": 0.0, "error": 0.0},
                    artifacts={
                        "stderr": str(e),
                        "traceback": traceback.format_exc(),
                        "failure_stage": "stage1",
                    },
                )

            # Check threshold
            if not self._passes_threshold(
                stage1_eval_result.metrics, self.config.cascade_thresholds[0]
            ):
                return stage1_eval_result

            # Check if second stage exists
            if not hasattr(module, "evaluate_stage2"):
                return stage1_eval_result

            # Run second stage with timeout
            try:

                async def run_stage2():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage2,
                                                       program_text,
                                                       test_suite_path,
                                                       ref_wrapper_path,
                                                       wrapper_fn_name,
                                                       unit_tests_path,
                                                       n_warmup,
                                                       n_iters,
                                                       atol,
                                                       rtol,
                                                       verbose,
                                                    )

                stage2_result = await asyncio.wait_for(run_stage2(), timeout=self.config.timeout)
                stage2_eval_result = self._process_evaluation_result(stage2_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 2 evaluation timed out after {self.config.timeout}s")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_timeout": True,
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                stage1_eval_result.metrics["timeout"] = True
                return stage1_eval_result
            except Exception as e:
                logger.error(f"Error in stage 2 evaluation: {str(e)}")
                # Capture stage 2 failure, but keep stage 1 results
                stage1_eval_result.artifacts.update(
                    {
                        "stage2_stderr": str(e),
                        "stage2_traceback": traceback.format_exc(),
                        "failure_stage": "stage2",
                    }
                )
                stage1_eval_result.metrics["stage2_passed"] = 0.0
                return stage1_eval_result

            # Merge results from stage 1 and 2
            merged_metrics = {}
            # Convert all values to float to avoid type errors
            for name, value in stage1_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            for name, value in stage2_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_metrics[name] = float(value)

            # Merge artifacts
            merged_artifacts = {}
            merged_artifacts.update(stage1_eval_result.artifacts)
            merged_artifacts.update(stage2_eval_result.artifacts)

            merged_result = EvaluationResult(metrics=merged_metrics, artifacts=merged_artifacts)

            # Check threshold for stage 3
            if len(self.config.cascade_thresholds) < 2 or not self._passes_threshold(
                merged_result.metrics, self.config.cascade_thresholds[1]
            ):
                return merged_result

            # Check if third stage exists
            if not hasattr(module, "evaluate_stage3"):
                return merged_result

            # Run third stage with timeout
            try:

                async def run_stage3():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, module.evaluate_stage3,
                                                       program_text,
                                                       test_suite_path,
                                                       ref_wrapper_path,
                                                       wrapper_fn_name,
                                                       unit_tests_path,
                                                       n_warmup,
                                                       n_iters,
                                                       atol,
                                                       rtol,
                                                       verbose,
                                                       )

                stage3_result = await asyncio.wait_for(run_stage3(), timeout=self.config.timeout)
                stage3_eval_result = self._process_evaluation_result(stage3_result)
            except asyncio.TimeoutError:
                logger.warning(f"Stage 3 evaluation timed out after {self.config.timeout}s")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_timeout": True,
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                merged_result.metrics["timeout"] = True
                return merged_result
            except Exception as e:
                logger.error(f"Error in stage 3 evaluation: {str(e)}")
                # Capture stage 3 failure, but keep previous results
                merged_result.artifacts.update(
                    {
                        "stage3_stderr": str(e),
                        "stage3_traceback": traceback.format_exc(),
                        "failure_stage": "stage3",
                    }
                )
                merged_result.metrics["stage3_passed"] = 0.0
                return merged_result

            # Merge stage 3 results
            for name, value in stage3_eval_result.metrics.items():
                if isinstance(value, (int, float)) and name != "error":
                    merged_result.metrics[name] = float(value)

            merged_result.artifacts.update(stage3_eval_result.artifacts)

            return merged_result

        except Exception as e:
            logger.error(f"Error in cascade evaluation: {str(e)}")
            # Return proper cascade failure result instead of re-raising
            return EvaluationResult(
                metrics={"stage1_passed": 0.0, "error": 0.0},
                artifacts={
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                    "failure_stage": "cascade_setup",
                },
            )

    async def _llm_evaluate(self, program_code: str, program_id: str = "") -> Dict[str, float]:
        """
        Use LLM to evaluate code quality

        Args:
            program_code: Code to evaluate
            program_id: Optional ID for logging

        Returns:
            Dictionary of metric name to score
        """
        if not self.llm_ensemble:
            return {}

        try:
            # Create prompt for LLM
            prompt = self.prompt_sampler.build_prompt(
                current_program=program_code, template_key="evaluation"
            )

            # Get LLM response
            responses = await self.llm_ensemble.generate_all_with_context(
                prompt["system"], [{"role": "user", "content": prompt["user"]}]
            )

            # Log prompt and response to database
            if self.database and program_id:
                self.database.log_prompt(
                    program_id=program_id,
                    template_key="evaluation",
                    prompt=prompt,
                    responses=responses,
                )

            # Extract JSON from response
            try:
                # Try to find JSON block
                json_pattern = r"```json\n(.*?)\n```"
                import re
                
                artifacts = {}
                avg_metrics = {}
                for i, response in enumerate(responses):
                    json_match = re.search(json_pattern, response, re.DOTALL)

                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to extract JSON directly
                        json_str = response
                        # Remove non-JSON parts
                        start_idx = json_str.find("{")
                        end_idx = json_str.rfind("}") + 1
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = json_str[start_idx:end_idx]

                    # Parse JSON
                    result = json.loads(json_str)

                    # All non-numeric values are artifacts, all numeric values are metrics
                    metrics = {}
                    for key, value in result.items():
                        if not isinstance(value, (int, float)):
                            artifacts[key] = value
                        else:
                            metrics[key] = float(value)

                    # Weight of the model in the ensemble
                    weight = self.llm_ensemble.weights[i] if self.llm_ensemble.weights else 1.0

                    # Average the metrics
                    for name, value in metrics.items():
                        if name in avg_metrics:
                            avg_metrics[name] += value * weight
                        else:
                            avg_metrics[name] = value * weight

                return EvaluationResult(
                    metrics=avg_metrics,
                    artifacts=artifacts,
                )

            except Exception as e:
                logger.warning(f"Error parsing LLM response during feedback generation: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"Error in LLM evaluation during feedback generation: {str(e)}")
            traceback.print_exc()
            return {}

    def _passes_threshold(self, metrics: Dict[str, float], threshold: float) -> bool:
        """
        Check if metrics pass a threshold

        Args:
            metrics: Dictionary of metric name to score
            threshold: Threshold to pass

        Returns:
            True if metrics pass threshold
        """
        if not metrics:
            return False

        # Calculate average score, skipping non-numeric values and 'error' key
        valid_metrics = []
        for name, value in metrics.items():
            # Skip 'error' keys and ensure values are numeric
            if name != "error" and isinstance(value, (int, float)):
                try:
                    valid_metrics.append(float(value))
                except (TypeError, ValueError):
                    logger.warning(f"Skipping non-numeric metric: {name}={value}")
                    continue

        if not valid_metrics:
            return False

        avg_score = sum(valid_metrics) / len(valid_metrics)
        return avg_score >= threshold

    async def evaluate_multiple(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple programs in parallel, each on its own exclusive GPU.

        The GPU assignment is handled by the GPUPool inside evaluate_program(),
        which acquires an exclusive GPU before benchmarking and releases it
        after.  The task pool concurrency is capped to the number of available
        GPUs, so no two evaluations ever share a GPU.

        Args:
            programs: List of (program_code, program_id) tuples

        Returns:
            List of metric dictionaries
        """
        tasks = [
            self.task_pool.create_task(self.evaluate_program, program_code, program_id)
            for program_code, program_id in programs
        ]

        return await asyncio.gather(*tasks)
