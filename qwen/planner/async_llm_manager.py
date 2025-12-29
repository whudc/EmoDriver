"""Asynchronous LLM Manager using Multi-Process + Shared Memory"""
import torch
import torch.multiprocessing as mp
import numpy as np
import time
import logging
from multiprocessing import shared_memory
from typing import Optional, Dict, Any
import pickle
import struct

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

logger = logging.getLogger(__name__)


class IterationWrapper:
    def __init__(self, index: int):
        self.index = index

    def __mod__(self, other):
        return self.index % other

    def __eq__(self, other):
        if isinstance(other, IterationWrapper):
            return self.index == other.index
        return self.index == other

    def __int__(self):
        return self.index

    def __repr__(self):
        return f"IterationWrapper({self.index})"


class SharedMemoryBuffer:
    def __init__(self, buffer_size_mb: int = 50):
        self.buffer_size = buffer_size_mb * 1024 * 1024

        self.input_shm = shared_memory.SharedMemory(
            create=True, size=self.buffer_size, name=f'llm_input_{id(self)}'
        )
        self.output_shm = shared_memory.SharedMemory(
            create=True, size=self.buffer_size, name=f'llm_output_{id(self)}'
        )

        self._init_flags()
        logger.info(f"SharedMemory created: {self.input_shm.name}, {self.output_shm.name}")

    def _init_flags(self):
        struct.pack_into('I', self.input_shm.buf, 0, 0)
        struct.pack_into('I', self.output_shm.buf, 0, 0)

    def write_input(self, data: Dict[str, torch.Tensor], timeout: float = 1.0):
        try:
            # Tensor -> NumPy -> pickle
            serialized = pickle.dumps({
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            })

            data_size = len(serialized)
            if data_size + 8 > self.buffer_size:
                raise ValueError(f"Data size {data_size} exceeds buffer size {self.buffer_size}")

            struct.pack_into('I', self.input_shm.buf, 4, data_size)
            self.input_shm.buf[8:8+data_size] = serialized
            struct.pack_into('I', self.input_shm.buf, 0, 1)

        except Exception as e:
            logger.error(f"Error writing to shared memory: {e}")
            raise

    def read_input(self) -> Optional[Dict[str, Any]]:
        try:
            input_ready = struct.unpack_from('I', self.input_shm.buf, 0)[0]
            if not input_ready:
                return None

            data_size = struct.unpack_from('I', self.input_shm.buf, 4)[0]
            serialized = bytes(self.input_shm.buf[8:8+data_size])
            data = pickle.loads(serialized)

            struct.pack_into('I', self.input_shm.buf, 0, 0)
            return data

        except Exception as e:
            logger.error(f"Error reading from shared memory: {e}")
            return None

    def write_output(self, data: Dict[str, Any]):
        try:
            serialized = pickle.dumps(data)
            data_size = len(serialized)
            if data_size + 8 > self.buffer_size:
                raise ValueError(f"Data size {data_size} exceeds buffer size {self.buffer_size}")

            struct.pack_into('I', self.output_shm.buf, 4, data_size)
            self.output_shm.buf[8:8+data_size] = serialized
            struct.pack_into('I', self.output_shm.buf, 0, 1)

        except Exception as e:
            logger.error(f"Error writing output to shared memory: {e}")
            raise

    def read_output(self) -> Optional[Dict[str, Any]]:
        try:
            output_ready = struct.unpack_from('I', self.output_shm.buf, 0)[0]
            if not output_ready:
                return None

            data_size = struct.unpack_from('I', self.output_shm.buf, 4)[0]
            serialized = bytes(self.output_shm.buf[8:8+data_size])
            data = pickle.loads(serialized)
            
            return data

        except Exception as e:
            logger.error(f"Error reading output from shared memory: {e}")
            return None

    def cleanup(self):
        try:
            self.input_shm.close()
            self.input_shm.unlink()
            self.output_shm.close()
            self.output_shm.unlink()
            logger.info("Shared memory cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up shared memory: {e}")


def llm_worker_process(
    input_shm_name: str,
    output_shm_name: str,
    buffer_size: int,
    model_config: Dict[str, Any],
    update_interval: float,
    device: str
):
    try:
        logging.basicConfig(level=logging.INFO, format='[LLM Worker] %(asctime)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.info(f"LLM Worker starting on device {device}")

        # Attach to shared memory created by main process
        input_shm = shared_memory.SharedMemory(name=input_shm_name)
        output_shm = shared_memory.SharedMemory(name=output_shm_name)
        shm_buffer = SharedMemoryBuffer.__new__(SharedMemoryBuffer)
        shm_buffer.input_shm = input_shm
        shm_buffer.output_shm = output_shm
        shm_buffer.buffer_size = buffer_size

        # Load LLM model
        logger.info("Loading LLM model...")
        from qwen.planner.qwen4drive import Qwen2DriveModel
        torch.cuda.set_device(device)
        model_config['devices'] = device
        llm_model = Qwen2DriveModel(model_config)
        logger.info("LLM model loaded successfully")

        # Wait for warmup data from main process
        logger.info("Waiting for initial data from main process...")
        first_input = None
        wait_start = time.time()
        while first_input is None and (time.time() - wait_start) < 5:
            first_input = shm_buffer.read_input()
            if first_input is None:
                time.sleep(0.5)

        # Run warmup inference
        if first_input is not None:
            logger.info("Received initial data, running warmup inference...")
            try:
                features = {
                    k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
                    for k, v in first_input['features'].items()
                }
                ref_path = first_input.get('ref_path', None)
                cur_iter = first_input.get('cur_iter', None)
                if cur_iter is not None and isinstance(cur_iter, int):
                    cur_iter = IterationWrapper(cur_iter)

                warmup_start = time.time()
                with torch.no_grad():
                    output = llm_model.inference(features, ref_path, cur_iter)
                warmup_time = time.time() - warmup_start

                warmup_output = {
                    'predictions': output.predictions,
                    'plan': output.plan.cpu().numpy() if output.plan is not None else None,
                    'llm_plan': output.llm_plan.cpu().numpy() if hasattr(output, 'llm_plan') and output.llm_plan is not None else None,
                    'initialized': True,
                    'iteration': 0,
                    'inference_time': warmup_time,
                    'timestamp': time.time()
                }
                shm_buffer.write_output(warmup_output)
                logger.info(f"Warmup inference completed! (took {warmup_time:.3f}s)")
            except Exception as e:
                logger.error(f"Warmup inference failed: {e}", exc_info=True)
                shm_buffer.write_output({'predictions': None, 'plan': None, 'llm_plan': None, 'initialized': False})
        else:
            logger.error("No input data received within 60s timeout!")
            shm_buffer.write_output({'predictions': None, 'plan': None, 'llm_plan': None, 'initialized': False})

        iteration_count = 1
        last_inference_time = time.time()

        event_driven_mode = (update_interval <= 0)

        if event_driven_mode:
            logger.info("=" * 60)
            logger.info("LLM Worker Mode: EVENT-DRIVEN (immediate use)")
            logger.info("LLM processes new data immediately after inference completes")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info(f"LLM Worker Mode: FIXED INTERVAL ({update_interval}s)")
            logger.info(f"LLM updates every {update_interval} seconds")
            logger.info("=" * 60)

        while True:
            if event_driven_mode:
                input_data = shm_buffer.read_input()

                if input_data is not None:
                    try:
                        # Deserialize and convert to tensors
                        features = {
                            k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
                            for k, v in input_data['features'].items()
                        }
                        ref_path = input_data.get('ref_path', None)
                        cur_iter = input_data.get('cur_iter', None)
                        if cur_iter is not None and isinstance(cur_iter, int):
                            cur_iter = IterationWrapper(cur_iter)
                        # Run inference
                        inference_start = time.time()
                        with torch.no_grad():
                            output = llm_model.inference(features, ref_path, cur_iter)
                        inference_time = time.time() - inference_start
                        current_time = time.time()
                        # Write output immediately
                        output_data = {
                            'predictions': output.predictions,
                            'plan': output.plan.cpu().numpy() if output.plan is not None else None,
                            'llm_plan': output.llm_plan.cpu().numpy() if hasattr(output, 'llm_plan') and output.llm_plan is not None else None,
                            'initialized': True,
                            'iteration': iteration_count,
                            'inference_time': inference_time,
                            'timestamp': current_time
                        }
                        shm_buffer.write_output(output_data)

                        interval_since_last = current_time - last_inference_time
                        iteration_count += 1
                        last_inference_time = current_time

                        logger.info(f"Iteration {iteration_count}: LLM inference {inference_time:.3f}s (interval: {interval_since_last:.3f}s)")

                    except Exception as e:
                        logger.error(f"Error during LLM inference: {e}", exc_info=True)
                        time.sleep(0.05)
                else:
                    time.sleep(0.1)  # No new data, brief sleep

            else:
                # Fixed interval: check time before processing
                current_time = time.time()

                if current_time - last_inference_time >= update_interval:
                    input_data = shm_buffer.read_input()

                    if input_data is not None:
                        try:
                            features = {
                                k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v
                                for k, v in input_data['features'].items()
                            }
                            ref_path = input_data.get('ref_path', None)
                            cur_iter = input_data.get('cur_iter', None)
                            if cur_iter is not None and isinstance(cur_iter, int):
                                cur_iter = IterationWrapper(cur_iter)

                            inference_start = time.time()
                            with torch.no_grad():
                                output = llm_model.inference(features, ref_path, cur_iter)
                            inference_time = time.time() - inference_start

                            output_data = {
                                'predictions': output.predictions,
                                'plan': output.plan.cpu().numpy() if output.plan is not None else None,
                                'llm_plan': output.llm_plan.cpu().numpy() if hasattr(output, 'llm_plan') and output.llm_plan is not None else None,
                                'initialized': True,
                                'iteration': iteration_count,
                                'inference_time': inference_time,
                                'timestamp': current_time
                            }
                            shm_buffer.write_output(output_data)

                            iteration_count += 1
                            last_inference_time = current_time

                            logger.info(f"Iteration {iteration_count}: LLM inference {inference_time:.3f}s")

                        except Exception as e:
                            logger.error(f"Error during LLM inference: {e}", exc_info=True)
                    else:
                        time.sleep(0.01)
                else:
                    # Sleep until next interval
                    sleep_time = max(0.01, update_interval - (current_time - last_inference_time))
                    time.sleep(min(sleep_time, 0.1))

    except KeyboardInterrupt:
        logger.info("LLM Worker received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error in LLM worker: {e}", exc_info=True)
    finally:
        try:
            input_shm.close()
            output_shm.close()
        except:
            pass
        logger.info("LLM Worker shutting down")


class AsyncLLMManager:
    def __init__(
        self,
        model_config: Dict[str, Any],
        update_interval: float = 1.5,
        buffer_size_mb: int = 50,
        llm_device: str = 'cuda:0'
    ):
        self.model_config = model_config
        self.update_interval = update_interval
        self.llm_device = llm_device

        self.shm_buffer = SharedMemoryBuffer(buffer_size_mb=buffer_size_mb)
        self.latest_output = None  
        self.llm_process = None
        self.is_started = False

        logger.info(f"AsyncLLMManager initialized with update_interval={update_interval}s")

    def start(self):
        if self.is_started:
            logger.warning("AsyncLLMManager already started")
            return

        logger.info("Starting LLM worker process...")

        # Create and start worker process
        self.llm_process = mp.Process(
            target=llm_worker_process,
            args=(
                self.shm_buffer.input_shm.name,
                self.shm_buffer.output_shm.name,
                self.shm_buffer.buffer_size,
                self.model_config,
                self.update_interval,
                self.llm_device
            ),
            daemon=False
        )
        self.llm_process.start()
        self.is_started = True
        logger.info(f"LLM worker process started (PID: {self.llm_process.pid})")

        # Wait for initialization (timeout: 60s)
        logger.info("Waiting for LLM to initialize...")
        start_time = time.time()
        while time.time() - start_time < 60:
            output = self.shm_buffer.read_output()
            if output is not None:
                logger.info("LLM initialized successfully")
                break
            time.sleep(0.5)
        else:
            logger.warning("LLM initialization timeout, continuing anyway")

    def update_scene(self, features: Dict[str, torch.Tensor], ref_path=None, cur_iter=None):
        if not self.is_started:
            raise RuntimeError("AsyncLLMManager not started. Call start() first.")

        try:
            input_data = {'features': features, 'ref_path': ref_path, 'cur_iter': cur_iter}
            self.shm_buffer.write_input(input_data)
        except Exception as e:
            logger.error(f"Error updating scene: {e}")

    def get_latest_output(self) -> Optional[Any]:
        if not self.is_started:
            raise RuntimeError("AsyncLLMManager not started. Call start() first.")

        new_output = self.shm_buffer.read_output()
        if new_output is not None:
            self.latest_output = new_output

        return self.latest_output

    def stop(self):
        if not self.is_started:
            return
        logger.info("Stopping AsyncLLMManager...")

        # Terminate worker process
        if self.llm_process is not None and self.llm_process.is_alive():
            pid = self.llm_process.pid
            logger.info(f"Terminating LLM worker (PID: {pid})...")
            self.llm_process.terminate()
            self.llm_process.join(timeout=5)

            if self.llm_process.is_alive():
                logger.warning("LLM process didn't terminate gracefully, killing...")
                self.llm_process.kill()
                self.llm_process.join()

            logger.info(f"LLM worker (PID: {pid}) terminated")

        self.shm_buffer.cleanup()
        self.is_started = False
        logger.info("AsyncLLMManager stopped")

    def __del__(self):
        self.stop()
