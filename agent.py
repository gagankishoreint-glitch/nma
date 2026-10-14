"""Text-only multi-session agent experiment; simulator, never real-world actions.

Modes are explicit. Scripted transport is plumbing validation only. The genuine
llama.cpp experiment must pass real-model stability probes before memory tasks.
"""
import argparse
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import torch

from memory.encoder_bridge import EncoderBridge, HashEncoder
from memory.orchestrator import MemoryOrchestrator
from memory.procedures import ProcedureStore
from memory.titans_memory import NeuralLongTermMemory
from llm.scripted import ScriptedLLM
from prompts.context import ROOT, build_messages
from prompts.procedure_extraction import extract_procedure
from src.data.restaurant_task import (RestaurantWorld, Outcome, GOALS,
                                      INTENTS, MISSING_SLOTS, parse_action)
from src.eval.agent_stability import not_run, run_stability_suite
from src.utils.agent_profile import ResourceProfiler, environment

METHODS = ['no_memory', 'context_window', 'neural_only', 'full_frozen_memory', 'full']


def print_mission():
    print((ROOT/'MISSION.md').read_text(), flush=True)


def parameter_hash(memory):
    digest = hashlib.sha256()
    for name, value in memory.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class NativeMemoryAgent:
    def __init__(self, llm, method, encoder=None, memory=None, procedures=None,
                 context_turns=4, print_scope=True):
        if print_scope:
            print_mission()
        if method not in METHODS or context_turns < 0:
            raise ValueError('Unknown method or invalid context size')
        self.llm, self.method = llm, method
        self.history = deque(maxlen=context_turns)
        self.procedures = procedures or ProcedureStore()
        self.memory = None
        if method in {'full', 'neural_only', 'full_frozen_memory'}:
            if encoder is None:
                raise ValueError('A neural method requires an independent encoder')
            self.memory = MemoryOrchestrator(encoder, memory,
                                             update_enabled=method != 'full_frozen_memory')

    def run_session(self, session, executor, profiler, seed, max_steps=2):
        if max_steps < 1:
            raise ValueError('At least one step required')
        goal, intent, slots = GOALS[session], INTENTS[session], MISSING_SLOTS[session]
        observation = 'A new session has started. No prior task facts are included in this observation.'
        records, current_log = [], []  # session-local; never implicitly replayed
        private_values = []
        completed = False
        for step in range(max_steps):
            start = time.perf_counter()
            recalled = self.memory.recall(slots) if self.memory else []
            proc = self.procedures.match(intent) if self.method in {'full', 'full_frozen_memory'} else None
            history = list(self.history) if self.method == 'context_window' else []
            system, user = build_messages(goal, proc, recalled, observation, history)
            try:
                raw = self.llm.generate(system, user, max_tokens=128, temperature=0.)
            except Exception as error:
                # Transport/context-limit failures are visible failed actions,
                # never silently retried with a different prompt or model.
                raw = json.dumps({'transport_error': f'{type(error).__name__}: {error}'})
            try:
                action = parse_action(raw)
                outcome = executor.execute(session, action)
            except (ValueError, TypeError) as error:
                # Invalid actions are never executed and cannot count as success.
                outcome = Outcome(f'Rejected malformed action: {error}', terminal=True)
            # Remember verified TOOL observations, not the model's assertions.
            surprises = {}
            if self.memory:
                if outcome.slots:
                    surprises['slot_batch'] = self.memory.observe_slots(outcome.slots)
                else:
                    surprises['reconstruction_only'] = self.memory.observe(outcome.observation)
            private_values.extend(outcome.slots.values())
            turn = {'goal': goal, 'action': raw, 'observation': outcome.observation,
                    'succeeded': outcome.succeeded}
            current_log.append(turn)
            if self.method == 'context_window':
                self.history.append(turn)
            profile = profiler.snapshot(f'{self.method}:seed{seed}:session{session+1}:step{step}')
            records.append({'backend': self.llm.backend, 'method': self.method, 'seed': seed,
                            'session': session+1, 'step': step, 'goal': goal,
                            'observation_before': observation, 'recalled_memories': json.dumps(recalled),
                            'procedure_used': json.dumps(proc), 'action': raw,
                            'tool_observation': outcome.observation, 'succeeded': outcome.succeeded,
                            'surprise': json.dumps(surprises),
                            'memory_updates': self.memory.memory.update_count if self.memory else 0,
                            'peak_device_vram_mib': profile['peak_device_used_mib'],
                            'peak_process_rss_mib': profile['peak_process_rss_mib'],
                            'elapsed_seconds': time.perf_counter()-start})
            if outcome.terminal:
                completed = outcome.succeeded
                break
            observation = outcome.observation
        if completed and self.method in {'full', 'full_frozen_memory'}:
            proc = extract_procedure(self.llm, json.dumps(current_log), succeeded=True,
                                     forbidden_values=private_values, expected_domain=intent)
            if 'error' not in proc:
                self.procedures.store(proc, forbidden_values=private_values)
            else:
                records[-1]['extraction_error'] = json.dumps(proc)
            profiler.snapshot(f'{self.method}:post_extraction')
        return completed, records


def warmup_procedure(llm):
    """Separate successful simulator episode, not the evaluation restaurant."""
    world = RestaurantWorld(seed=999999)
    found = world.execute(0, {'tool': 'find_restaurant', 'arguments': {'city': 'Beijing'}})
    booked = world.execute(1, {'tool': 'book_table', 'arguments': {'name': world.name}})
    log = [{'action': 'find_restaurant', 'observation': found.observation},
           {'action': 'book_table', 'observation': booked.observation}]
    return extract_procedure(llm, json.dumps(log), succeeded=found.succeeded and booked.succeeded,
                             forbidden_values=[world.name, world.address], expected_domain='restaurant_booking')


def run_benchmark(llm, encoder, profiler, output, seeds, context_turns=4, max_steps=2,
                  stability_report=None):
    if llm.backend != 'scripted' and (not stability_report or
            stability_report.get('backend') != 'llama_cpp' or
            stability_report.get('status') != 'passed' or
            stability_report.get('all_passed') is not True):
        raise RuntimeError('Real memory tasks require a passed real-model stability report')
    rows, summaries, hashes = [], [], []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    warmup = warmup_procedure(llm)
    (output/'warmup_extraction.json').write_text(json.dumps(warmup, indent=2))
    exported = ProcedureStore()
    try:
        for seed in seeds:
            for method in METHODS:
                procedures = ProcedureStore()
                if method in {'full', 'full_frozen_memory'} and 'error' not in warmup:
                    procedures.store(warmup)
                # Every condition receives independent parameters and the same world.
                memory = (NeuralLongTermMemory(input_dim=encoder.dim, seed=seed)
                          if method in {'full', 'neural_only', 'full_frozen_memory'} else None)
                before = parameter_hash(memory) if memory is not None else None
                agent = NativeMemoryAgent(llm, method, encoder, memory, procedures,
                                          context_turns, print_scope=False)
                profiler.snapshot(f'{method}:seed{seed}:agent_ready')
                world = RestaurantWorld(seed)
                successes = []
                for session in range(3):
                    success, records = agent.run_session(session, world, profiler, seed, max_steps)
                    rows.extend(records)
                    successes.append(success)
                after = parameter_hash(memory) if memory is not None else None
                hashes.append({'method': method, 'seed': seed, 'before': before, 'after': after,
                               'updates': memory.update_count if memory is not None else 0})
                if method == 'full_frozen_memory' and before != after:
                    raise AssertionError('Frozen-memory ablation changed weights')
                summaries.append({'backend': llm.backend, 'method': method, 'seed': seed,
                                  'session1_success': successes[0], 'session2_success': successes[1],
                                  'session3_address_recall': successes[2]})
                if method == 'full':
                    exported = agent.procedures
    finally:
        for filename, values in [('agent_log.csv', rows), ('summary.csv', summaries)]:
            if not values:
                continue
            fields = list(dict.fromkeys(key for row in values for key in row))
            with open(output/filename, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fields, lineterminator='\n')
                writer.writeheader()
                writer.writerows(values)
        (output/'memory_hashes.json').write_text(json.dumps(hashes, indent=2))
        exported.export(output/'PROCEDURES.md', llm.backend)
    return summaries


def main():
    print_mission()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['scripted', 'llama_cpp'], default='scripted')
    parser.add_argument('--model', help='Local chat GGUF path; never downloaded automatically')
    parser.add_argument('--encoder', help='Local sentence-transformer directory')
    parser.add_argument('--output', default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--n-ctx', type=int, default=1024)
    parser.add_argument('--n-gpu-layers', type=int, default=0)
    parser.add_argument('--context-turns', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=2)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.context_turns < 0 or args.max_steps < 1 or args.n_gpu_layers < 0 or args.n_ctx < 512:
        parser.error('Use unique seeds, nonnegative context/offload, positive steps, context >=512')
    output = Path(args.output or ('results/agent_smoke' if args.backend == 'scripted' else 'results/agent_real'))
    output.mkdir(parents=True, exist_ok=True)
    for name in ['agent_log.csv', 'summary.csv', 'memory_hashes.json', 'PROCEDURES.md', 'warmup_extraction.json']:
        (output/name).unlink(missing_ok=True)
    env = environment()
    print(json.dumps({'environment': env}), flush=True)
    (output/'environment.json').write_text(json.dumps(env, indent=2))
    profiler = ResourceProfiler().start()
    llm = None
    start = time.perf_counter()
    status = not_run('No real-model stability probes have run.', args.backend)
    manifest = {'backend': args.backend, 'config': vars(args), 'environment': env,
                'python': platform.python_version(), 'torch': torch.__version__,
                'scope': 'scripted plumbing only' if args.backend == 'scripted' else 'synthetic real-LLM evaluation',
                'status': 'started'}
    exit_code = 0
    try:
        torch.set_num_threads(1)
        profiler.snapshot('environment')
        if args.backend == 'scripted':
            print('SCRIPTED PLUMBING ONLY: no real LLM; behavioral stability NOT RUN.', flush=True)
            llm = ScriptedLLM()
            encoder = HashEncoder()
            status = not_run('Scripted transport is not an LLM; no behavioral stability claim.', 'scripted')
        else:
            if not args.model or not args.encoder:
                raise ValueError('Real mode requires --model LOCAL_GGUF and --encoder LOCAL_SENTENCE_TRANSFORMER_DIR')
            if not Path(args.model).is_file() or not Path(args.encoder).is_dir():
                raise FileNotFoundError('Provide an existing GGUF file and a local encoder directory')
            manifest['gguf_sha256'] = file_digest(args.model)
            manifest['encoder_file_sha256'] = {str(p.relative_to(args.encoder)): file_digest(p)
                                               for p in sorted(Path(args.encoder).rglob('*'))
                                               if p.is_file() and p.suffix in {'.json', '.txt', '.safetensors', '.bin'}}
            if args.n_gpu_layers > 0 and profiler.snapshot('pre_offload_check')['gpu_device_used_mib'] is None:
                raise RuntimeError('GPU offload requires working nvidia-smi telemetry; use --n-gpu-layers 0')
            from llm.stable_llm import StableLLM
            llm = StableLLM(args.model, args.n_ctx, args.n_gpu_layers)
            profiler.snapshot('llm_loaded')
            status = run_stability_suite(llm, profiler)
            if not status['all_passed']:
                raise RuntimeError('Real-model stability gate failed; memory benchmark NOT RUN')
            encoder = EncoderBridge(args.encoder, local_files_only=True)
        profiler.snapshot('encoder_loaded')
        manifest['encoder_backend'] = encoder.backend
        manifest['stability_gate'] = 'not_applicable: scripted plumbing' if args.backend == 'scripted' else 'passed'
        summaries = run_benchmark(llm, encoder, profiler, output, args.seeds,
                                   args.context_turns, args.max_steps,
                                   stability_report=status)
        manifest['status'] = 'completed'
        manifest['memory_parameters'] = sum(p.numel() for p in NeuralLongTermMemory(input_dim=encoder.dim).parameters())
        for method in METHODS:
            subset = [x for x in summaries if x['method'] == method]
            print(f'{method}: session3 address recall = {sum(x["session3_address_recall"] for x in subset)}/{len(subset)}')
    except Exception as error:
        manifest.update(status='blocked_or_failed', error=f'{type(error).__name__}: {error}')
        if status['status'] == 'not_run':
            status['reason'] = manifest['error']
        print(manifest['error'], flush=True)
        exit_code = 2
    finally:
        if llm is not None and hasattr(llm, 'close'):
            llm.close()
        profiler.close()
        manifest.update(wall_seconds=time.perf_counter()-start, resources=profiler.report())
        manifest['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for directory in ['memory', 'llm', 'prompts']
                                     for path in sorted((ROOT/directory).rglob('*'))
                                     if path.is_file() and path.suffix in {'.py', '.txt'}}
        for relative in ['agent.py', 'MISSION.md', 'requirements-agent.txt', 'scripts/run_agent.sh',
                         'src/data/restaurant_task.py', 'src/eval/agent_stability.py', 'src/utils/agent_profile.py']:
            manifest['source_sha256'][relative] = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        (output/'stability_tests.json').write_text(json.dumps(status, indent=2))
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        # Canonical artifact aliases always have their provenance next to them.
        # Never overwrite successful agent artifacts with a failed run's CSV.
        (ROOT/'results/stability_tests.json').write_text(json.dumps(status, indent=2))
        (ROOT/'results/agent_attempt_manifest.json').write_text(json.dumps(manifest, indent=2))
        if manifest['status'] == 'completed':
            (ROOT/'results/agent_manifest.json').write_text(json.dumps(manifest, indent=2))
            for source, target in [(output/'agent_log.csv', ROOT/'results/agent_log.csv'),
                                   (output/'PROCEDURES.md', ROOT/'PROCEDURES.md')]:
                if source.resolve() != target.resolve():
                    shutil.copyfile(source, target)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
