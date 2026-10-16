"""Finite, versioned run settings; live release is a separate operation."""
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class RunConfig:
    schema_version: str = 'web_evidence_v1'
    prompt_version: str = 'web_prompt_v1'
    experiment: str = 'W1'
    execution_protocol: str = 'auto'
    seed: int = 20261016
    team_sizes: tuple[int, ...] = (2,3,4,5)
    regime: str = 'P'
    packet_cap: int = 192
    metadata_cap: int = 96
    max_control_frames: int = 6
    max_output_per_call: int = 2048
    max_phases: int = 3
    max_completions: int = 240
    max_attempts: int = 280
    max_input_tokens: int = 1000000
    max_output_tokens: int = 500000
    framing_allowance: int = 128
    max_retries: int = 2
    model: str = 'mock-model'
    deployment_revision: str = 'mock-v1'
    tokenizer_revision: str = 'mock-v1'
    mode: str = 'MOCK'
    replicate_id: str = '0'
    packet_mode: str = 'atomic'
    prompt_variant: str = 'enhanced'
    include_answer: bool = True
    stopping: bool = True
    router_checkpoint: str = ''
    provider_base_url: str = ''

    def to_dict(self):
        return asdict(self)


def validate_config(config):
    if isinstance(config, RunConfig):
        config = config.to_dict()
    if not isinstance(config, dict) or set(config) - {f.name for f in fields(RunConfig)}:
        raise ValueError('unknown_config_field')
    values = dict(config)
    values['team_sizes'] = tuple(values.get('team_sizes', (2,3,4,5)))
    c = RunConfig(**values)
    for name in ('packet_cap','metadata_cap','max_control_frames','max_output_per_call','max_phases',
                 'max_completions','max_attempts','max_input_tokens','max_output_tokens'):
        if type(getattr(c,name)) is not int or getattr(c,name) <= 0:
            raise ValueError('positive_integer_required:' + name)
    if not c.team_sizes or any(type(n) is not int or n not in (2,3,4,5) for n in c.team_sizes):
        raise ValueError('unsupported_team_size')
    if c.mode not in ('MOCK','REAL') or c.regime not in ('P','R') or c.experiment not in ('W1','W2','W3','W4'):
        raise ValueError('unsupported_run_mode')
    if c.max_attempts < c.max_completions or c.max_phases > 3 or c.max_control_frames > 6:
        raise ValueError('inconsistent_ceilings')
    if type(c.max_retries) is not int or not 0 <= c.max_retries <= 2 or type(c.framing_allowance) is not int or c.framing_allowance < 0:
        raise ValueError('invalid_retry_or_framing')
    if c.mode == 'REAL' and (c.model.startswith('mock') or c.deployment_revision.startswith('mock') or c.tokenizer_revision in ('main','mock-v1','')):
        raise ValueError('real_identity_required')
    if c.packet_mode not in ('atomic','prefix') or c.prompt_variant not in ('enhanced','legacy') or type(c.include_answer) is not bool or type(c.stopping) is not bool:
        raise ValueError('invalid_ablation_configuration')
    if c.execution_protocol not in ('auto','fixed','adaptive') or (c.execution_protocol=='adaptive' and c.experiment in ('W1','W3')) or (c.execution_protocol=='fixed' and c.experiment=='W2'):
        raise ValueError('inconsistent_execution_protocol')
    return c


def uses_adaptive(config):
    return config.experiment=='W2' or config.execution_protocol=='adaptive'


def logical_call_ceiling(config,n):
    base=5*n+4*max(1,n//2)+10 if uses_adaptive(config) else 2*n+6
    # W4 stopping effects are paired on the very same initial states within a run.
    return base+(max(1,n//2)+2 if config.experiment=='W4' and uses_adaptive(config) else 0)
