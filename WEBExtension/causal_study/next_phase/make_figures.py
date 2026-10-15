"""Render an observed-result figure from saved audit records; no model calls."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/projectrl-next-phase-matplotlib')
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    records = json.loads((root / 'transition_artifacts/abstention_reaggregation.json').read_text())
    p = next(row for row in records if row['regime'] == 'P')
    correct = sum(row['reaggregated_em'] == 100 for row in p['question_readouts'])
    if correct != 62 or p['n'] != 300:
        raise ValueError('Figure observations no longer match audited report')
    # Original and broadcast counts are read from the paired baseline artifact.
    contrast = next(c for c in p['comparisons'] if c['before'] == 'P_NoComm' and c['after'] == 'P_Broadcast' and c['n'] == 300)
    cells = contrast['vote_transition']
    original = cells['10']['n'] + cells['11']['n']
    broadcast = cells['01']['n'] + cells['11']['n']
    directory = root / 'figures'
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    bars = ax.bar(['Original initial vote', 'Abstention-aware\ninitial vote', 'Broadcast'],
                  [original, correct, broadcast], color=['#949a9f', '#356986', '#949a9f'], width=.58)
    ax.bar_label(bars, padding=5, fontsize=13)
    ax.set_ylim(0, 75)
    ax.set_ylabel('Exactly correct final answers (of 300 questions)')
    ax.set_xlabel('Readout / communication condition')
    ax.set_title('A stronger zero-call readout changes the comparison', loc='left', pad=18)
    ax.yaxis.grid(True, alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.12, .065, 'Source: saved E5 partitioned traces; same 300 questions.\n'
             'Abstention-aware reaggregation is exploratory and adds no inference calls.',
             fontsize=9, color='#444444')
    fig.subplots_adjust(left=.12, right=.97, top=.85, bottom=.25)
    fig.savefig(directory / 'observed_readout_comparison.png', dpi=180)
    fig.savefig(directory / 'observed_readout_comparison.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
